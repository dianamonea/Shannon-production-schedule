package worldmodel

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Pose represents a 3D pose with quaternion orientation
type Pose struct {
	X, Y, Z float64
	Qw, Qx, Qy, Qz float64
}

// Velocity represents 3D linear velocity
type Velocity struct {
	Vx, Vy, Vz float64
}

// ObjectState tracks a dynamic object in the environment
type ObjectState struct {
	ObjectID   string
	Pose       Pose
	Velocity   Velocity
	Covariance [3][3]float64 // position covariance only
	UpdatedAt  time.Time
}

// SensorObservation is a timestamped observation of an object
type SensorObservation struct {
	ObjectID   string
	Pose       Pose
	Covariance [3][3]float64
	ObservedAt time.Time
}

// OccupancyGrid models a 2D occupancy grid for workspace
// Each cell holds occupancy probability (0.0-1.0)
type OccupancyGrid struct {
	Resolution float64 // meters per cell
	Width      int
	Height     int
	OriginX    float64
	OriginY    float64
	Grid       [][]float64
	LastUpdate [][]time.Time
}

// NewOccupancyGrid creates a new occupancy grid
func NewOccupancyGrid(width, height int, resolution, originX, originY float64) *OccupancyGrid {
	grid := make([][]float64, height)
	last := make([][]time.Time, height)
	for i := 0; i < height; i++ {
		grid[i] = make([]float64, width)
		last[i] = make([]time.Time, width)
	}

	return &OccupancyGrid{
		Resolution: resolution,
		Width:      width,
		Height:     height,
		OriginX:    originX,
		OriginY:    originY,
		Grid:       grid,
		LastUpdate: last,
	}
}

// UpdateCell updates occupancy probability for a world position
func (og *OccupancyGrid) UpdateCell(x, y, occupancy float64) error {
	ix, iy, err := og.worldToGrid(x, y)
	if err != nil {
		return err
	}

	if occupancy < 0.0 {
		occupancy = 0.0
	} else if occupancy > 1.0 {
		occupancy = 1.0
	}

	og.Grid[iy][ix] = occupancy
	og.LastUpdate[iy][ix] = time.Now()
	return nil
}

// IsOccupied checks occupancy at a world position
func (og *OccupancyGrid) IsOccupied(x, y float64, threshold float64) (bool, error) {
	ix, iy, err := og.worldToGrid(x, y)
	if err != nil {
		return false, err
	}

	return og.Grid[iy][ix] >= threshold, nil
}

// ClearStale clears cells not updated within ttl
func (og *OccupancyGrid) ClearStale(ttl time.Duration) {
	cutoff := time.Now().Add(-ttl)
	for y := 0; y < og.Height; y++ {
		for x := 0; x < og.Width; x++ {
			if !og.LastUpdate[y][x].IsZero() && og.LastUpdate[y][x].Before(cutoff) {
				og.Grid[y][x] = 0.0
			}
		}
	}
}

func (og *OccupancyGrid) worldToGrid(x, y float64) (int, int, error) {
	ix := int(math.Floor((x - og.OriginX) / og.Resolution))
	iy := int(math.Floor((y - og.OriginY) / og.Resolution))

	if ix < 0 || iy < 0 || ix >= og.Width || iy >= og.Height {
		return 0, 0, fmt.Errorf("position out of grid bounds")
	}
	return ix, iy, nil
}

// WorldModel maintains a dynamic representation of the environment
// including objects, occupancy grid, and simple collision prediction
//
// Inspired by modern robotics world modeling practices but implemented
// with minimal dependencies for portability.
type WorldModel struct {
	mu sync.RWMutex

	objects map[string]*ObjectState
	grid    *OccupancyGrid
	surfaces map[string]*SurfacePatch

	maxObjectAge time.Duration
}

// CollisionRisk represents a predicted collision between two objects
type CollisionRisk struct {
	ObjectA  string
	ObjectB  string
	Distance float64
}

// NewWorldModel creates a new world model
func NewWorldModel(grid *OccupancyGrid, maxObjectAge time.Duration) *WorldModel {
	return &WorldModel{
		objects:      make(map[string]*ObjectState),
		grid:         grid,
		surfaces:     make(map[string]*SurfacePatch),
		maxObjectAge: maxObjectAge,
	}
}

// GetGrid returns the occupancy grid (may be nil if not configured)
func (wm *WorldModel) GetGrid() *OccupancyGrid {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return wm.grid
}

// SurfacePatch represents a planar surface for affordance reasoning
type SurfacePatch struct {
	SurfaceID   string
	Center      Pose
	Normal      [3]float64
	Area        float64
	Friction    float64
	UpdatedAt   time.Time
}

// AddOrUpdateSurface registers a surface patch
func (wm *WorldModel) AddOrUpdateSurface(surface *SurfacePatch) {
	if surface == nil {
		return
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.surfaces[surface.SurfaceID] = surface
}

// HasSurfaces returns true if any surface patches are available
func (wm *WorldModel) HasSurfaces() bool {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return len(wm.surfaces) > 0
}

// IsPlacementStable checks if a pose is stable on a known surface
// Returns (stable, surfaceID)
func (wm *WorldModel) IsPlacementStable(pose Pose, maxTiltDeg float64) (bool, string) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	for id, surface := range wm.surfaces {
		dist := distance3D(pose.X, pose.Y, pose.Z, surface.Center.X, surface.Center.Y, surface.Center.Z)
		if dist > math.Sqrt(surface.Area)/2.0 {
			continue
		}

		// Tilt check: assume Z-axis up, normal close to [0,0,1]
		angle := angleBetween(surface.Normal, [3]float64{0, 0, 1})
		if angle <= maxTiltDeg {
			return true, id
		}
	}

	return false, ""
}

// AffordanceScore returns a score (0-1) for placing at pose on known surfaces
func (wm *WorldModel) AffordanceScore(pose Pose, maxTiltDeg float64) (float64, string) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	bestScore := 0.0
	bestSurface := ""
	for id, surface := range wm.surfaces {
		dist := distance3D(pose.X, pose.Y, pose.Z, surface.Center.X, surface.Center.Y, surface.Center.Z)
		if dist > math.Sqrt(surface.Area)/2.0 {
			continue
		}

		angle := angleBetween(surface.Normal, [3]float64{0, 0, 1})
		if angle > maxTiltDeg {
			continue
		}
		stability := 1.0 - (angle / maxTiltDeg)
		score := stability * surface.Friction
		if score > bestScore {
			bestScore = score
			bestSurface = id
		}
	}

	return bestScore, bestSurface
}

// UpdateObject updates object state using a simple Kalman-like fusion
func (wm *WorldModel) UpdateObject(obs SensorObservation) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	state, exists := wm.objects[obs.ObjectID]
	if !exists {
		wm.objects[obs.ObjectID] = &ObjectState{
			ObjectID:   obs.ObjectID,
			Pose:       obs.Pose,
			Velocity:   Velocity{},
			Covariance: obs.Covariance,
			UpdatedAt:  obs.ObservedAt,
		}
		return
	}

	dt := obs.ObservedAt.Sub(state.UpdatedAt).Seconds()
	if dt > 0 {
		vx := (obs.Pose.X - state.Pose.X) / dt
		vy := (obs.Pose.Y - state.Pose.Y) / dt
		vz := (obs.Pose.Z - state.Pose.Z) / dt
		state.Velocity = Velocity{Vx: vx, Vy: vy, Vz: vz}
	}

	// Weighted update based on covariance (simple fusion)
	state.Pose.X = blend(state.Pose.X, obs.Pose.X, state.Covariance[0][0], obs.Covariance[0][0])
	state.Pose.Y = blend(state.Pose.Y, obs.Pose.Y, state.Covariance[1][1], obs.Covariance[1][1])
	state.Pose.Z = blend(state.Pose.Z, obs.Pose.Z, state.Covariance[2][2], obs.Covariance[2][2])
	state.Pose.Qw, state.Pose.Qx, state.Pose.Qy, state.Pose.Qz = obs.Pose.Qw, obs.Pose.Qx, obs.Pose.Qy, obs.Pose.Qz

	state.Covariance = obs.Covariance
	state.UpdatedAt = obs.ObservedAt
}

// UpdateObjects批量更新对象观测
func (wm *WorldModel) UpdateObjects(observations []SensorObservation) {
	for _, obs := range observations {
		wm.UpdateObject(obs)
	}
}

// PredictObject predicts object pose at a future time (constant velocity)
func (wm *WorldModel) PredictObject(objectID string, at time.Time) (Pose, bool) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	state, ok := wm.objects[objectID]
	if !ok {
		return Pose{}, false
	}
	if wm.isStale(state.UpdatedAt) {
		return Pose{}, false
	}

	dt := at.Sub(state.UpdatedAt).Seconds()
	pred := Pose{
		X:  state.Pose.X + state.Velocity.Vx*dt,
		Y:  state.Pose.Y + state.Velocity.Vy*dt,
		Z:  state.Pose.Z + state.Velocity.Vz*dt,
		Qw: state.Pose.Qw,
		Qx: state.Pose.Qx,
		Qy: state.Pose.Qy,
		Qz: state.Pose.Qz,
	}

	return pred, true
}

// DetectCollisionRisk checks if two objects are likely to collide within horizon
func (wm *WorldModel) DetectCollisionRisk(aID, bID string, horizon time.Duration, threshold float64) (bool, float64) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	a, aok := wm.objects[aID]
	b, bok := wm.objects[bID]
	if !aok || !bok {
		return false, math.Inf(1)
	}
	if wm.isStale(a.UpdatedAt) || wm.isStale(b.UpdatedAt) {
		return false, math.Inf(1)
	}

	dt := horizon.Seconds()
	ax := a.Pose.X + a.Velocity.Vx*dt
	ay := a.Pose.Y + a.Velocity.Vy*dt
	az := a.Pose.Z + a.Velocity.Vz*dt
	bx := b.Pose.X + b.Velocity.Vx*dt
	by := b.Pose.Y + b.Velocity.Vy*dt
	bz := b.Pose.Z + b.Velocity.Vz*dt

	dist := distance3D(ax, ay, az, bx, by, bz)
	return dist <= threshold, dist
}

// DetectCollisionRisks checks all pairs for collision risks
func (wm *WorldModel) DetectCollisionRisks(objectIDs []string, horizon time.Duration, threshold float64) []CollisionRisk {
	risks := make([]CollisionRisk, 0)
	for i := 0; i < len(objectIDs); i++ {
		for j := i + 1; j < len(objectIDs); j++ {
			risk, dist := wm.DetectCollisionRisk(objectIDs[i], objectIDs[j], horizon, threshold)
			if risk {
				risks = append(risks, CollisionRisk{
					ObjectA:  objectIDs[i],
					ObjectB:  objectIDs[j],
					Distance: dist,
				})
			}
		}
	}
	return risks
}

// PruneStaleObjects removes objects not updated within maxObjectAge
func (wm *WorldModel) PruneStaleObjects() int {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	cutoff := time.Now().Add(-wm.maxObjectAge)
	removed := 0
	for id, obj := range wm.objects {
		if obj.UpdatedAt.Before(cutoff) {
			delete(wm.objects, id)
			removed++
		}
	}
	return removed
}

// GetObjectConfidence returns confidence based on freshness (0.0-1.0)
func (wm *WorldModel) GetObjectConfidence(objectID string) float64 {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	state, ok := wm.objects[objectID]
	if !ok {
		return 0.0
	}

	if wm.maxObjectAge <= 0 {
		return 1.0
	}

	age := time.Since(state.UpdatedAt)
	if age <= 0 {
		return 1.0
	}
	if age >= wm.maxObjectAge {
		return 0.0
	}
	return 1.0 - (age.Seconds() / wm.maxObjectAge.Seconds())
}

// GetActiveObjectIDs returns non-stale object IDs
func (wm *WorldModel) GetActiveObjectIDs() []string {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	ids := make([]string, 0, len(wm.objects))
	for id, obj := range wm.objects {
		if !wm.isStale(obj.UpdatedAt) {
			ids = append(ids, id)
		}
	}
	return ids
}

// UpdateOccupancy updates occupancy grid from a list of poses
func (wm *WorldModel) UpdateOccupancy(poses []Pose, occupancy float64) {
	if wm.grid == nil {
		return
	}

	for _, p := range poses {
		_ = wm.grid.UpdateCell(p.X, p.Y, occupancy)
	}
}

// ClearStaleOccupancy clears stale occupancy cells using TTL
func (wm *WorldModel) ClearStaleOccupancy(ttl time.Duration) {
	if wm.grid == nil {
		return
	}
	wm.grid.ClearStale(ttl)
}

// GetObject returns a copy of object state if present
func (wm *WorldModel) GetObject(objectID string) (ObjectState, bool) {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	state, ok := wm.objects[objectID]
	if !ok {
		return ObjectState{}, false
	}

	return *state, true
}

func blend(prev, obs, prevVar, obsVar float64) float64 {
	if prevVar+obsVar == 0 {
		return obs
	}
	k := prevVar / (prevVar + obsVar)
	return prev + k*(obs-prev)
}

func distance3D(ax, ay, az, bx, by, bz float64) float64 {
	dx := ax - bx
	dy := ay - by
	dz := az - bz
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func (wm *WorldModel) isStale(updatedAt time.Time) bool {
	if wm.maxObjectAge <= 0 {
		return false
	}
	return time.Since(updatedAt) > wm.maxObjectAge
}

func angleBetween(a, b [3]float64) float64 {
	dot := a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
	na := math.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
	nb := math.Sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])
	if na == 0 || nb == 0 {
		return 180.0
	}
	cos := dot / (na * nb)
	if cos > 1 {
		cos = 1
	} else if cos < -1 {
		cos = -1
	}
	return math.Acos(cos) * 180.0 / math.Pi
}
