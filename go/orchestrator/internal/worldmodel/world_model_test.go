package worldmodel

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOccupancyGrid_UpdateAndQuery(t *testing.T) {
	grid := NewOccupancyGrid(10, 10, 1.0, 0.0, 0.0)

	err := grid.UpdateCell(2.2, 3.7, 0.8)
	assert.NoError(t, err)

	occupied, err := grid.IsOccupied(2.2, 3.7, 0.5)
	assert.NoError(t, err)
	assert.True(t, occupied)

	occupied, err = grid.IsOccupied(2.2, 3.7, 0.9)
	assert.NoError(t, err)
	assert.False(t, occupied)
}

func TestOccupancyGrid_ClearStale(t *testing.T) {
	grid := NewOccupancyGrid(5, 5, 1.0, 0.0, 0.0)

	err := grid.UpdateCell(1.0, 1.0, 0.9)
	assert.NoError(t, err)

	grid.LastUpdate[1][1] = time.Now().Add(-2 * time.Minute)
	grid.ClearStale(30 * time.Second)

	occupied, err := grid.IsOccupied(1.0, 1.0, 0.1)
	assert.NoError(t, err)
	assert.False(t, occupied)
}

func TestWorldModel_UpdatePredict(t *testing.T) {
	wm := NewWorldModel(NewOccupancyGrid(10, 10, 1.0, 0.0, 0.0), 5*time.Minute)

	obs1 := SensorObservation{
		ObjectID: "OBJ_1",
		Pose:     Pose{X: 1.0, Y: 1.0, Z: 0.0, Qw: 1.0},
		Covariance: [3][3]float64{{0.2, 0, 0}, {0, 0.2, 0}, {0, 0, 0.2}},
		ObservedAt: time.Now().Add(-2 * time.Second),
	}
	wm.UpdateObject(obs1)

	obs2 := SensorObservation{
		ObjectID: "OBJ_1",
		Pose:     Pose{X: 3.0, Y: 1.0, Z: 0.0, Qw: 1.0},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	}
	wm.UpdateObject(obs2)

	pred, ok := wm.PredictObject("OBJ_1", time.Now().Add(1*time.Second))
	assert.True(t, ok)
	assert.Greater(t, pred.X, 3.0)
}

func TestWorldModel_CollisionRisk(t *testing.T) {
	wm := NewWorldModel(nil, 5*time.Minute)

	wm.UpdateObject(SensorObservation{
		ObjectID: "A",
		Pose:     Pose{X: 0, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now().Add(-1 * time.Second),
	})
	wm.UpdateObject(SensorObservation{
		ObjectID: "A",
		Pose:     Pose{X: 1, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})

	wm.UpdateObject(SensorObservation{
		ObjectID: "B",
		Pose:     Pose{X: 3, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})

	risk, dist := wm.DetectCollisionRisk("A", "B", 2*time.Second, 1.5)
	assert.True(t, risk)
	assert.LessOrEqual(t, dist, 1.5)
}

func TestWorldModel_DetectCollisionRisks(t *testing.T) {
	wm := NewWorldModel(nil, 5*time.Minute)

	wm.UpdateObject(SensorObservation{
		ObjectID: "A",
		Pose:     Pose{X: 0, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now().Add(-1 * time.Second),
	})
	wm.UpdateObject(SensorObservation{
		ObjectID: "A",
		Pose:     Pose{X: 1, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})
	wm.UpdateObject(SensorObservation{
		ObjectID: "B",
		Pose:     Pose{X: 2, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})
	wm.UpdateObject(SensorObservation{
		ObjectID: "C",
		Pose:     Pose{X: 10, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})

	risks := wm.DetectCollisionRisks([]string{"A", "B", "C"}, 2*time.Second, 1.5)
	assert.Len(t, risks, 1)
	assert.Equal(t, "A", risks[0].ObjectA)
	assert.Equal(t, "B", risks[0].ObjectB)
}

func TestWorldModel_ConfidenceAndStaleCollision(t *testing.T) {
	wm := NewWorldModel(nil, 2*time.Second)

	wm.UpdateObject(SensorObservation{
		ObjectID: "STALE",
		Pose:     Pose{X: 0, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now().Add(-5 * time.Second),
	})
	wm.UpdateObject(SensorObservation{
		ObjectID: "FRESH",
		Pose:     Pose{X: 1, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})

	conf := wm.GetObjectConfidence("STALE")
	assert.Equal(t, 0.0, conf)

	risk, _ := wm.DetectCollisionRisk("STALE", "FRESH", 2*time.Second, 2.0)
	assert.False(t, risk)
}

func TestWorldModel_Prune(t *testing.T) {
	wm := NewWorldModel(nil, 1*time.Second)

	wm.UpdateObject(SensorObservation{
		ObjectID: "OLD",
		Pose:     Pose{X: 0, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now().Add(-10 * time.Second),
	})

	removed := wm.PruneStaleObjects()
	assert.Equal(t, 1, removed)
}

func TestWorldModel_PlacementStability(t *testing.T) {
	wm := NewWorldModel(nil, 5*time.Minute)

	wm.AddOrUpdateSurface(&SurfacePatch{
		SurfaceID: "TABLE_1",
		Center:    Pose{X: 0, Y: 0, Z: 0.75, Qw: 1},
		Normal:    [3]float64{0, 0, 1},
		Area:      1.0,
		Friction:  0.6,
		UpdatedAt: time.Now(),
	})

	stable, surfaceID := wm.IsPlacementStable(Pose{X: 0.1, Y: 0.1, Z: 0.75, Qw: 1}, 10)
	assert.True(t, stable)
	assert.Equal(t, "TABLE_1", surfaceID)
}

func TestWorldModel_AffordanceScore(t *testing.T) {
	wm := NewWorldModel(nil, 5*time.Minute)

	wm.AddOrUpdateSurface(&SurfacePatch{
		SurfaceID: "FIXTURE_1",
		Center:    Pose{X: 0, Y: 0, Z: 0.5, Qw: 1},
		Normal:    [3]float64{0, 0, 1},
		Area:      1.0,
		Friction:  0.8,
		UpdatedAt: time.Now(),
	})

	score, surfaceID := wm.AffordanceScore(Pose{X: 0.1, Y: 0.1, Z: 0.5, Qw: 1}, 15)
	assert.Greater(t, score, 0.0)
	assert.Equal(t, "FIXTURE_1", surfaceID)
}

func TestWorldModel_ClearStaleOccupancy(t *testing.T) {
	grid := NewOccupancyGrid(3, 3, 1.0, 0.0, 0.0)
	wm := NewWorldModel(grid, 5*time.Minute)

	_ = grid.UpdateCell(1.0, 1.0, 1.0)
	grid.LastUpdate[1][1] = time.Now().Add(-2 * time.Minute)

	wm.ClearStaleOccupancy(30 * time.Second)

	occupied, err := grid.IsOccupied(1.0, 1.0, 0.1)
	assert.NoError(t, err)
	assert.False(t, occupied)
}

func TestObservationFromMap(t *testing.T) {
	obs, ok := ObservationFromMap(map[string]interface{}{
		"object_id": "OBJ_1",
		"x":         1.0,
		"y":         2.0,
		"z":         0.5,
		"qw":        1.0,
	})
	assert.True(t, ok)
	assert.Equal(t, "OBJ_1", obs.ObjectID)
	assert.Equal(t, 1.0, obs.Pose.X)
}

func TestWorldModel_UpdateFromSemantic(t *testing.T) {
	grid := NewOccupancyGrid(5, 5, 1.0, 0.0, 0.0)
	wm := NewWorldModel(grid, 5*time.Minute)

	wm.UpdateFromSemantic([]SemanticObject{
		{
			ObjectID: "TABLE_1",
			Label:    "table",
			Pose:     Pose{X: 1, Y: 1, Z: 0.75, Qw: 1},
			Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		},
	})

	occupied, err := grid.IsOccupied(1, 1, 0.5)
	assert.NoError(t, err)
	assert.True(t, occupied)
	assert.True(t, wm.HasSurfaces())
}

func TestObservationsFromJSON(t *testing.T) {
	jsonPayload := []byte(`[
		{"object_id":"OBJ_1","pose":{"position":[1.0,2.0,3.0],"orientation":[1.0,0.0,0.0,0.0]}}
	]`)

	obs := ObservationsFromJSON(jsonPayload)
	assert.Len(t, obs, 1)
	assert.Equal(t, "OBJ_1", obs[0].ObjectID)
	assert.Equal(t, 1.0, obs[0].Pose.X)
}

func TestWorldModel_UpdateFromJSON(t *testing.T) {
	wm := NewWorldModel(nil, 5*time.Minute)
	jsonPayload := []byte(`[
		{"object_id":"OBJ_2","pose":{"position":[0.5,0.0,0.0],"orientation":[1.0,0.0,0.0,0.0]}}
	]`)

	count := wm.UpdateFromJSON(jsonPayload)
	assert.Equal(t, 1, count)
	_, ok := wm.GetObject("OBJ_2")
	assert.True(t, ok)
}

func TestWorldModel_UpdateFromSemanticJSON(t *testing.T) {
	wm := NewWorldModel(NewOccupancyGrid(5, 5, 1.0, 0.0, 0.0), 5*time.Minute)
	jsonPayload := []byte(`[
		{"object_id":"TABLE_X","label":"table","pose":{"position":[1.0,1.0,0.7],"orientation":[1.0,0.0,0.0,0.0]}}
	]`)

	count := wm.UpdateFromSemanticJSON(jsonPayload)
	assert.Equal(t, 1, count)
	assert.True(t, wm.HasSurfaces())
	// Expect table friction to be >= 0.6
	score, _ := wm.AffordanceScore(Pose{X: 1.0, Y: 1.0, Z: 0.7, Qw: 1}, 10)
	assert.GreaterOrEqual(t, score, 0.6)
}
