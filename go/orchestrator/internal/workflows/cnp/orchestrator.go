package cnp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/worldmodel"
)

// Contract Net Protocol (CNP) implementation for multi-agent manufacturing scheduling
// RFC: FIPA Contract Net Protocol specification

// TaskDescription describes a manufacturing task to be bid on
type TaskDescription struct {
	TaskID          string                 `json:"task_id"`
	TaskType        string                 `json:"task_type"`        // MILLING, ASSEMBLY, INSPECTION, etc.
	MaterialSpec    string                 `json:"material_spec"`    // Material requirements
	ToolRequirement string                 `json:"tool_requirement"` // Tool IDs needed
	PrecisionLevel  float64                `json:"precision_level"`  // ±μm
	EstimatedTime   time.Duration          `json:"estimated_time"`   // Estimated duration
	Deadline        time.Time              `json:"deadline"`         // Hard deadline
	Priority        int                    `json:"priority"`         // 1-10
	ExtraParameters map[string]interface{} `json:"extra_parameters"`
}

// BidProposal is submitted by edge agents in response to task advertisement
type BidProposal struct {
	BidID              string        `json:"bid_id"`
	TaskID             string        `json:"task_id"`
	AgentID            string        `json:"agent_id"`
	EstimatedDuration  time.Duration `json:"estimated_duration"`
	EstimatedCost      float64       `json:"estimated_cost"`
	CurrentLoad        float64       `json:"current_load"`        // 0.0-1.0
	ToolAvailability   map[string]bool `json:"tool_availability"`
	MaterialInventory  map[string]float64 `json:"material_inventory"` // material -> available_qty
	ToolConditions     map[string]float64 `json:"tool_conditions"`    // tool_id -> health %
	EstimatedQuality   float64       `json:"estimated_quality"`   // 0.0-1.0
	LocalDisruptionRisk float64      `json:"local_disruption_risk"` // 0.0-1.0
	SubmissionTime     time.Time     `json:"submission_time"`
}

// AwardNotification is sent to winning bidder
type AwardNotification struct {
	BidID              string        `json:"bid_id"`
	TaskID             string        `json:"task_id"`
	AgentID            string        `json:"agent_id"`
	Award              bool          `json:"award"` // true=win, false=reject
	ConfirmationDeadline time.Time   `json:"confirmation_deadline"`
	ControlLock        *ResourceLock `json:"control_lock,omitempty"`
}

// ResourceLock prevents concurrent access to resources
type ResourceLock struct {
	LockID      string
	AgentID     string
	Resources   map[string]int // resource_id -> qty locked
	LockedUntil time.Time
}

// BidEvaluationCriteria for task allocation
type BidEvaluationCriteria struct {
	WeightDuration    float64 `json:"weight_duration"`    // 0.0-1.0
	WeightCost        float64 `json:"weight_cost"`        // 0.0-1.0
	WeightQuality     float64 `json:"weight_quality"`     // 0.0-1.0
	WeightLoad        float64 `json:"weight_load"`        // 0.0-1.0, prefer lower load
	WeightToolHealth  float64 `json:"weight_tool_health"` // 0.0-1.0
	LocalDisruptionPenalty float64 `json:"local_disruption_penalty"` // penalty for schedule changes
}

// CNPOrchestrator manages the contract net protocol
type CNPOrchestrator struct {
	logger     *zap.Logger
	taskChan   chan TaskDescription
	bidsChan   chan BidProposal
	awards     chan AwardNotification

	activeTasks    map[string]*TaskDescription
	activeBids     map[string][]BidProposal
	resourceLocks  map[string]*ResourceLock
	agentRegistry  map[string]*AgentInfo
	worldModel     *worldmodel.WorldModel

	mu                sync.RWMutex
	biddingTimeout    time.Duration
	confirmationTimeout time.Duration
	evaluationCriteria BidEvaluationCriteria
}

// AgentInfo tracks edge agent state
type AgentInfo struct {
	AgentID          string
	LastSeen         time.Time
	CurrentLoad      float64
	ActiveTasks      []string
	ToolInventory    map[string]ToolState
	MaterialStock    map[string]float64
}

// ToolState tracks tool condition
type ToolState struct {
	ToolID   string
	Health   float64 // 0.0-1.0
	Cycles   int64
	MaxCycles int64
	LastMaintenance time.Time
}

// ScoringResult for task allocation decision
type ScoringResult struct {
	BidID      string
	AgentID    string
	Score      float64 // 0.0-1.0
	Components map[string]float64 // breakdown: duration, cost, quality, etc.
}

// NewCNPOrchestrator creates new orchestrator
func NewCNPOrchestrator(logger *zap.Logger) *CNPOrchestrator {
	return &CNPOrchestrator{
		logger:               logger,
		taskChan:             make(chan TaskDescription, 100),
		bidsChan:             make(chan BidProposal, 1000),
		awards:               make(chan AwardNotification, 100),
		activeTasks:          make(map[string]*TaskDescription),
		activeBids:           make(map[string][]BidProposal),
		resourceLocks:        make(map[string]*ResourceLock),
		agentRegistry:        make(map[string]*AgentInfo),
		worldModel:           nil,
		biddingTimeout:       3 * time.Second,  // Fast timeout for manufacturing
		confirmationTimeout:  2 * time.Second,
		evaluationCriteria: BidEvaluationCriteria{
			WeightDuration:     0.30,
			WeightCost:         0.20,
			WeightQuality:      0.25,
			WeightLoad:         0.15,
			WeightToolHealth:   0.10,
			LocalDisruptionPenalty: 0.05,
		},
	}
}

// UpdateEvaluationCriteria allows online optimization to update bid weights
func (o *CNPOrchestrator) UpdateEvaluationCriteria(criteria BidEvaluationCriteria) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.evaluationCriteria = criteria
}

// SetWorldModel injects the world model for spatial reasoning
func (o *CNPOrchestrator) SetWorldModel(wm *worldmodel.WorldModel) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.worldModel = wm
}

// UpdateWorldModelFromSemanticJSON ingests semantic observations into world model
func (o *CNPOrchestrator) UpdateWorldModelFromSemanticJSON(payload []byte) int {
	o.mu.RLock()
	wm := o.worldModel
	o.mu.RUnlock()
	if wm == nil {
		return 0
	}
	return wm.UpdateFromJSON(payload)
}

// UpdateWorldModelFromSemanticObjectsJSON ingests semantic objects with labels
func (o *CNPOrchestrator) UpdateWorldModelFromSemanticObjectsJSON(payload []byte) int {
	o.mu.RLock()
	wm := o.worldModel
	o.mu.RUnlock()
	if wm == nil {
		return 0
	}
	return wm.UpdateFromSemanticJSON(payload)
}

// AdvertiseTask broadcasts task to all available agents
func (o *CNPOrchestrator) AdvertiseTask(ctx context.Context, task TaskDescription) error {
	if task.TaskID == "" {
		task.TaskID = uuid.New().String()
	}

	o.mu.Lock()
	o.activeTasks[task.TaskID] = &task
	o.activeBids[task.TaskID] = make([]BidProposal, 0)
	o.mu.Unlock()

	o.logger.Info("Task advertised",
		zap.String("task_id", task.TaskID),
		zap.String("task_type", task.TaskType),
		zap.Time("deadline", task.Deadline),
	)

	// In production, broadcast to all online agents via pub/sub or RPC
	// For now, return success
	return nil
}

// RegisterBid receives bid from edge agent
func (o *CNPOrchestrator) RegisterBid(ctx context.Context, bid BidProposal) error {
	if bid.BidID == "" {
		bid.BidID = uuid.New().String()
	}
	if bid.SubmissionTime.IsZero() {
		bid.SubmissionTime = time.Now()
	}

	o.mu.Lock()
	activeBids, exists := o.activeBids[bid.TaskID]
	if !exists {
		o.mu.Unlock()
		return fmt.Errorf("task %s not found", bid.TaskID)
	}

	activeBids = append(activeBids, bid)
	o.activeBids[bid.TaskID] = activeBids

	// Update agent status
	if agent, ok := o.agentRegistry[bid.AgentID]; ok {
		agent.CurrentLoad = bid.CurrentLoad
		agent.LastSeen = time.Now()
	} else {
		o.agentRegistry[bid.AgentID] = &AgentInfo{
			AgentID:     bid.AgentID,
			LastSeen:    time.Now(),
			CurrentLoad: bid.CurrentLoad,
		}
	}
	o.mu.Unlock()

	o.logger.Debug("Bid registered",
		zap.String("bid_id", bid.BidID),
		zap.String("agent_id", bid.AgentID),
		zap.String("task_id", bid.TaskID),
		zap.Duration("estimated_duration", bid.EstimatedDuration),
	)

	return nil
}

// EvaluateBids scores all bids for a task and selects winner
func (o *CNPOrchestrator) EvaluateBids(ctx context.Context, taskID string) (*ScoringResult, error) {
	o.mu.RLock()
	task, ok := o.activeTasks[taskID]
	if !ok {
		o.mu.RUnlock()
		return nil, fmt.Errorf("task %s not found", taskID)
	}

	bids, ok := o.activeBids[taskID]
	if !ok || len(bids) == 0 {
		o.mu.RUnlock()
		return nil, fmt.Errorf("no bids for task %s", taskID)
	}
	o.mu.RUnlock()

	if len(bids) == 0 {
		return nil, fmt.Errorf("no valid bids for task")
	}

	// Score all bids
	scores := make([]*ScoringResult, 0, len(bids))
	for _, bid := range bids {
		score := o.scoreBid(task, &bid)
		scores = append(scores, score)

		o.logger.Debug("Bid scored",
			zap.String("bid_id", bid.BidID),
			zap.String("agent_id", bid.AgentID),
			zap.Float64("score", score.Score),
		)
	}

	// Find winner (highest score)
	var winner *ScoringResult
	for _, score := range scores {
		if winner == nil || score.Score > winner.Score {
			winner = score
		}
	}

	o.logger.Info("Task awarded",
		zap.String("task_id", taskID),
		zap.String("winner_agent", winner.AgentID),
		zap.Float64("winning_score", winner.Score),
	)

	return winner, nil
}

// scoreBid applies evaluation criteria to calculate score
func (o *CNPOrchestrator) scoreBid(task *TaskDescription, bid *BidProposal) *ScoringResult {
	crit := o.evaluationCriteria
	if task != nil && task.ExtraParameters != nil {
		crit = o.overrideCriteria(task.ExtraParameters, crit)
	}

	// Normalize scores to 0.0-1.0 range
	// Duration: prefer shorter, normalize against estimated
	durationScore := 1.0
	if bid.EstimatedDuration > task.EstimatedTime {
		ratio := float64(task.EstimatedTime) / float64(bid.EstimatedDuration)
		durationScore = ratio
	}
	if durationScore > 1.0 {
		durationScore = 1.0
	}

	// Cost: prefer lower cost, normalize against a baseline
	costScore := 1.0 - (bid.EstimatedCost / 1000.0) // baseline 1000 units
	if costScore < 0.0 {
		costScore = 0.0
	}

	// Quality: direct score
	qualityScore := bid.EstimatedQuality

	// Load: prefer lower load
	loadScore := 1.0 - bid.CurrentLoad

	// Tool health: prefer better condition
	toolHealthScore := 0.8 // default
	if len(bid.ToolConditions) > 0 {
		sum := 0.0
		for _, health := range bid.ToolConditions {
			sum += health
		}
		toolHealthScore = sum / float64(len(bid.ToolConditions))
	}

	// Calculate weighted score
	totalScore := (durationScore * crit.WeightDuration) +
		(costScore * crit.WeightCost) +
		(qualityScore * crit.WeightQuality) +
		(loadScore * crit.WeightLoad) +
		(toolHealthScore * crit.WeightToolHealth)

	// Apply local disruption penalty if provided
	if bid.LocalDisruptionRisk > 0 {
		penalty := bid.LocalDisruptionRisk * crit.LocalDisruptionPenalty
		totalScore -= penalty
		if totalScore < 0 {
			totalScore = 0
		}
	}

	// Normalize to 0.0-1.0
	totalScore = totalScore / (crit.WeightDuration + crit.WeightCost + crit.WeightQuality + crit.WeightLoad + crit.WeightToolHealth)

	return &ScoringResult{
		BidID:   bid.BidID,
		AgentID: bid.AgentID,
		Score:   totalScore,
		Components: map[string]float64{
			"duration":   durationScore,
			"cost":       costScore,
			"quality":    qualityScore,
			"load":       loadScore,
			"tool_health": toolHealthScore,
			"disruption_risk": bid.LocalDisruptionRisk,
		},
	}
}

func (o *CNPOrchestrator) overrideCriteria(params map[string]interface{}, base BidEvaluationCriteria) BidEvaluationCriteria {
	if v, ok := params["weight_duration"]; ok {
		base.WeightDuration = parseFloat(v, base.WeightDuration)
	}
	if v, ok := params["weight_cost"]; ok {
		base.WeightCost = parseFloat(v, base.WeightCost)
	}
	if v, ok := params["weight_quality"]; ok {
		base.WeightQuality = parseFloat(v, base.WeightQuality)
	}
	if v, ok := params["weight_load"]; ok {
		base.WeightLoad = parseFloat(v, base.WeightLoad)
	}
	if v, ok := params["weight_tool_health"]; ok {
		base.WeightToolHealth = parseFloat(v, base.WeightToolHealth)
	}
	if v, ok := params["local_disruption_penalty"]; ok {
		base.LocalDisruptionPenalty = parseFloat(v, base.LocalDisruptionPenalty)
	}
	return base
}

// AwardTask sends award notification to winning agent
func (o *CNPOrchestrator) AwardTask(ctx context.Context, taskID string, score *ScoringResult) error {
	lock := &ResourceLock{
		LockID:      uuid.New().String(),
		AgentID:     score.AgentID,
		Resources:   make(map[string]int),
		LockedUntil: time.Now().Add(1 * time.Hour),
	}

	o.mu.Lock()
	o.resourceLocks[lock.LockID] = lock

	task, ok := o.activeTasks[taskID]
	if !ok {
		o.mu.Unlock()
		return fmt.Errorf("task %s not found", taskID)
	}

	// Spatial safety check (optional)
	if o.worldModel != nil {
		if err := o.checkSpatialConstraints(task); err != nil {
			o.mu.Unlock()
			return err
		}
	}

	// Update agent's active tasks
	if agent, ok := o.agentRegistry[score.AgentID]; ok {
		agent.ActiveTasks = append(agent.ActiveTasks, taskID)
	}
	o.mu.Unlock()

	award := AwardNotification{
		BidID:               score.BidID,
		TaskID:              taskID,
		AgentID:             score.AgentID,
		Award:               true,
		ConfirmationDeadline: time.Now().Add(o.confirmationTimeout),
		ControlLock:         lock,
	}

	o.logger.Info("Task award sent",
		zap.String("task_id", taskID),
		zap.String("agent_id", score.AgentID),
		zap.String("lock_id", lock.LockID),
		zap.Time("locked_until", lock.LockedUntil),
	)

	// In production, send award to agent via RPC
	select {
	case o.awards <- award:
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}

func (o *CNPOrchestrator) checkSpatialConstraints(task *TaskDescription) error {
	if task == nil || task.ExtraParameters == nil {
		return nil
	}

	// Occupancy check for target pose
	if pose, ok := parsePose(task.ExtraParameters["target_pose"]); ok {
		occupied, err := o.worldModelGridOccupied(pose.X, pose.Y)
		if err != nil {
			return err
		}
		if occupied {
			return fmt.Errorf("target pose occupied: (%.3f, %.3f)", pose.X, pose.Y)
		}

		// Lock occupancy for the target pose to avoid conflicts
		o.worldModel.UpdateOccupancy([]worldmodel.Pose{pose}, 1.0)

		// Placement stability check (affordance reasoning)
		if o.worldModel.HasSurfaces() {
			maxTilt := parseFloat(task.ExtraParameters["placement_max_tilt_deg"], 10.0)
			minAfford := parseFloat(task.ExtraParameters["placement_min_affordance"], 0.0)
			if stable, surfaceID := o.worldModel.IsPlacementStable(pose, maxTilt); !stable {
				return fmt.Errorf("unstable placement at target pose; no suitable surface (maxTilt=%.1f)", maxTilt)
			} else if surfaceID != "" {
				o.logger.Info("Placement surface selected",
					zap.String("surface_id", surfaceID),
					zap.String("task_id", task.TaskID),
				)
			}

			if minAfford > 0 {
				score, surfaceID := o.worldModel.AffordanceScore(pose, maxTilt)
				if score < minAfford {
					return fmt.Errorf("affordance score %.2f below threshold %.2f", score, minAfford)
				}
				if surfaceID != "" {
					o.logger.Info("Affordance surface selected",
						zap.String("surface_id", surfaceID),
						zap.String("task_id", task.TaskID),
						zap.Float64("score", score),
					)
				}
			}
		}
	}

	// Collision prediction based on object IDs
	objectIDs := parseStringSlice(task.ExtraParameters["object_ids"])
	if len(objectIDs) >= 2 {
		horizon := parseFloat(task.ExtraParameters["collision_horizon_sec"], 2.0)
		threshold := parseFloat(task.ExtraParameters["collision_threshold_m"], 0.5)
		for i := 0; i < len(objectIDs); i++ {
			for j := i + 1; j < len(objectIDs); j++ {
				risk, dist := o.worldModel.DetectCollisionRisk(objectIDs[i], objectIDs[j], time.Duration(horizon*float64(time.Second)), threshold)
				if risk {
					return fmt.Errorf("collision risk between %s and %s (dist=%.3f)", objectIDs[i], objectIDs[j], dist)
				}
			}
		}
	}

	return nil
}

func (o *CNPOrchestrator) worldModelGridOccupied(x, y float64) (bool, error) {
	// If grid is not configured, skip occupancy check
	if o.worldModel == nil {
		return false, nil
	}
	// 0.6 occupancy threshold by default
	grid := o.worldModelGrid()
	if grid == nil {
		return false, nil
	}
	return grid.IsOccupied(x, y, 0.6)
}

func (o *CNPOrchestrator) worldModelGrid() *worldmodel.OccupancyGrid {
	return o.worldModel.GetGrid()
}

func parsePose(value interface{}) (worldmodel.Pose, bool) {
	poseMap, ok := value.(map[string]interface{})
	if !ok {
		return worldmodel.Pose{}, false
	}

	return worldmodel.Pose{
		X:  parseFloat(poseMap["x"], 0),
		Y:  parseFloat(poseMap["y"], 0),
		Z:  parseFloat(poseMap["z"], 0),
		Qw: parseFloat(poseMap["qw"], 1),
		Qx: parseFloat(poseMap["qx"], 0),
		Qy: parseFloat(poseMap["qy"], 0),
		Qz: parseFloat(poseMap["qz"], 0),
	}, true
}

func parseStringSlice(value interface{}) []string {
	if value == nil {
		return nil
	}

	switch v := value.(type) {
	case []string:
		return v
	case []interface{}:
		items := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				items = append(items, s)
			}
		}
		return items
	default:
		return nil
	}
}

func parseFloat(value interface{}, fallback float64) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return fallback
	}
}

// ReleaseResourceLock frees resources after task completion
func (o *CNPOrchestrator) ReleaseResourceLock(lockID string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	lock, ok := o.resourceLocks[lockID]
	if !ok {
		return fmt.Errorf("lock %s not found", lockID)
	}

	o.logger.Debug("Resource lock released", zap.String("lock_id", lockID))
	delete(o.resourceLocks, lockID)

	return nil
}

// HandleLocalDisruption triggers re-planning when local environment changes
func (o *CNPOrchestrator) HandleLocalDisruption(ctx context.Context, agentID string, disruptionType string) error {
	o.mu.RLock()
	agent, ok := o.agentRegistry[agentID]
	if !ok {
		o.mu.RUnlock()
		return fmt.Errorf("agent %s not found", agentID)
	}

	activeTasks := make([]string, len(agent.ActiveTasks))
	copy(activeTasks, agent.ActiveTasks)
	o.mu.RUnlock()

	o.logger.Warn("Local disruption detected",
		zap.String("agent_id", agentID),
		zap.String("disruption_type", disruptionType),
		zap.Int("affected_tasks", len(activeTasks)),
	)

	// Trigger re-evaluation for all affected tasks
	// In production, this would coordinate with agent's local re-planner
	for _, taskID := range activeTasks {
		o.logger.Info("Task requires re-planning due to disruption",
			zap.String("task_id", taskID),
			zap.String("disruption", disruptionType),
		)
	}

	return nil
}

// GetAgentInfo returns current state of an agent
func (o *CNPOrchestrator) GetAgentInfo(agentID string) *AgentInfo {
	o.mu.RLock()
	defer o.mu.RUnlock()

	if agent, ok := o.agentRegistry[agentID]; ok {
		// Return a copy to avoid race conditions
		copy := *agent
		return &copy
	}
	return nil
}

// ListActiveTasks returns all tasks awaiting bidding or in progress
func (o *CNPOrchestrator) ListActiveTasks() []*TaskDescription {
	o.mu.RLock()
	defer o.mu.RUnlock()

	tasks := make([]*TaskDescription, 0, len(o.activeTasks))
	for _, task := range o.activeTasks {
		tasks = append(tasks, task)
	}
	return tasks
}
