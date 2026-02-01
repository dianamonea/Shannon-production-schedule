package control

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"shannon/internal/activities"
	"shannon/internal/models"
)

// DeviationDetector monitors actual vs. estimated job execution
type DeviationDetector struct {
	mu                 sync.RWMutex
	deviationThreshold float64         // percent, e.g., 15.0 for 15%
	replanThreshold    float64         // percent, e.g., 25.0 for 25%
	activeJobs         map[string]*JobMetrics
	deviationAlerts    []DeviationAlert
	alertCallbacks     []func(DeviationAlert)
}

// JobMetrics tracks execution progress
type JobMetrics struct {
	JobID              string
	EstimatedDuration  time.Duration
	ActualStartTime    time.Time
	LastUpdateTime     time.Time
	ElapsedTime        time.Duration
	RemainingEstimate  time.Duration
	CompletionPercent  float64  // 0.0-1.0
	DeviationPercent   float64
	Status             string
	AssignedAgent      string
}

// DeviationAlert represents detection of a problem
type DeviationAlert struct {
	AlertID             string
	AlertTime           time.Time
	JobID               string
	AgentID             string
	DeviationType       DeviationType
	Severity            AlertSeverity
	ExpectedTime        time.Duration
	ActualElapsedTime   time.Duration
	DeviationPercent    float64
	RemainingTime       time.Duration
	RecommendedAction   string
	RequiresReplanning  bool
}

type DeviationType string

const (
	DeviationTypeDelay      DeviationType = "DELAY"
	DeviationTypeEarlyCompletion DeviationType = "EARLY_COMPLETION"
	DeviationTypeQualityRisk DeviationType = "QUALITY_RISK"
	DeviationTypeResourceConflict DeviationType = "RESOURCE_CONFLICT"
)

type AlertSeverity string

const (
	SeverityInfo     AlertSeverity = "INFO"
	SeverityWarning  AlertSeverity = "WARNING"
	SeverityCritical AlertSeverity = "CRITICAL"
)

// NewDeviationDetector creates a new detector with thresholds
func NewDeviationDetector(deviationThreshold, replanThreshold float64) *DeviationDetector {
	return &DeviationDetector{
		deviationThreshold: deviationThreshold,
		replanThreshold:    replanThreshold,
		activeJobs:         make(map[string]*JobMetrics),
		deviationAlerts:    make([]DeviationAlert, 0),
		alertCallbacks:     make([]func(DeviationAlert), 0),
	}
}

// RegisterJob starts tracking a job
func (dd *DeviationDetector) RegisterJob(jobID string, agent string, estimatedDuration time.Duration) {
	dd.mu.Lock()
	defer dd.mu.Unlock()

	dd.activeJobs[jobID] = &JobMetrics{
		JobID:             jobID,
		EstimatedDuration: estimatedDuration,
		ActualStartTime:   time.Now(),
		LastUpdateTime:    time.Now(),
		RemainingEstimate: estimatedDuration,
		Status:            "RUNNING",
		AssignedAgent:     agent,
	}
}

// UpdateJobProgress reports actual execution metrics from edge agent
func (dd *DeviationDetector) UpdateJobProgress(jobID string, completionPercent float64) error {
	dd.mu.Lock()
	defer dd.mu.Unlock()

	metrics, exists := dd.activeJobs[jobID]
	if !exists {
		return fmt.Errorf("job %s not registered", jobID)
	}

	now := time.Now()
	metrics.LastUpdateTime = now
	metrics.ElapsedTime = now.Sub(metrics.ActualStartTime)
	metrics.CompletionPercent = completionPercent

	// Calculate expected elapsed time at this completion percentage
	expectedElapsedTime := time.Duration(float64(metrics.EstimatedDuration) * completionPercent)
	
	// Calculate deviation
	actualDeviation := metrics.ElapsedTime - expectedElapsedTime
	deviationPercent := (actualDeviation.Seconds() / expectedElapsedTime.Seconds()) * 100

	metrics.DeviationPercent = deviationPercent
	metrics.RemainingEstimate = time.Duration(
		float64(metrics.EstimatedDuration) * (1 - completionPercent),
	)

	// Check deviation thresholds
	if deviationPercent > dd.deviationThreshold {
		alert := DeviationAlert{
			AlertID:           fmt.Sprintf("ALERT-%s-%d", jobID, now.Unix()),
			AlertTime:         now,
			JobID:             jobID,
			AgentID:           metrics.AssignedAgent,
			DeviationType:     DeviationTypeDelay,
			ExpectedTime:      expectedElapsedTime,
			ActualElapsedTime: metrics.ElapsedTime,
			DeviationPercent:  deviationPercent,
			RemainingTime:     metrics.RemainingEstimate,
		}

		// Determine severity based on deviation magnitude
		if deviationPercent > 50 {
			alert.Severity = SeverityCritical
			alert.RecommendedAction = "IMMEDIATE_REPLAN"
			alert.RequiresReplanning = true
		} else if deviationPercent > dd.replanThreshold {
			alert.Severity = SeverityWarning
			alert.RecommendedAction = "SCHEDULE_REPLAN"
			alert.RequiresReplanning = true
		} else {
			alert.Severity = SeverityInfo
			alert.RecommendedAction = "MONITOR"
			alert.RequiresReplanning = false
		}

		dd.deviationAlerts = append(dd.deviationAlerts, alert)

		// Trigger callbacks
		for _, callback := range dd.alertCallbacks {
			go callback(alert)
		}

		return nil
	}

	// Check for early completion (less common but indicates potential quality issue)
	if deviationPercent < -dd.deviationThreshold {
		alert := DeviationAlert{
			AlertID:           fmt.Sprintf("ALERT-%s-%d", jobID, now.Unix()),
			AlertTime:         now,
			JobID:             jobID,
			AgentID:           metrics.AssignedAgent,
			DeviationType:     DeviationTypeEarlyCompletion,
			ExpectedTime:      expectedElapsedTime,
			ActualElapsedTime: metrics.ElapsedTime,
			DeviationPercent:  deviationPercent,
			RemainingTime:     metrics.RemainingEstimate,
			Severity:          SeverityWarning,
			RecommendedAction: "QUALITY_CHECK",
		}

		dd.deviationAlerts = append(dd.deviationAlerts, alert)
		for _, callback := range dd.alertCallbacks {
			go callback(alert)
		}
	}

	return nil
}

// CompleteJob marks job as done and removes from tracking
func (dd *DeviationDetector) CompleteJob(jobID string) (*JobMetrics, error) {
	dd.mu.Lock()
	defer dd.mu.Unlock()

	metrics, exists := dd.activeJobs[jobID]
	if !exists {
		return nil, fmt.Errorf("job %s not found", jobID)
	}

	metrics.Status = "COMPLETED"
	metrics.ElapsedTime = time.Now().Sub(metrics.ActualStartTime)

	// Clean up
	delete(dd.activeJobs, jobID)

	return metrics, nil
}

// GetDeviationAlerts returns all recorded alerts
func (dd *DeviationDetector) GetDeviationAlerts(jobID string) []DeviationAlert {
	dd.mu.RLock()
	defer dd.mu.RUnlock()

	var alerts []DeviationAlert
	for _, alert := range dd.deviationAlerts {
		if alert.JobID == jobID || jobID == "" {
			alerts = append(alerts, alert)
		}
	}
	return alerts
}

// RegisterAlertCallback adds a handler for deviation alerts
func (dd *DeviationDetector) RegisterAlertCallback(callback func(DeviationAlert)) {
	dd.mu.Lock()
	defer dd.mu.Unlock()
	dd.alertCallbacks = append(dd.alertCallbacks, callback)
}

// ========== PID Adaptive Control Loop ==========

// AdaptivePIDController implements feedback control for production schedules
type AdaptivePIDController struct {
	mu sync.RWMutex

	// PID Parameters
	Kp float64 // Proportional gain
	Ki float64 // Integral gain
	Kd float64 // Derivative gain

	// State tracking
	previousError        float64
	integralAccumulator  float64
	lastUpdateTime       time.Time
	integralWindupLimit  float64
	controlOutputHistory []ControlOutput

	// Configuration
	setpointDuration     time.Duration // target job duration
	maxControlOutput     float64        // max adjustment (e.g., 1.5 = 50% speed increase)
	minControlOutput     float64        // min adjustment (e.g., 0.5 = 50% speed decrease)

	// Callbacks
	onControlAdjustment func(ControlOutput)
}

// ControlOutput represents the PID controller's output
type ControlOutput struct {
	Timestamp          time.Time
	JobID              string
	Error              float64       // setpoint - actual (seconds)
	ProportionalTerm   float64       // Kp * error
	IntegralTerm       float64       // Ki * integral(error)
	DerivativeTerm     float64       // Kd * derivative(error)
	TotalOutput        float64       // P + I + D
	AdjustmentFactor   float64       // bounded [minOutput, maxOutput]
	RemainingDuration  time.Duration
	RecommendedSpeed   float64       // 1.0 = normal, 1.5 = 50% faster, 0.5 = 50% slower
	Action             string        // "CONTINUE", "ACCELERATE", "DECELERATE", "REPLAN"
}

// NewAdaptivePIDController creates a PID controller for manufacturing
func NewAdaptivePIDController(setpointDuration time.Duration) *AdaptivePIDController {
	return &AdaptivePIDController{
		// Manufacturing-tuned PID parameters (start conservative)
		Kp: 0.5,  // proportional response
		Ki: 0.1,  // integral response (prevent steady-state error)
		Kd: 0.05, // derivative response (smooth transitions)

		setpointDuration:    setpointDuration,
		maxControlOutput:    1.5,  // allow 50% speed increase
		minControlOutput:    0.5,  // allow 50% speed decrease
		integralWindupLimit: 30.0, // prevent integral windup
		controlOutputHistory: make([]ControlOutput, 0),
		lastUpdateTime:      time.Now(),
	}
}

// CalculateControl computes PID output and recommended adjustment
func (pc *AdaptivePIDController) CalculateControl(jobID string, actualElapsedTime time.Duration, completionPercent float64) ControlOutput {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	now := time.Now()
	dt := now.Sub(pc.lastUpdateTime).Seconds()
	if dt <= 0 {
		dt = 0.1 // minimum time step
	}

	// Expected elapsed time at current completion percentage
	expectedTime := time.Duration(float64(pc.setpointDuration) * completionPercent).Seconds()

	// Error: setpoint - actual (positive means we're behind)
	error := expectedTime - actualElapsedTime.Seconds()

	// Proportional term
	proportional := pc.Kp * error

	// Integral term (with anti-windup)
	pc.integralAccumulator += error * dt
	if pc.integralAccumulator > pc.integralWindupLimit {
		pc.integralAccumulator = pc.integralWindupLimit
	} else if pc.integralAccumulator < -pc.integralWindupLimit {
		pc.integralAccumulator = -pc.integralWindupLimit
	}
	integral := pc.Ki * pc.integralAccumulator

	// Derivative term
	var derivative float64
	if dt > 0 {
		derivative = pc.Kd * (error - pc.previousError) / dt
	}
	pc.previousError = error

	// Total control output
	totalOutput := proportional + integral + derivative

	// Bound the output to realistic adjustment factors
	adjustmentFactor := totalOutput
	if adjustmentFactor > pc.maxControlOutput {
		adjustmentFactor = pc.maxControlOutput
	} else if adjustmentFactor < pc.minControlOutput {
		adjustmentFactor = pc.minControlOutput
	}

	// Calculate recommended speed multiplier
	// adjustmentFactor > 1.0 means speed up, < 1.0 means slow down
	recommendedSpeed := 1.0 / adjustmentFactor // inverse: if we're slow, speed up

	// Determine action
	action := "CONTINUE"
	if completionPercent > 0.95 && error > 5 { // >5 sec behind at 95% done
		action = "ACCELERATE"
	} else if error < -10 && completionPercent < 0.3 { // >10 sec ahead at start
		action = "DECELERATE"
	} else if error > 60 { // >60 seconds behind
		action = "REPLAN"
	}

	remainingDuration := time.Duration(float64(pc.setpointDuration)*(1-completionPercent)) * time.Second

	output := ControlOutput{
		Timestamp:          now,
		JobID:              jobID,
		Error:              error,
		ProportionalTerm:   proportional,
		IntegralTerm:       integral,
		DerivativeTerm:     derivative,
		TotalOutput:        totalOutput,
		AdjustmentFactor:   adjustmentFactor,
		RemainingDuration:  remainingDuration,
		RecommendedSpeed:   recommendedSpeed,
		Action:             action,
	}

	pc.controlOutputHistory = append(pc.controlOutputHistory, output)

	// Keep history size manageable
	if len(pc.controlOutputHistory) > 1000 {
		pc.controlOutputHistory = pc.controlOutputHistory[100:]
	}

	pc.lastUpdateTime = now

	// Trigger callback
	if pc.onControlAdjustment != nil {
		go pc.onControlAdjustment(output)
	}

	return output
}

// TuneParameters adjusts PID gains (used by learning system)
func (pc *AdaptivePIDController) TuneParameters(kp, ki, kd float64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.Kp = kp
	pc.Ki = ki
	pc.Kd = kd
	log.Printf("PID parameters updated: Kp=%.3f, Ki=%.3f, Kd=%.3f", kp, ki, kd)
}

// GetControlHistory returns recent control outputs
func (pc *AdaptivePIDController) GetControlHistory(limit int) []ControlOutput {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	if limit > len(pc.controlOutputHistory) {
		limit = len(pc.controlOutputHistory)
	}

	return pc.controlOutputHistory[len(pc.controlOutputHistory)-limit:]
}

// SetControlAdjustmentCallback registers handler for control outputs
func (pc *AdaptivePIDController) SetControlAdjustmentCallback(callback func(ControlOutput)) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.onControlAdjustment = callback
}

// ========== Closed-Loop Control Loop Activity ==========

// ClosedLoopControlActivity represents the Temporal activity
// that runs adaptive feedback control
type ClosedLoopControlActivity struct {
	devDetector *DeviationDetector
	pidControl  *AdaptivePIDController
}

// NewClosedLoopControlActivity creates the activity
func NewClosedLoopControlActivity() *ClosedLoopControlActivity {
	return &ClosedLoopControlActivity{
		devDetector: NewDeviationDetector(15.0, 25.0), // 15% alert, 25% replan
		pidControl:  NewAdaptivePIDController(10 * time.Minute),
	}
}

// ExecuteClosedLoopControl is the main activity logic
// Called by Temporal workflow to manage feedback control
func (clc *ClosedLoopControlActivity) ExecuteClosedLoopControl(
	ctx context.Context,
	jobID string,
	agentID string,
	estimatedDuration time.Duration,
) ([]ControlOutput, error) {

	log.Printf("Starting closed-loop control for job %s on agent %s", jobID, agentID)

	clc.devDetector.RegisterJob(jobID, agentID, estimatedDuration)
	clc.pidControl = NewAdaptivePIDController(estimatedDuration)

	controlOutputs := make([]ControlOutput, 0)

	// Register callbacks
	clc.devDetector.RegisterAlertCallback(func(alert DeviationAlert) {
		log.Printf("Deviation Alert: %s - %s (%.1f%% deviation)", alert.JobID, alert.DeviationType, alert.DeviationPercent)
		if alert.RequiresReplanning {
			log.Printf("  Action Required: %s", alert.RecommendedAction)
		}
	})

	clc.pidControl.SetControlAdjustmentCallback(func(output ControlOutput) {
		log.Printf("Control Output: Job=%s, Error=%.1fs, Action=%s, Speed=%.2fx",
			output.JobID, output.Error, output.Action, output.RecommendedSpeed)
		controlOutputs = append(controlOutputs, output)
	})

	// Simulated monitoring loop (in production, would receive real-time updates from edge agent)
	monitorTicker := time.NewTicker(5 * time.Second) // check every 5 seconds
	defer monitorTicker.Stop()

	completionCheckTicker := time.NewTicker(estimatedDuration + 5*time.Second)
	defer completionCheckTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return controlOutputs, ctx.Err()

		case <-monitorTicker.C:
			// In production, would receive actual progress from edge agent via gRPC/WebSocket
			// For now, simulate gradual progress
			simpleCompletionPercent := time.Since(clc.devDetector.activeJobs[jobID].ActualStartTime).Seconds() /
				estimatedDuration.Seconds()

			if simpleCompletionPercent >= 1.0 {
				break
			}

			actualElapsed := time.Since(clc.devDetector.activeJobs[jobID].ActualStartTime)

			// Update deviation detector
			_ = clc.devDetector.UpdateJobProgress(jobID, simpleCompletionPercent)

			// Calculate PID control
			controlOutput := clc.pidControl.CalculateControl(jobID, actualElapsed, simpleCompletionPercent)

			// Apply control action based on output
			if controlOutput.Action == "REPLAN" {
				log.Printf("PID triggered re-planning for job %s", jobID)
				_, _ = clc.devDetector.CompleteJob(jobID)
				return controlOutputs, fmt.Errorf("replan_required")
			}

		case <-completionCheckTicker.C:
			_, _ = clc.devDetector.CompleteJob(jobID)
			log.Printf("Closed-loop control completed for job %s", jobID)
			return controlOutputs, nil
		}
	}
}

// GetJobMetrics returns current metrics for a job
func (clc *ClosedLoopControlActivity) GetJobMetrics(jobID string) *JobMetrics {
	clc.devDetector.mu.RLock()
	defer clc.devDetector.mu.RUnlock()
	
	if metrics, exists := clc.devDetector.activeJobs[jobID]; exists {
		// Return copy to avoid mutation
		copy := *metrics
		return &copy
	}
	return nil
}

// GetDeviationAlerts returns all alerts for a job
func (clc *ClosedLoopControlActivity) GetDeviationAlerts(jobID string) []DeviationAlert {
	return clc.devDetector.GetDeviationAlerts(jobID)
}

// ========== Integration with Orchestrator ==========

// TriggerReplanning is called when deviation exceeds threshold
func TriggerReplanning(ctx context.Context, jobID string, reason string) error {
	log.Printf("Triggering re-planning for job %s: %s", jobID, reason)
	
	// In production Temporal workflow:
	// - Signal workflow to enter re-planning mode
	// - Call CNP orchestrator to re-evaluate assignments
	// - Update schedule and notify agents
	
	return nil
}

// ApplyControlAdjustment sends speed adjustment to edge agent
func ApplyControlAdjustment(ctx context.Context, agentID string, adjustment ControlOutput) error {
	log.Printf("Applying control adjustment to agent %s: speed=%.2fx", agentID, adjustment.RecommendedSpeed)
	
	// In production:
	// - Send gRPC command to agent-core (Rust service)
	// - Adjust spindle speed, feed rate, or movement velocity
	// - Return confirmation of applied adjustment
	
	return nil
}

// ========== Knowledge Distillation for Edge Processing ==========

// EdgeDistilledRule represents a simplified rule for edge execution
type EdgeDistilledRule struct {
	RuleID      string
	Condition   string // human-readable
	Action      string // human-readable
	ConfidenceScore float64
	Examples    []RuleExample
}

type RuleExample struct {
	Condition string
	Result    string
}

// DistillComplexPolicyToEdgeRules converts complex scheduling logic to simple If-Then rules
func DistillComplexPolicyToEdgeRules(pidController *AdaptivePIDController) []EdgeDistilledRule {
	rules := make([]EdgeDistilledRule, 0)

	// Rule 1: Load-based bidding
	rules = append(rules, EdgeDistilledRule{
		RuleID:      "EDGE_RULE_001",
		Condition:   "IF current_load > 0.8 AND tool_health < 0.5 THEN",
		Action:      "REJECT_BID for new jobs, request tool maintenance",
		ConfidenceScore: 0.92,
		Examples: []RuleExample{
			{
				Condition: "Robot at 85% utilization with 6 active jobs, drill bit at 45% lifecycle",
				Result:    "Decline assignment, suggest waiting 30 minutes for completion",
			},
		},
	})

	// Rule 2: Deadline-driven acceleration
	rules = append(rules, EdgeDistilledRule{
		RuleID:      "EDGE_RULE_002",
		Condition:   "IF time_to_deadline < 2*estimated_job_time AND current_load < 0.5 THEN",
		Action:      "ACCELERATE execution, increase spindle RPM by 15%, accept job",
		ConfidenceScore: 0.88,
		Examples: []RuleExample{
			{
				Condition: "Job deadline in 30 min, estimated duration 20 min, robot idle",
				Result:    "Accept job, run at 1.15x speed to finish safely by deadline",
			},
		},
	})

	// Rule 3: Quality gate
	rules = append(rules, EdgeDistilledRule{
		RuleID:      "EDGE_RULE_003",
		Condition:   "IF required_tolerance < achievable_tolerance_with_margin AND quality_score < 0.85 THEN",
		Action:      "REJECT_BID or REQUEST_MANUAL_INSPECTION before proceeding",
		ConfidenceScore: 0.95,
		Examples: []RuleExample{
			{
				Condition: "Job requires ±0.05mm, robot's SPC shows avg ±0.08mm, recent pass rate 78%",
				Result:    "Decline job, suggest grinding wheel maintenance or manual operation",
			},
		},
	})

	// Rule 4: Maintenance window prediction
	rules = append(rules, EdgeDistilledRule{
		RuleID:      "EDGE_RULE_004",
		Condition:   "IF tool_cycles > 0.9*max_tool_cycles AND job_duration > 30min THEN",
		Action:      "SCHEDULE_TOOL_CHANGE between this job and next, accept if 45min available",
		ConfidenceScore: 0.90,
		Examples: []RuleExample{
			{
				Condition: "End mill at 9000/10000 cycles, 45min job queued, next job in 90min",
				Result:    "Schedule tool change (10min) between jobs, avoid breakage during job",
			},
		},
	})

	// Rule 5: PID-derived speed adjustment
	rules = append(rules, EdgeDistilledRule{
		RuleID:      "EDGE_RULE_005",
		Condition:   "IF elapsed_time > expected_time*1.2 AND completion < 0.8 THEN",
		Action:      fmt.Sprintf("INCREASE_SPEED: multiply feedrate by %.2f to catch up", pidController.Kp),
		ConfidenceScore: 0.85,
		Examples: []RuleExample{
			{
				Condition: "20% into job, already 20% over time",
				Result:    "Increase feedrate by 50% (Kp=0.5) to recover timing",
			},
		},
	})

	return rules
}
