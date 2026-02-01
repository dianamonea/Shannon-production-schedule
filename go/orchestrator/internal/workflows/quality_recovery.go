package workflows

import (
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/models"
)

// QualityMetric represents quality measurement types
type QualityMetric string

const (
	DimensionalAccuracy QualityMetric = "DIMENSIONAL_ACCURACY"
	SurfaceFinish       QualityMetric = "SURFACE_FINISH"
	AssemblyAlignment   QualityMetric = "ASSEMBLY_ALIGNMENT"
	VisualDefect        QualityMetric = "VISUAL_DEFECT"
)

// DefectType categorizes types of defects
type DefectType string

const (
	DefectDimensional DefectType = "DIMENSIONAL"
	DefectSurface     DefectType = "SURFACE"
	DefectAssembly    DefectType = "ASSEMBLY"
	DefectMaterial    DefectType = "MATERIAL"
	DefectOther       DefectType = "OTHER"
)

// DefectSeverity indicates how critical a defect is
type DefectSeverity int

const (
	SeverityMinor    DefectSeverity = 1
	SeverityModerate DefectSeverity = 5
	SeverityCritical DefectSeverity = 10
)

// QualityInspection represents an inspection result
type QualityInspection struct {
	InspectionID   string
	JobID          string
	AgentID        string
	Metric         QualityMetric
	MeasuredValue  float64
	TargetValue    float64
	Tolerance      float64
	IsDefective    bool
	DetectedAt     time.Time
	InspectorAgent string
}

// Defect represents a detected quality issue
type Defect struct {
	DefectID   string
	JobID      string
	Type       DefectType
	Severity   DefectSeverity
	DetectedAt time.Time
	AgentID    string     // Agent that produced the defect
	ToolID     string     // Tool used
	MaterialID string     // Material batch
	Description string
	CanRework  bool
	ReworkCost float64
}

// ReworkAction defines how to fix a defect
type ReworkAction struct {
	ActionID    string
	DefectID    string
	JobID       string
	Action      string // REGRIND, REASSEMBLE, REPOLISH, SCRAP
	AssignedTo  string // Agent ID
	EstimatedTime time.Duration
	Cost        float64
	ScheduledAt time.Time
	CompletedAt *time.Time
	Status      string // PENDING, IN_PROGRESS, COMPLETED, FAILED
}

// SPCChart represents Statistical Process Control chart data
type SPCChart struct {
	Metric        QualityMetric
	Samples       []float64
	Timestamps    []time.Time
	Mean          float64
	StdDev        float64
	UCL           float64 // Upper Control Limit
	LCL           float64 // Lower Control Limit
	OutOfControl  bool
}

// RootCauseAnalysis traces a defect to its source
type RootCauseAnalysis struct {
	DefectID       string
	ProbableRoot   string // AGENT_MALFUNCTION, TOOL_WEAR, MATERIAL_VARIANCE, PROCESS_DRIFT
	AffectedJobs   []string
	AffectedAgents []string
	AffectedTools  []string
	Confidence     float64 // 0.0 - 1.0
	RecommendedAction string
}

// QualityRecovery manages quality control and defect recovery
type QualityRecovery struct {
	mu sync.RWMutex
	
	inspections map[string]*QualityInspection
	defects     map[string]*Defect
	reworkQueue []*ReworkAction
	spcCharts   map[QualityMetric]*SPCChart
	investigationCallbacks []func(float64)
	lastInvestigationAt    *time.Time
	investigationCooldown  time.Duration
	
	// Thresholds
	defectRateThreshold float64 // Trigger investigation if exceeded
	spcSigmaMultiplier  float64 // For UCL/LCL calculation
}

// NewQualityRecovery creates a new quality recovery manager
func NewQualityRecovery() *QualityRecovery {
	return &QualityRecovery{
		inspections:         make(map[string]*QualityInspection),
		defects:             make(map[string]*Defect),
		reworkQueue:         make([]*ReworkAction, 0),
		spcCharts:           make(map[QualityMetric]*SPCChart),
		investigationCallbacks: make([]func(float64), 0),
		investigationCooldown:  10 * time.Minute,
		defectRateThreshold: 0.05, // 5% defect rate threshold
		spcSigmaMultiplier:  3.0,  // 3-sigma control limits
	}
}

// RegisterInvestigationCallback registers a callback for investigation triggers
func (qr *QualityRecovery) RegisterInvestigationCallback(cb func(float64)) {
	qr.mu.Lock()
	defer qr.mu.Unlock()
	qr.investigationCallbacks = append(qr.investigationCallbacks, cb)
}

// SetInvestigationCooldown updates cooldown for investigation alerts
func (qr *QualityRecovery) SetInvestigationCooldown(cooldown time.Duration) {
	qr.mu.Lock()
	defer qr.mu.Unlock()
	qr.investigationCooldown = cooldown
}

// ShouldTriggerInvestigation checks if defect rate exceeds threshold
func (qr *QualityRecovery) ShouldTriggerInvestigation() bool {
	qr.mu.RLock()
	defer qr.mu.RUnlock()
	return qr.shouldTriggerInvestigationLocked()
}

func (qr *QualityRecovery) shouldTriggerInvestigationLocked() bool {
	rate := qr.calculateDefectRateLocked()
	if rate < qr.defectRateThreshold {
		return false
	}
	if qr.lastInvestigationAt == nil {
		return true
	}
	return time.Since(*qr.lastInvestigationAt) >= qr.investigationCooldown
}

func (qr *QualityRecovery) calculateDefectRateLocked() float64 {
	if len(qr.inspections) == 0 {
		return 0.0
	}
	defectCount := 0
	for _, inspection := range qr.inspections {
		if inspection.IsDefective {
			defectCount++
		}
	}
	return float64(defectCount) / float64(len(qr.inspections))
}

// PredictDefectRisk provides a simple risk score based on SPC chart
func (qr *QualityRecovery) PredictDefectRisk(metric QualityMetric) float64 {
	qr.mu.RLock()
	defer qr.mu.RUnlock()

	chart, exists := qr.spcCharts[metric]
	if !exists || len(chart.Samples) < 20 {
		return 0.1
	}
	if chart.OutOfControl {
		return 0.9
	}
	if chart.StdDev > 0.5*math.Abs(chart.Mean) {
		return 0.6
	}
	return 0.2
}

// BuildQualityDisruptionEvent constructs a disruption event for quality issues
func (qr *QualityRecovery) BuildQualityDisruptionEvent(jobID string, severity int, metadata map[string]interface{}) *models.DisruptionEvent {
	if severity <= 0 {
		severity = 5
	}
	return &models.DisruptionEvent{
		EventID:      fmt.Sprintf("QUALITY_%d", time.Now().UnixNano()),
		EventType:    models.QualityIssue,
		Severity:     severity,
		AffectedAsset: jobID,
		DetectedAt:   time.Now(),
		ImpactedJobs: []string{jobID},
		Metadata:     metadata,
	}
}

// RecordInspection records a quality inspection result
func (qr *QualityRecovery) RecordInspection(inspection *QualityInspection) {
	qr.mu.Lock()
	defer qr.mu.Unlock()
	
	qr.inspections[inspection.InspectionID] = inspection
	
	// Check if measurement is out of tolerance
	deviation := math.Abs(inspection.MeasuredValue - inspection.TargetValue)
	inspection.IsDefective = deviation > inspection.Tolerance
	
	if inspection.IsDefective {
		// Automatically create defect record
		defect := &Defect{
			DefectID:    fmt.Sprintf("DEFECT_%d", time.Now().UnixNano()),
			JobID:       inspection.JobID,
			Type:        qr.classifyDefectType(inspection.Metric),
			Severity:    qr.calculateSeverity(deviation, inspection.Tolerance),
			DetectedAt:  time.Now(),
			AgentID:     inspection.AgentID,
			Description: fmt.Sprintf("%s out of spec: %.3f (target: %.3f ± %.3f)",
				inspection.Metric, inspection.MeasuredValue, inspection.TargetValue, inspection.Tolerance),
			CanRework:   qr.isReworkable(inspection.Metric, deviation),
			ReworkCost:  qr.estimateReworkCost(inspection.Metric),
		}
		
		qr.defects[defect.DefectID] = defect
		log.Printf("[QUALITY] Defect detected: %s (Severity %d)", defect.DefectID, defect.Severity)
		
		// Create rework action if reworkable
		if defect.CanRework {
			qr.createReworkAction(defect)
		}
	}
	
	// Update SPC chart
	qr.updateSPCChart(inspection.Metric, inspection.MeasuredValue)

	// Trigger investigation if defect rate exceeds threshold and cooldown allows
	if qr.shouldTriggerInvestigationLocked() {
		rate := qr.calculateDefectRateLocked()
		for _, cb := range qr.investigationCallbacks {
			go cb(rate)
		}
		now := time.Now()
		qr.lastInvestigationAt = &now
	}
}

// classifyDefectType maps quality metric to defect type
func (qr *QualityRecovery) classifyDefectType(metric QualityMetric) DefectType {
	switch metric {
	case DimensionalAccuracy:
		return DefectDimensional
	case SurfaceFinish:
		return DefectSurface
	case AssemblyAlignment:
		return DefectAssembly
	default:
		return DefectOther
	}
}

// calculateSeverity determines defect severity based on deviation
func (qr *QualityRecovery) calculateSeverity(deviation, tolerance float64) DefectSeverity {
	ratio := deviation / tolerance
	
	if ratio < 1.5 {
		return SeverityMinor
	} else if ratio < 3.0 {
		return SeverityModerate
	} else {
		return SeverityCritical
	}
}

// isReworkable determines if a defect can be fixed
func (qr *QualityRecovery) isReworkable(metric QualityMetric, deviation float64) bool {
	switch metric {
	case DimensionalAccuracy:
		return deviation < 2.0 // Can rework if within 2mm
	case SurfaceFinish:
		return true // Surface can usually be repolished
	case AssemblyAlignment:
		return true // Can reassemble
	default:
		return false
	}
}

// estimateReworkCost estimates cost to fix defect
func (qr *QualityRecovery) estimateReworkCost(metric QualityMetric) float64 {
	costs := map[QualityMetric]float64{
		DimensionalAccuracy: 150.0, // Regrinding cost
		SurfaceFinish:       80.0,  // Repolishing cost
		AssemblyAlignment:   200.0, // Reassembly cost
		VisualDefect:        50.0,
	}
	
	if cost, ok := costs[metric]; ok {
		return cost
	}
	return 100.0
}

// createReworkAction creates a rework action for a defect
func (qr *QualityRecovery) createReworkAction(defect *Defect) {
	action := &ReworkAction{
		ActionID:      fmt.Sprintf("REWORK_%d", time.Now().UnixNano()),
		DefectID:      defect.DefectID,
		JobID:         defect.JobID,
		Action:        qr.selectReworkAction(defect.Type),
		EstimatedTime: qr.estimateReworkTime(defect.Type),
		Cost:          defect.ReworkCost,
		ScheduledAt:   time.Now().Add(5 * time.Minute), // Schedule 5 min later
		Status:        "PENDING",
	}
	
	qr.reworkQueue = append(qr.reworkQueue, action)
	log.Printf("[QUALITY] Rework action created: %s (%s)", action.ActionID, action.Action)
}

// selectReworkAction determines appropriate rework action
func (qr *QualityRecovery) selectReworkAction(defectType DefectType) string {
	actions := map[DefectType]string{
		DefectDimensional: "REGRIND",
		DefectSurface:     "REPOLISH",
		DefectAssembly:    "REASSEMBLE",
		DefectMaterial:    "SCRAP",
		DefectOther:       "MANUAL_INSPECT",
	}
	
	if action, ok := actions[defectType]; ok {
		return action
	}
	return "SCRAP"
}

// estimateReworkTime estimates time required for rework
func (qr *QualityRecovery) estimateReworkTime(defectType DefectType) time.Duration {
	times := map[DefectType]time.Duration{
		DefectDimensional: 30 * time.Minute,
		DefectSurface:     15 * time.Minute,
		DefectAssembly:    45 * time.Minute,
		DefectMaterial:    5 * time.Minute,  // Just scrap
		DefectOther:       20 * time.Minute,
	}
	
	if dur, ok := times[defectType]; ok {
		return dur
	}
	return 30 * time.Minute
}

// updateSPCChart updates Statistical Process Control chart
func (qr *QualityRecovery) updateSPCChart(metric QualityMetric, value float64) {
	chart, exists := qr.spcCharts[metric]
	if !exists {
		chart = &SPCChart{
			Metric:     metric,
			Samples:    make([]float64, 0),
			Timestamps: make([]time.Time, 0),
		}
		qr.spcCharts[metric] = chart
	}
	
	chart.Samples = append(chart.Samples, value)
	chart.Timestamps = append(chart.Timestamps, time.Now())
	
	// Keep last 100 samples
	if len(chart.Samples) > 100 {
		chart.Samples = chart.Samples[len(chart.Samples)-100:]
		chart.Timestamps = chart.Timestamps[len(chart.Timestamps)-100:]
	}
	
	// Recalculate statistics if enough samples
	if len(chart.Samples) >= 20 {
		chart.Mean = qr.calculateMean(chart.Samples)
		chart.StdDev = qr.calculateStdDev(chart.Samples, chart.Mean)
		chart.UCL = chart.Mean + qr.spcSigmaMultiplier*chart.StdDev
		chart.LCL = chart.Mean - qr.spcSigmaMultiplier*chart.StdDev
		
		// Check if latest value is out of control
		chart.OutOfControl = value > chart.UCL || value < chart.LCL
		
		if chart.OutOfControl {
			log.Printf("[QUALITY] SPC Alert: %s out of control (value: %.3f, UCL: %.3f, LCL: %.3f)",
				metric, value, chart.UCL, chart.LCL)
		}
	}
}

// calculateMean calculates average
func (qr *QualityRecovery) calculateMean(samples []float64) float64 {
	sum := 0.0
	for _, val := range samples {
		sum += val
	}
	return sum / float64(len(samples))
}

// calculateStdDev calculates standard deviation
func (qr *QualityRecovery) calculateStdDev(samples []float64, mean float64) float64 {
	variance := 0.0
	for _, val := range samples {
		variance += math.Pow(val-mean, 2)
	}
	variance /= float64(len(samples))
	return math.Sqrt(variance)
}

// CalculateDefectRate calculates overall defect rate
func (qr *QualityRecovery) CalculateDefectRate() float64 {
	qr.mu.RLock()
	defer qr.mu.RUnlock()
	
	if len(qr.inspections) == 0 {
		return 0.0
	}
	
	defectCount := 0
	for _, inspection := range qr.inspections {
		if inspection.IsDefective {
			defectCount++
		}
	}
	
	return float64(defectCount) / float64(len(qr.inspections))
}

// PerformRootCauseAnalysis analyzes defect patterns
func (qr *QualityRecovery) PerformRootCauseAnalysis(defectID string) (*RootCauseAnalysis, error) {
	qr.mu.RLock()
	defer qr.mu.RUnlock()
	
	defect, exists := qr.defects[defectID]
	if !exists {
		return nil, fmt.Errorf("defect %s not found", defectID)
	}
	
	// Analyze recent defects from same agent
	sameAgentDefects := 0
	sameToolDefects := 0
	sameMaterialDefects := 0
	
	for _, d := range qr.defects {
		if d.DefectID == defectID {
			continue
		}
		
		// Check if within last 24 hours
		if time.Since(d.DetectedAt) > 24*time.Hour {
			continue
		}
		
		if d.AgentID == defect.AgentID {
			sameAgentDefects++
		}
		if d.ToolID == defect.ToolID && defect.ToolID != "" {
			sameToolDefects++
		}
		if d.MaterialID == defect.MaterialID && defect.MaterialID != "" {
			sameMaterialDefects++
		}
	}
	
	// Determine root cause
	analysis := &RootCauseAnalysis{
		DefectID:       defectID,
		AffectedJobs:   []string{defect.JobID},
		AffectedAgents: []string{defect.AgentID},
	}
	
	if sameAgentDefects >= 3 {
		analysis.ProbableRoot = "AGENT_MALFUNCTION"
		analysis.Confidence = 0.8
		analysis.RecommendedAction = fmt.Sprintf("Inspect and calibrate agent %s", defect.AgentID)
	} else if sameToolDefects >= 3 {
		analysis.ProbableRoot = "TOOL_WEAR"
		analysis.Confidence = 0.75
		analysis.RecommendedAction = fmt.Sprintf("Replace tool %s", defect.ToolID)
		analysis.AffectedTools = []string{defect.ToolID}
	} else if sameMaterialDefects >= 2 {
		analysis.ProbableRoot = "MATERIAL_VARIANCE"
		analysis.Confidence = 0.7
		analysis.RecommendedAction = fmt.Sprintf("Inspect material batch %s", defect.MaterialID)
	} else {
		analysis.ProbableRoot = "PROCESS_DRIFT"
		analysis.Confidence = 0.5
		analysis.RecommendedAction = "Monitor process parameters"
	}
	
	log.Printf("[QUALITY] Root cause analysis: %s (confidence: %.2f)", 
		analysis.ProbableRoot, analysis.Confidence)
	
	return analysis, nil
}

// GetReworkQueue returns pending rework actions
func (qr *QualityRecovery) GetReworkQueue() []*ReworkAction {
	qr.mu.RLock()
	defer qr.mu.RUnlock()
	
	pending := make([]*ReworkAction, 0)
	for _, action := range qr.reworkQueue {
		if action.Status == "PENDING" {
			pending = append(pending, action)
		}
	}
	
	return pending
}

// CompleteReworkAction marks a rework action as completed
func (qr *QualityRecovery) CompleteReworkAction(actionID string, success bool) error {
	qr.mu.Lock()
	defer qr.mu.Unlock()
	
	for _, action := range qr.reworkQueue {
		if action.ActionID == actionID {
			now := time.Now()
			action.CompletedAt = &now
			if success {
				action.Status = "COMPLETED"
			} else {
				action.Status = "FAILED"
			}
			
			log.Printf("[QUALITY] Rework action %s %s", actionID, action.Status)
			return nil
		}
	}
	
	return fmt.Errorf("rework action %s not found", actionID)
}

// GetSPCChart retrieves SPC chart for a metric
func (qr *QualityRecovery) GetSPCChart(metric QualityMetric) (*SPCChart, bool) {
	qr.mu.RLock()
	defer qr.mu.RUnlock()
	
	chart, exists := qr.spcCharts[metric]
	return chart, exists
}
