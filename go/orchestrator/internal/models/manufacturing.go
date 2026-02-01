package models

import (
	"time"
)

// Manufacturing Domain Models for Shannon orchestrator
// Aligns agent understanding with industrial manufacturing concepts

// ========== Job and Work Order Models ==========

// WorkOrder represents a production order from business layer
type WorkOrder struct {
	OrderID          string
	CustomerID       string
	OrderDate        time.Time
	RequiredDate     time.Time
	Priority         int          // 1-10, higher=urgent
	Status           OrderStatus
	Jobs             []JobStep
	TotalEstimatedTime time.Duration
	SpecialInstructions string
}

type OrderStatus string

const (
	OrderStatusPending    OrderStatus = "PENDING"
	OrderStatusScheduled  OrderStatus = "SCHEDULED"
	OrderStatusInProgress OrderStatus = "IN_PROGRESS"
	OrderStatusCompleted  OrderStatus = "COMPLETED"
	OrderStatusCanceled   OrderStatus = "CANCELED"
)

// JobStep represents a single manufacturing operation
type JobStep struct {
	JobID              string
	WorkOrderID        string
	StepSequence       int
	OperationType      OperationType  // MILLING, TURNING, ASSEMBLY, etc.
	RequiredTooling    []Tooling
	MaterialType       MaterialType
	MaterialQuantity   float64        // in kg or pieces
	TargetPrecision    float64        // tolerance in μm
	SurfaceFinish      string         // Ra value or finish spec
	EstimatedDuration  time.Duration
	Deadline           time.Time
	Priority           int
	Status             JobStatus
	AssignedAgentID    string         // which robot/cell
	DependencyJobIDs   []string       // must complete after these
	ProcessParameters  map[string]interface{}
	QualityRequirements QualitySpec
}

type JobStatus string

const (
	JobStatusPending     JobStatus = "PENDING"
	JobStatusAssigned    JobStatus = "ASSIGNED"
	JobStatusInProgress  JobStatus = "IN_PROGRESS"
	JobStatusCompleted   JobStatus = "COMPLETED"
	JobStatusFailed      JobStatus = "FAILED"
	JobStatusHeld        JobStatus = "HELD"
)

type OperationType string

const (
	OperationTypeMilling    OperationType = "MILLING"
	OperationTypeTurning    OperationType = "TURNING"
	OperationTypeAssembly   OperationType = "ASSEMBLY"
	OperationTypeInspection OperationType = "INSPECTION"
	OperationTypeWashing    OperationType = "WASHING"
	OperationTypePainting   OperationType = "PAINTING"
	OperationTypeSoldering  OperationType = "SOLDERING"
	OperationTypeOther      OperationType = "OTHER"
)

// ========== Material and BOM Models ==========

// BOM (Bill of Materials)
type BillOfMaterials struct {
	BOMID       string
	JobID       string
	Materials   []MaterialLine
	TotalWeight float64
	TotalCost   float64
}

// MaterialLine represents one line item in BOM
type MaterialLine struct {
	MaterialID    string
	MaterialType  MaterialType
	Quantity      float64
	Unit          string  // kg, pieces, m, etc.
	UnitCost      float64
	Supplier      string
	LeadTimeDays  int
	CriticalPath  bool    // material affects critical path
}

type MaterialType string

const (
	MaterialTypeSteel      MaterialType = "STEEL"
	MaterialTypeAluminum   MaterialType = "ALUMINUM"
	MaterialTypeTitanium   MaterialType = "TITANIUM"
	MaterialTypeComposite  MaterialType = "COMPOSITE"
	MaterialTypePlastic    MaterialType = "PLASTIC"
	MaterialTypeOther      MaterialType = "OTHER"
)

// ========== Tooling and Equipment Models ==========

// Tooling represents cutting tools, fixtures, etc.
type Tooling struct {
	ToolID           string
	ToolType         string         // endmill, drill, saw, etc.
	Material         string         // carbide, HSS, ceramic
	Size             string         // diameter, length, etc.
	MaxRPM           float64
	CuttingFluid     string
	Lifespan         int64          // cycles before replacement
	CycleCost        float64        // cost per cycle
}

// ResourceConstraint represents capacity, tool, or material limits
type ResourceConstraint struct {
	ConstraintID    string
	ResourceType    ConstraintType
	ResourceID      string         // robot_id, tool_id, material_id
	AvailableQty    float64
	MaxConsumption  float64        // per time unit
	RechargeTime    time.Duration  // how long to replenish
	CriticalThreshold float64      // alert level
}

type ConstraintType string

const (
	ConstraintTypeToolInventory    ConstraintType = "TOOL_INVENTORY"
	ConstraintTypeMaterialStock    ConstraintType = "MATERIAL_STOCK"
	ConstraintTypeRobotCapacity    ConstraintType = "ROBOT_CAPACITY"
	ConstraintTypeWorkstationQueue ConstraintType = "WORKSTATION_QUEUE"
	ConstraintTypePowerBudget      ConstraintType = "POWER_BUDGET"
)

// ========== Agent/Robot State Models ==========

// ManufacturingAgent represents an edge robot or manufacturing cell
type ManufacturingAgent struct {
	AgentID              string
	AgentType            AgentType
	Location             string
	Status               AgentStatus
	CurrentLoad          float64        // 0.0-1.0
	ActiveJobIDs         []string
	AvailableTools       map[string]Tooling
	MaterialStock        map[string]float64
	EstimatedIdleTime    time.Duration
	LastMaintenanceTime  time.Time
	MaintenanceInterval  time.Duration
	FaultHistory         []FaultRecord
	PerformanceMetrics   PerformanceMetrics
}

type AgentStatus string

const (
	AgentStatusIdle        AgentStatus = "IDLE"
	AgentStatusProcessing  AgentStatus = "PROCESSING"
	AgentStatusMaintenance AgentStatus = "MAINTENANCE"
	AgentStatusFault       AgentStatus = "FAULT"
	AgentStatusOffline     AgentStatus = "OFFLINE"
)

type AgentType string

const (
	AgentTypeRoboticArm    AgentType = "ROBOTIC_ARM"
	AgentTypeCNCMachine    AgentType = "CNC_MACHINE"
	AgentTypeAssemblyCell  AgentType = "ASSEMBLY_CELL"
	AgentTypeInspectionCell AgentType = "INSPECTION_CELL"
	AgentTypeAGV           AgentType = "AGV"
	AgentTypeOther         AgentType = "OTHER"
)

// FaultRecord tracks equipment failures
type FaultRecord struct {
	FaultID       string
	FaultTime     time.Time
	FaultType     string
	Severity      int        // 1-5
	Description   string
	ResolutionTime time.Duration
	Reason        string
}

// PerformanceMetrics tracks agent KPIs
type PerformanceMetrics struct {
	TotalJobsCompleted      int64
	SuccessRate             float64   // 0.0-1.0
	AverageJobDuration      time.Duration
	AverageQualityScore     float64   // 0.0-1.0
	UtilizationRate         float64   // 0.0-1.0
	MTBF                    time.Duration // Mean Time Between Failures
	MTTR                    time.Duration // Mean Time To Repair
	TotalDowntimeHours      float64
	EnergyPerJob            float64   // kWh
	CostPerJob              float64
	LastUpdated             time.Time
}

// ========== Quality and Inspection Models ==========

// QualitySpec defines quality requirements for a job
type QualitySpec struct {
	DimensionalTolerance   float64       // μm
	SurfaceRoughnessRa     float64       // Ra value
	SurfaceRoughnessRz     float64       // Rz value
	Flatness               float64       // μm
	Perpendicularity       float64       // μm
	Parallelism            float64       // μm
	InspectionMethod       string        // CMM, vision, tactile
	SamplingRate           float64       // 0.0-1.0
	AcceptanceRate         float64       // 0.0-1.0 (min)
	CertificationRequired  bool
	DocumentationRequired  bool
}

// InspectionResult records quality check outcome
type InspectionResult struct {
	InspectionID    string
	JobID           string
	InspectionTime  time.Time
	Inspector       string
	Status          InspectionStatus
	MeasuredValues  map[string]float64
	PassedChecks    int
	FailedChecks    int
	Comments        string
	CertificatePath string
}

type InspectionStatus string

const (
	InspectionStatusPass  InspectionStatus = "PASS"
	InspectionStatusFail  InspectionStatus = "FAIL"
	InspectionStatusHold  InspectionStatus = "HOLD"
)

// ========== Production Schedule Models ==========

// ProductionSchedule represents the optimized assignment of jobs to agents
type ProductionSchedule struct {
	ScheduleID        string
	ScheduleVersion   int
	CreatedAt         time.Time
	GeneratedTime     time.Time
	ValidFrom         time.Time
	ValidUntil        time.Time
	Assignments       []*JobAssignment
	EstimatedMakespan time.Duration
	EstimatedCost     float64
	QualityMetrics    ScheduleQuality
	Feasible          bool
	ConstraintViolations []string
}

// JobAssignment maps a job to an agent and time window
type JobAssignment struct {
	AssignmentID       string
	JobID              string
	AgentID            string
	StartTime          time.Time
	EndTime            time.Time
	ScheduledStartTime time.Time
	ScheduledEndTime   time.Time
	EstimatedDuration  time.Duration
	Priority           int
	PredecessorJobs    []string
	SuccessorJobs      []string
}

// RemoveAssignment removes a job assignment from the schedule
func (ps *ProductionSchedule) RemoveAssignment(jobID string) {
	if ps.Assignments == nil {
		return
	}
	
	for i, assignment := range ps.Assignments {
		if assignment != nil && assignment.JobID == jobID {
			ps.Assignments = append(ps.Assignments[:i], ps.Assignments[i+1:]...)
			return
		}
	}
}

// AddAssignment adds a new job assignment to the schedule
func (ps *ProductionSchedule) AddAssignment(assignment *JobAssignment) {
	if ps.Assignments == nil {
		ps.Assignments = make([]*JobAssignment, 0)
	}
	ps.Assignments = append(ps.Assignments, assignment)
}

// FindAssignment finds an assignment by job ID
func (ps *ProductionSchedule) FindAssignment(jobID string) *JobAssignment {
	if ps.Assignments == nil {
		return nil
	}
	
	for _, assignment := range ps.Assignments {
		if assignment != nil && assignment.JobID == jobID {
			return assignment
		}
	}
	return nil
}

// GetUnfinishedJobs returns job IDs that are not completed
func (ps *ProductionSchedule) GetUnfinishedJobs() []string {
	if ps.Assignments == nil {
		return []string{}
	}
	
	unfinished := make([]string, 0)
	now := time.Now()
	
	for _, assignment := range ps.Assignments {
		if assignment != nil && assignment.EndTime.After(now) {
			unfinished = append(unfinished, assignment.JobID)
		}
	}
	
	return unfinished
}

// ScheduleQuality metrics for the generated schedule
type ScheduleQuality struct {
	MakespanHours        float64       // total production time
	AverageWaitTime      time.Duration // avg wait before job starts
	RobotIdleTime        map[string]time.Duration // idle per robot
	ToolChangeCount      int           // number of tool changes
	MaterialWastePercent float64
	EstimatedEnergyCost  float64
	OptimalityGap        float64       // 0.0 = optimal
}

// ========== Disruption and Recovery Models ==========

// DisruptionEvent represents unexpected production changes
type DisruptionEvent struct {
	EventID           string
	EventTime         time.Time
	DisruptionType    DisruptionType
	Severity          int           // 1-5
	AffectedJobIDs    []string
	AffectedAgentIDs  []string
	EstimatedImpact   DisruptionImpact
	ResolutionStatus  string        // PENDING, IN_PROGRESS, RESOLVED
}

type DisruptionType string

const (
	DisruptionTypeEquipmentFailure  DisruptionType = "EQUIPMENT_FAILURE"
	DisruptionTypeMaterialShortage  DisruptionType = "MATERIAL_SHORTAGE"
	DisruptionTypeOrderInsertion    DisruptionType = "ORDER_INSERTION"
	DisruptionTypeOrderCancellation DisruptionType = "ORDER_CANCELLATION"
	DisruptionTypeQualityIssue      DisruptionType = "QUALITY_ISSUE"
)

// DisruptionImpact quantifies the effect
type DisruptionImpact struct {
	DelayedJobs    []string
	RescheduleRequired bool
	EstimatedDelay time.Duration
	CostIncrease   float64
}

// ========== Utility Functions ==========

// GetAgentStatusColor returns color for UI visualization
func GetAgentStatusColor(status AgentStatus) string {
	switch status {
	case AgentStatusIdle:
		return "#4CAF50"  // green
	case AgentStatusProcessing:
		return "#2196F3"  // blue
	case AgentStatusMaintenance:
		return "#FF9800"  // orange
	case AgentStatusFault:
		return "#F44336"  // red
	case AgentStatusOffline:
		return "#757575"  // grey
	default:
		return "#999999"
	}
}

// EstimateCompletionTime calculates total time including setup and inspection
func EstimateCompletionTime(job JobStep, agent ManufacturingAgent) time.Duration {
	setupTime := 5 * time.Minute      // tool setup
	inspectionTime := 5 * time.Minute // quality check
	cleanupTime := 2 * time.Minute    // cleanup
	
	return setupTime + job.EstimatedDuration + inspectionTime + cleanupTime
}
