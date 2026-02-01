package models

import "time"

// DisruptionEvent 扰动事件
type DisruptionEvent struct {
	EventID               string
	EventType             DisruptionType
	Severity              int    // 1-10 (1=minor, 10=critical)
	AffectedAsset         string // AgentID, MaterialID, etc.
	DetectedAt            time.Time
	EstimatedRecoveryTime *time.Duration
	ImpactedJobs          []string // 受影响的JobID列表
	Metadata              map[string]interface{}
}

type DisruptionType string

const (
	EquipmentFailure  DisruptionType = "EQUIPMENT_FAILURE"
	MaterialShortage  DisruptionType = "MATERIAL_SHORTAGE"
	QualityIssue      DisruptionType = "QUALITY_ISSUE"
	UrgentOrderInsert DisruptionType = "URGENT_ORDER"
	DeadlineAtRisk    DisruptionType = "DEADLINE_AT_RISK"
	ToolWearExcessive DisruptionType = "TOOL_WEAR"
	AgentOverloaded   DisruptionType = "AGENT_OVERLOADED"
)

// ReplanTrigger 重规划触发器
type ReplanTrigger struct {
	TriggerType  TriggerType
	Threshold    float64
	CurrentValue float64
	TriggeredAt  time.Time
	RelatedEvent *DisruptionEvent
}

type TriggerType string

const (
	ProgressDeviation   TriggerType = "PROGRESS_DEVIATION"   // 进度偏差>25%
	ResourceFailure     TriggerType = "RESOURCE_FAILURE"     // 设备故障
	ConstraintViolation TriggerType = "CONSTRAINT_VIOLATION" // 约束违反（截止期）
	ManualTrigger       TriggerType = "MANUAL"               // 人工触发
)
