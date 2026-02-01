package scheduling

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/models"
	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/worldmodel"
)

type DynamicReplanner struct {
	currentSchedule   *models.ProductionSchedule
	replanHistory     []ReplanRecord
	replanCostModel   *ReplanCostModel
	triggerThresholds map[models.DisruptionType]float64
	worldModel        *worldmodel.WorldModel
}

type ReplanRecord struct {
	ReplanID      string
	TriggerEvent  *models.DisruptionEvent
	StartTime     time.Time
	EndTime       time.Time
	OldSchedule   *models.ProductionSchedule
	NewSchedule   *models.ProductionSchedule
	ImpactMetrics ReplanImpact
}

type ReplanImpact struct {
	JobsRescheduled   int
	MakespanChange    time.Duration // 正=延长，负=缩短
	CostChange        float64
	AgentsReassigned  int
	InterruptionCost  float64 // 中断正在执行的任务的成本
}

type ReplanCostModel struct {
	TaskInterruptionCost float64 // 中断一个任务的固定成本
	AgentSwitchCost      float64 // 切换智能体的成本
	DelayCostPerHour     float64 // 延期的成本（每小时）
}

func NewDynamicReplanner() *DynamicReplanner {
	return &DynamicReplanner{
		replanHistory: make([]ReplanRecord, 0, 100),
		replanCostModel: &ReplanCostModel{
			TaskInterruptionCost: 50.0,
			AgentSwitchCost:      20.0,
			DelayCostPerHour:     100.0,
		},
		triggerThresholds: map[models.DisruptionType]float64{
			models.EquipmentFailure:  1.0, // 任何设备故障都触发
			models.UrgentOrderInsert: 1.0, // 紧急订单都触发
			models.DeadlineAtRisk:    0.8, // 完成概率<80%触发
			models.MaterialShortage:  0.9, // 严重物料短缺触发
			models.AgentOverloaded:   0.85,
		},
	}
}

// SetWorldModel injects a world model for spatial risk evaluation
func (dr *DynamicReplanner) SetWorldModel(wm *worldmodel.WorldModel) {
	dr.worldModel = wm
}

// UpdateCostModel allows online learning to adjust replan costs
func (dr *DynamicReplanner) UpdateCostModel(model *ReplanCostModel) {
	if model == nil {
		return
	}
	dr.replanCostModel = model
}

// EvaluateReplanNeed 评估是否需要重规划
func (dr *DynamicReplanner) EvaluateReplanNeed(event *models.DisruptionEvent) (bool, string) {
	if event == nil {
		return false, "No replan needed - nil event"
	}

	// 1. 检查事件严重性
	if event.Severity >= 8 {
		return true, fmt.Sprintf("Critical severity event (severity=%d)", event.Severity)
	}

	// 2. 检查受影响任务数量
	if len(event.ImpactedJobs) > 5 {
		return true, fmt.Sprintf("%d jobs impacted", len(event.ImpactedJobs))
	}

	// 3. 类型特定逻辑
	switch event.EventType {
	case models.EquipmentFailure:
		// 检查恢复时间
		if event.EstimatedRecoveryTime != nil && *event.EstimatedRecoveryTime > 2*time.Hour {
			return true, fmt.Sprintf("Long recovery time: %v", *event.EstimatedRecoveryTime)
		}

	case models.UrgentOrderInsert:
		return true, "Urgent order always triggers replan"

	case models.DeadlineAtRisk:
		return true, "Deadline violation risk"

	case models.MaterialShortage:
		if event.Severity >= 7 {
			return true, "Critical material shortage"
		}

	case models.AgentOverloaded:
		if len(event.ImpactedJobs) > 3 {
			return true, "Agent overload affecting multiple jobs"
		}
	}

	// 3.5 空间风险评估（可选）
	if dr.worldModel != nil {
		objectIDs := []string{}
		if event.Metadata != nil {
			objectIDs = parseStringSlice(event.Metadata["object_ids"])
		}
		if len(objectIDs) == 0 {
			objectIDs = dr.worldModel.GetActiveObjectIDs()
		}
		if len(objectIDs) >= 2 {
			horizon := 2.0
			threshold := 0.5
			if event.Metadata != nil {
				horizon = parseFloat(event.Metadata["collision_horizon_sec"], horizon)
				threshold = parseFloat(event.Metadata["collision_threshold_m"], threshold)
			}
			if len(objectIDs) > 50 {
				objectIDs = objectIDs[:50]
			}
			risks := dr.worldModel.DetectCollisionRisks(objectIDs, time.Duration(horizon*float64(time.Second)), threshold)
			if len(risks) > 0 {
				first := risks[0]
				return true, fmt.Sprintf("Collision risk: %s vs %s (dist=%.3f)", first.ObjectA, first.ObjectB, first.Distance)
			}
		}
	}

	// 4. 成本-收益分析
	estimatedReplanCost := dr.estimateReplanCost(event)
	estimatedImproveBenefit := dr.estimateImproveBenefit(event)

	if estimatedImproveBenefit > estimatedReplanCost*1.2 {
		return true, fmt.Sprintf("Benefit (%.2f) > Cost (%.2f) * 1.2", estimatedImproveBenefit, estimatedReplanCost)
	}

	return false, "No replan needed - cost/benefit analysis failed"
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

// ExecuteReplan 执行重规划
func (dr *DynamicReplanner) ExecuteReplan(ctx context.Context, event *models.DisruptionEvent) (*models.ProductionSchedule, error) {
	startTime := time.Now()

	// 1. 保存当前调度
	oldSchedule := dr.cloneSchedule(dr.currentSchedule)

	// 2. 识别受影响任务
	affectedJobs := event.ImpactedJobs
	if len(affectedJobs) == 0 {
		affectedJobs = dr.identifyAffectedJobs(event)
	}

	log.Printf("[REPLAN] Starting replan for %d affected jobs, trigger: %s, severity: %d",
		len(affectedJobs), event.EventType, event.Severity)

	// 3. 选择重规划策略
	strategy := dr.selectReplanStrategy(event, affectedJobs)
	log.Printf("[REPLAN] Selected strategy: %s", strategy)

	// 4. 执行重调度算法
	var newSchedule *models.ProductionSchedule
	var err error

	switch strategy {
	case IncrementalReplan:
		newSchedule, err = dr.incrementalReplan(affectedJobs, event)
	case GlobalReplan:
		newSchedule, err = dr.globalReplan(ctx)
	case RightShiftReplan:
		newSchedule, err = dr.rightShiftReplan(affectedJobs, event)
	default:
		return nil, fmt.Errorf("unknown replan strategy: %v", strategy)
	}

	if err != nil {
		return nil, fmt.Errorf("replan failed: %w", err)
	}

	// 5. 验证新调度
	if !dr.validateSchedule(newSchedule) {
		return nil, fmt.Errorf("new schedule validation failed")
	}

	// 6. 记录重规划
	record := ReplanRecord{
		ReplanID:      dr.generateReplanID(),
		TriggerEvent:  event,
		StartTime:     startTime,
		EndTime:       time.Now(),
		OldSchedule:   oldSchedule,
		NewSchedule:   newSchedule,
		ImpactMetrics: dr.calculateImpact(oldSchedule, newSchedule),
	}
	dr.replanHistory = append(dr.replanHistory, record)

	log.Printf("[REPLAN] Completed: %s, Jobs rescheduled: %d, Makespan change: %v, Duration: %v",
		record.ReplanID, record.ImpactMetrics.JobsRescheduled,
		record.ImpactMetrics.MakespanChange, record.EndTime.Sub(record.StartTime))

	dr.currentSchedule = newSchedule
	return newSchedule, nil
}

type ReplanStrategy int

const (
	IncrementalReplan ReplanStrategy = iota // 仅调整受影响任务
	GlobalReplan                             // 全局重优化
	RightShiftReplan                         // 简单右移（延后所有后续任务）
)

func (rs ReplanStrategy) String() string {
	switch rs {
	case IncrementalReplan:
		return "INCREMENTAL"
	case GlobalReplan:
		return "GLOBAL"
	case RightShiftReplan:
		return "RIGHT_SHIFT"
	default:
		return "UNKNOWN"
	}
}

// selectReplanStrategy 选择重规划策略
func (dr *DynamicReplanner) selectReplanStrategy(event *models.DisruptionEvent, affectedJobs []string) ReplanStrategy {
	// 少量任务受影响 → 增量重规划
	if len(affectedJobs) <= 3 {
		return IncrementalReplan
	}

	// 严重扰动 → 全局重优化
	if event.Severity >= 9 {
		return GlobalReplan
	}

	// 设备故障且有明确恢复时间 → 右移策略
	if event.EventType == models.EquipmentFailure && event.EstimatedRecoveryTime != nil {
		return RightShiftReplan
	}

	// 紧急订单插单 → 全局重优化
	if event.EventType == models.UrgentOrderInsert {
		return GlobalReplan
	}

	// 默认增量重规划
	return IncrementalReplan
}

// incrementalReplan 增量重规划（仅调整受影响任务）
func (dr *DynamicReplanner) incrementalReplan(affectedJobs []string, event *models.DisruptionEvent) (*models.ProductionSchedule, error) {
	newSchedule := dr.cloneSchedule(dr.currentSchedule)

	// 1. 移除受影响任务的分配
	for _, jobID := range affectedJobs {
		newSchedule.RemoveAssignment(jobID)
	}

	// 2. 为受影响任务寻找新分配
	reassignedCount := 0
	for _, jobID := range affectedJobs {
		// TODO: 从数据库查询JobStep
		// 这里使用模拟数据
		assignment := dr.findBestAssignment(jobID, event.AffectedAsset, newSchedule)
		if assignment != nil {
			newSchedule.AddAssignment(assignment)
			reassignedCount++
		} else {
			log.Printf("[REPLAN] WARNING: Could not reassign job %s", jobID)
		}
	}

	log.Printf("[REPLAN] Incremental: Reassigned %d/%d jobs", reassignedCount, len(affectedJobs))

	return newSchedule, nil
}

// globalReplan 全局重规划（所有未完成任务）
func (dr *DynamicReplanner) globalReplan(ctx context.Context) (*models.ProductionSchedule, error) {
	log.Printf("[REPLAN] Global replan: Rescheduling all unfinished jobs")

	// TODO: 调用优化算法（例如：遗传算法、模拟退火）
	// 这里使用简化的贪心算法

	newSchedule := &models.ProductionSchedule{
		ScheduleID:  dr.generateScheduleID(),
		CreatedAt:   time.Now(),
		Assignments: make([]*models.JobAssignment, 0),
	}

	// 获取所有未完成任务
	unfinishedJobs := dr.getUnfinishedJobs()
	log.Printf("[REPLAN] Found %d unfinished jobs", len(unfinishedJobs))

	// 使用贪心算法重新分配
	for _, jobID := range unfinishedJobs {
		assignment := dr.findBestAssignment(jobID, "", newSchedule)
		if assignment != nil {
			newSchedule.AddAssignment(assignment)
		}
	}

	return newSchedule, nil
}

// rightShiftReplan 右移策略（延后所有受影响任务）
func (dr *DynamicReplanner) rightShiftReplan(affectedJobs []string, event *models.DisruptionEvent) (*models.ProductionSchedule, error) {
	newSchedule := dr.cloneSchedule(dr.currentSchedule)

	if event.EstimatedRecoveryTime == nil {
		return nil, fmt.Errorf("recovery time not provided for right shift strategy")
	}

	delay := *event.EstimatedRecoveryTime
	log.Printf("[REPLAN] Right shift: Delaying %d jobs by %v", len(affectedJobs), delay)

	// 将所有受影响任务的开始时间延后
	shiftedCount := 0
	for _, jobID := range affectedJobs {
		assignment := newSchedule.FindAssignment(jobID)
		if assignment != nil {
			assignment.StartTime = assignment.StartTime.Add(delay)
			assignment.EndTime = assignment.EndTime.Add(delay)
			shiftedCount++
		}
	}

	log.Printf("[REPLAN] Right shift: Shifted %d jobs", shiftedCount)

	return newSchedule, nil
}

// SetCurrentSchedule 设置当前调度
func (dr *DynamicReplanner) SetCurrentSchedule(schedule *models.ProductionSchedule) {
	dr.currentSchedule = schedule
}

// GetReplanHistory 获取重规划历史
func (dr *DynamicReplanner) GetReplanHistory() []ReplanRecord {
	return dr.replanHistory
}

// 辅助函数

func (dr *DynamicReplanner) estimateReplanCost(event *models.DisruptionEvent) float64 {
	baseCost := float64(len(event.ImpactedJobs)) * dr.replanCostModel.TaskInterruptionCost
	return baseCost
}

func (dr *DynamicReplanner) estimateImproveBenefit(event *models.DisruptionEvent) float64 {
	// 简化：假设改进收益与严重性成正比
	return float64(event.Severity) * 100.0
}

func (dr *DynamicReplanner) identifyAffectedJobs(event *models.DisruptionEvent) []string {
	// TODO: 基于event类型和受影响资产查询数据库
	// 这里返回空切片，让调用者处理
	return []string{}
}

func (dr *DynamicReplanner) validateSchedule(schedule *models.ProductionSchedule) bool {
	if schedule == nil {
		return false
	}
	// TODO: 验证调度合法性（无资源冲突、满足约束等）
	return true
}

func (dr *DynamicReplanner) calculateImpact(old, new *models.ProductionSchedule) ReplanImpact {
	if old == nil || new == nil {
		return ReplanImpact{}
	}

	impact := ReplanImpact{
		JobsRescheduled: len(new.Assignments),
	}

	// 计算makespan变化
	if len(old.Assignments) > 0 && len(new.Assignments) > 0 {
		oldMakespan := dr.calculateMakespan(old)
		newMakespan := dr.calculateMakespan(new)
		impact.MakespanChange = newMakespan - oldMakespan
	}

	return impact
}

func (dr *DynamicReplanner) calculateMakespan(schedule *models.ProductionSchedule) time.Duration {
	if len(schedule.Assignments) == 0 {
		return 0
	}

	var maxEndTime time.Time
	for _, assignment := range schedule.Assignments {
		if assignment.EndTime.After(maxEndTime) {
			maxEndTime = assignment.EndTime
		}
	}

	if len(schedule.Assignments) > 0 && schedule.Assignments[0] != nil {
		return maxEndTime.Sub(schedule.Assignments[0].StartTime)
	}

	return 0
}

func (dr *DynamicReplanner) cloneSchedule(schedule *models.ProductionSchedule) *models.ProductionSchedule {
	if schedule == nil {
		return nil
	}

	clone := &models.ProductionSchedule{
		ScheduleID:  schedule.ScheduleID + "_clone",
		CreatedAt:   time.Now(),
		Assignments: make([]*models.JobAssignment, len(schedule.Assignments)),
	}

	for i, assignment := range schedule.Assignments {
		clone.Assignments[i] = &models.JobAssignment{
			AssignmentID: assignment.AssignmentID,
			JobID:        assignment.JobID,
			AgentID:      assignment.AgentID,
			StartTime:    assignment.StartTime,
			EndTime:      assignment.EndTime,
		}
	}

	return clone
}

func (dr *DynamicReplanner) findBestAssignment(jobID string, excludeAgentID string, schedule *models.ProductionSchedule) *models.JobAssignment {
	// TODO: 实现启发式算法找到最佳分配
	// 这里返回模拟的分配
	return &models.JobAssignment{
		AssignmentID: fmt.Sprintf("ASSIGN-%s-%d", jobID, time.Now().UnixNano()),
		JobID:        jobID,
		AgentID:      "AGENT_BACKUP",
		StartTime:    time.Now().Add(1 * time.Hour),
		EndTime:      time.Now().Add(2 * time.Hour),
	}
}

func (dr *DynamicReplanner) getUnfinishedJobs() []string {
	// TODO: 从数据库查询未完成任务
	return []string{}
}

func (dr *DynamicReplanner) generateReplanID() string {
	return fmt.Sprintf("REPLAN-%d", time.Now().UnixNano())
}

func (dr *DynamicReplanner) generateScheduleID() string {
	return fmt.Sprintf("SCHEDULE-%d", time.Now().UnixNano())
}
