# 具身智能多智能体制造系统 - 详细实施路线图

## 文档概述

本路线图基于[EMBODIED_MULTI_AGENT_GAP_ANALYSIS.md](./EMBODIED_MULTI_AGENT_GAP_ANALYSIS.md)识别的12个关键技术缺失，提供**具体的代码级实施方案、文件结构、API设计和集成步骤**。

---

## 阶段1：生产柔性核心补全 (Phase 1 - 4周)

### 优先级1: 材料流管理系统 (Material Flow Management)

**业务价值：** 完善WIP追踪、库存管理、Kanban拉动系统，实现真正的JIT生产

#### 1.1 数据结构设计

**文件：** `go/orchestrator/internal/models/material_flow.go`

```go
package models

import "time"

// InventoryItem 库存项目
type InventoryItem struct {
    MaterialID    string
    MaterialType  string
    Quantity      int
    Unit          string           // kg, pcs, m
    Location      string           // 仓库位置
    Batch         string           // 批次号
    ExpiryDate    *time.Time       // 保质期（可选）
    SafetyStock   int              // 安全库存
    ReorderPoint  int              // 再订购点
    LeadTimeDays  int              // 供应商交货期
    UnitCost      float64
    LastUpdated   time.Time
}

// WIPBuffer 在制品缓冲区
type WIPBuffer struct {
    BufferID      string
    WorkstationID string           // 缓冲区所属工位
    Capacity      int              // 最大容量
    CurrentLevel  int              // 当前数量
    Jobs          []string         // 等待的JobID列表
    FIFO          bool             // 先进先出 vs 优先级队列
    BlockedSince  *time.Time       // 阻塞时间（满载时）
}

// KanbanCard Kanban卡片
type KanbanCard struct {
    KanbanID          string
    PartNumber        string
    Quantity          int
    SourceWorkstation string      // 上游工位
    SinkWorkstation   string      // 下游工位
    Status            KanbanStatus // WAITING, IN_TRANSIT, CONSUMED
    IssuedAt          time.Time
    ConsumedAt        *time.Time
    Priority          int
}

type KanbanStatus string
const (
    KanbanWaiting    KanbanStatus = "WAITING"
    KanbanInTransit  KanbanStatus = "IN_TRANSIT"
    KanbanConsumed   KanbanStatus = "CONSUMED"
)

// MaterialFlowEvent 物料流事件
type MaterialFlowEvent struct {
    EventID       string
    EventType     MaterialEventType
    MaterialID    string
    Quantity      int
    FromLocation  string
    ToLocation    string
    TransportedBy string           // AGV ID
    Timestamp     time.Time
    RelatedJobID  string
}

type MaterialEventType string
const (
    MaterialArrival    MaterialEventType = "ARRIVAL"       // 入库
    MaterialIssued     MaterialEventType = "ISSUED"        // 发料
    MaterialConsumed   MaterialEventType = "CONSUMED"      // 消耗
    MaterialReplenished MaterialEventType = "REPLENISHED" // 补货
    MaterialDefective  MaterialEventType = "DEFECTIVE"    // 不良品
)
```

#### 1.2 库存管理器

**文件：** `go/orchestrator/internal/workflows/inventory_manager.go`

```go
package workflows

import (
    "context"
    "fmt"
    "sync"
    "time"
    "shannon/go/orchestrator/internal/models"
)

type InventoryManager struct {
    mu               sync.RWMutex
    inventory        map[string]*models.InventoryItem  // MaterialID -> Item
    wipBuffers       map[string]*models.WIPBuffer      // BufferID -> Buffer
    kanbanCards      map[string]*models.KanbanCard     // KanbanID -> Card
    replenishChannel chan ReplenishRequest
}

type ReplenishRequest struct {
    MaterialID string
    Quantity   int
    Urgency    int  // 1-10
}

func NewInventoryManager() *InventoryManager {
    return &InventoryManager{
        inventory:        make(map[string]*models.InventoryItem),
        wipBuffers:       make(map[string]*models.WIPBuffer),
        kanbanCards:      make(map[string]*models.KanbanCard),
        replenishChannel: make(chan ReplenishRequest, 100),
    }
}

// CheckAvailability 检查材料是否可用
func (im *InventoryManager) CheckAvailability(materialID string, requiredQty int) (bool, int) {
    im.mu.RLock()
    defer im.mu.RUnlock()
    
    item, exists := im.inventory[materialID]
    if !exists {
        return false, 0
    }
    
    availableQty := item.Quantity - item.SafetyStock
    if availableQty < 0 {
        availableQty = 0
    }
    
    return availableQty >= requiredQty, availableQty
}

// IssueMaterial 发料（减少库存）
func (im *InventoryManager) IssueMaterial(materialID string, qty int, jobID string) error {
    im.mu.Lock()
    defer im.mu.Unlock()
    
    item, exists := im.inventory[materialID]
    if !exists {
        return fmt.Errorf("material %s not found", materialID)
    }
    
    if item.Quantity < qty {
        return fmt.Errorf("insufficient stock: have %d, need %d", item.Quantity, qty)
    }
    
    item.Quantity -= qty
    item.LastUpdated = time.Now()
    
    // 触发再订购检查
    if item.Quantity <= item.ReorderPoint {
        im.replenishChannel <- ReplenishRequest{
            MaterialID: materialID,
            Quantity:   item.ReorderPoint - item.Quantity + item.SafetyStock,
            Urgency:    10,  // 紧急补货
        }
    }
    
    // 记录物料流事件
    event := &models.MaterialFlowEvent{
        EventID:      generateEventID(),
        EventType:    models.MaterialIssued,
        MaterialID:   materialID,
        Quantity:     qty,
        FromLocation: "WAREHOUSE",
        ToLocation:   "WORKSTATION",
        Timestamp:    time.Now(),
        RelatedJobID: jobID,
    }
    
    // TODO: 发送到事件总线
    _ = event
    
    return nil
}

// AddWIPToBuffer 将任务加入在制品缓冲区
func (im *InventoryManager) AddWIPToBuffer(bufferID string, jobID string) error {
    im.mu.Lock()
    defer im.mu.Unlock()
    
    buffer, exists := im.wipBuffers[bufferID]
    if !exists {
        return fmt.Errorf("buffer %s not found", bufferID)
    }
    
    if buffer.CurrentLevel >= buffer.Capacity {
        buffer.BlockedSince = timePtr(time.Now())
        return fmt.Errorf("buffer %s is full (capacity %d)", bufferID, buffer.Capacity)
    }
    
    buffer.Jobs = append(buffer.Jobs, jobID)
    buffer.CurrentLevel++
    buffer.BlockedSince = nil
    
    return nil
}

// IssueKanban 发出Kanban卡片（下游消耗触发）
func (im *InventoryManager) IssueKanban(partNumber string, fromStation, toStation string, qty int) string {
    im.mu.Lock()
    defer im.mu.Unlock()
    
    kanban := &models.KanbanCard{
        KanbanID:          generateKanbanID(),
        PartNumber:        partNumber,
        Quantity:          qty,
        SourceWorkstation: fromStation,
        SinkWorkstation:   toStation,
        Status:            models.KanbanWaiting,
        IssuedAt:          time.Now(),
        Priority:          5,
    }
    
    im.kanbanCards[kanban.KanbanID] = kanban
    
    // TODO: 触发上游生产/配送
    
    return kanban.KanbanID
}

// GetBufferStatus 获取缓冲区状态（用于调度决策）
func (im *InventoryManager) GetBufferStatus(bufferID string) (int, int, error) {
    im.mu.RLock()
    defer im.mu.RUnlock()
    
    buffer, exists := im.wipBuffers[bufferID]
    if !exists {
        return 0, 0, fmt.Errorf("buffer %s not found", bufferID)
    }
    
    return buffer.CurrentLevel, buffer.Capacity, nil
}

func generateEventID() string {
    return fmt.Sprintf("EVT-%d", time.Now().UnixNano())
}

func generateKanbanID() string {
    return fmt.Sprintf("KAN-%d", time.Now().UnixNano())
}

func timePtr(t time.Time) *time.Time {
    return &t
}
```

#### 1.3 集成到调度器

**文件修改：** `go/orchestrator/internal/cnp/orchestrator.go`

```go
// 在Orchestrator结构体中添加
type Orchestrator struct {
    // ... 现有字段
    inventoryMgr *workflows.InventoryManager
}

// 任务分配前检查材料
func (o *Orchestrator) AssignTask(task Task) (*JobAssignment, error) {
    // 1. 材料可用性检查
    for _, materialReq := range task.MaterialRequirements {
        available, stock := o.inventoryMgr.CheckAvailability(materialReq.MaterialID, materialReq.Quantity)
        if !available {
            return nil, fmt.Errorf("insufficient material %s: need %d, have %d", 
                materialReq.MaterialID, materialReq.Quantity, stock)
        }
    }
    
    // 2. WIP缓冲区检查
    bufferID := fmt.Sprintf("BUFFER_%s", task.TargetWorkstation)
    currentLevel, capacity, _ := o.inventoryMgr.GetBufferStatus(bufferID)
    if currentLevel >= capacity {
        return nil, fmt.Errorf("WIP buffer full at %s", task.TargetWorkstation)
    }
    
    // 3. 执行现有CNP逻辑
    assignment, err := o.runCNP(task)
    if err != nil {
        return nil, err
    }
    
    // 4. 发料
    for _, materialReq := range task.MaterialRequirements {
        err := o.inventoryMgr.IssueMaterial(materialReq.MaterialID, materialReq.Quantity, task.JobID)
        if err != nil {
            // 回滚分配
            return nil, err
        }
    }
    
    // 5. 加入WIP缓冲区
    o.inventoryMgr.AddWIPToBuffer(bufferID, task.JobID)
    
    return assignment, nil
}
```

#### 1.4 测试

**文件：** `go/orchestrator/internal/workflows/inventory_manager_test.go`

```go
func TestInventoryManager_IssueMaterial(t *testing.T) {
    im := NewInventoryManager()
    
    // 添加测试库存
    im.inventory["MAT001"] = &models.InventoryItem{
        MaterialID:   "MAT001",
        MaterialType: "Steel Plate",
        Quantity:     100,
        SafetyStock:  20,
        ReorderPoint: 30,
    }
    
    // 测试正常发料
    err := im.IssueMaterial("MAT001", 50, "JOB001")
    assert.NoError(t, err)
    assert.Equal(t, 50, im.inventory["MAT001"].Quantity)
    
    // 测试库存不足
    err = im.IssueMaterial("MAT001", 60, "JOB002")
    assert.Error(t, err)
}

func TestInventoryManager_WIPBuffer(t *testing.T) {
    im := NewInventoryManager()
    
    im.wipBuffers["BUF001"] = &models.WIPBuffer{
        BufferID:     "BUF001",
        Capacity:     5,
        CurrentLevel: 0,
        Jobs:         []string{},
    }
    
    // 添加3个任务
    for i := 1; i <= 3; i++ {
        err := im.AddWIPToBuffer("BUF001", fmt.Sprintf("JOB%03d", i))
        assert.NoError(t, err)
    }
    assert.Equal(t, 3, im.wipBuffers["BUF001"].CurrentLevel)
    
    // 测试缓冲区满
    for i := 4; i <= 5; i++ {
        im.AddWIPToBuffer("BUF001", fmt.Sprintf("JOB%03d", i))
    }
    err := im.AddWIPToBuffer("BUF001", "JOB006")
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "buffer .* is full")
}
```

---

### 优先级2: 动态重规划引擎 (Dynamic Replanning)

**业务价值：** 应对设备故障、物料短缺、紧急插单等扰动，自动触发智能重调度

#### 2.1 扰动事件定义

**文件：** `go/orchestrator/internal/models/disruption.go`

```go
package models

import "time"

type DisruptionEvent struct {
    EventID       string
    EventType     DisruptionType
    Severity      int              // 1-10 (1=minor, 10=critical)
    AffectedAsset string           // AgentID, MaterialID, etc.
    DetectedAt    time.Time
    EstimatedRecoveryTime *time.Duration
    ImpactedJobs  []string         // 受影响的JobID列表
    Metadata      map[string]interface{}
}

type DisruptionType string
const (
    EquipmentFailure   DisruptionType = "EQUIPMENT_FAILURE"
    MaterialShortage   DisruptionType = "MATERIAL_SHORTAGE"
    QualityIssue       DisruptionType = "QUALITY_ISSUE"
    UrgentOrderInsert  DisruptionType = "URGENT_ORDER"
    DeadlineAtRisk     DisruptionType = "DEADLINE_AT_RISK"
    ToolWearExcessive  DisruptionType = "TOOL_WEAR"
    AgentOverloaded    DisruptionType = "AGENT_OVERLOADED"
)

type ReplanTrigger struct {
    TriggerType   TriggerType
    Threshold     float64
    CurrentValue  float64
    TriggeredAt   time.Time
    RelatedEvent  *DisruptionEvent
}

type TriggerType string
const (
    ProgressDeviation  TriggerType = "PROGRESS_DEVIATION"   // 进度偏差>25%
    ResourceFailure    TriggerType = "RESOURCE_FAILURE"     // 设备故障
    ConstraintViolation TriggerType = "CONSTRAINT_VIOLATION" // 约束违反（截止期）
    ManualTrigger      TriggerType = "MANUAL"               // 人工触发
)
```

#### 2.2 重规划决策引擎

**文件：** `go/orchestrator/internal/workflows/scheduling/dynamic_replanning.go`

```go
package scheduling

import (
    "context"
    "fmt"
    "log"
    "shannon/go/orchestrator/internal/models"
    "time"
)

type DynamicReplanner struct {
    currentSchedule  *models.ProductionSchedule
    replanHistory    []ReplanRecord
    replanCostModel  *ReplanCostModel
    triggerThresholds map[models.DisruptionType]float64
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
    MakespanChange    time.Duration  // 正=延长，负=缩短
    CostChange        float64
    AgentsReassigned  int
    InterruptionCost  float64        // 中断正在执行的任务的成本
}

type ReplanCostModel struct {
    TaskInterruptionCost float64  // 中断一个任务的固定成本
    AgentSwitchCost      float64  // 切换智能体的成本
    DelayCostPerHour     float64  // 延期的成本（每小时）
}

func NewDynamicReplanner() *DynamicReplanner {
    return &DynamicReplanner{
        replanHistory: make([]ReplanRecord, 0),
        replanCostModel: &ReplanCostModel{
            TaskInterruptionCost: 50.0,
            AgentSwitchCost:      20.0,
            DelayCostPerHour:     100.0,
        },
        triggerThresholds: map[models.DisruptionType]float64{
            models.EquipmentFailure:  1.0,  // 任何设备故障都触发
            models.UrgentOrderInsert: 1.0,  // 紧急订单都触发
            models.DeadlineAtRisk:    0.8,  // 完成概率<80%触发
        },
    }
}

// EvaluateReplanNeed 评估是否需要重规划
func (dr *DynamicReplanner) EvaluateReplanNeed(event *models.DisruptionEvent) (bool, string) {
    // 1. 检查事件严重性
    if event.Severity >= 8 {
        return true, "Critical severity event"
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
            return true, "Long recovery time for equipment"
        }
    
    case models.UrgentOrderInsert:
        return true, "Urgent order always triggers replan"
    
    case models.DeadlineAtRisk:
        // 检查是否为高优先级订单
        // TODO: 查询订单优先级
        return true, "Deadline violation risk"
    }
    
    // 4. 成本-收益分析
    estimatedReplanCost := dr.estimateReplanCost(event)
    estimatedImproveBenefit := dr.estimateImproveBenefit(event)
    
    if estimatedImproveBenefit > estimatedReplanCost * 1.2 {
        return true, fmt.Sprintf("Benefit (%.2f) > Cost (%.2f)", estimatedImproveBenefit, estimatedReplanCost)
    }
    
    return false, "No replan needed"
}

// ExecuteReplan 执行重规划
func (dr *DynamicReplanner) ExecuteReplan(ctx context.Context, event *models.DisruptionEvent) (*models.ProductionSchedule, error) {
    startTime := time.Now()
    
    // 1. 保存当前调度
    oldSchedule := dr.currentSchedule.Clone()
    
    // 2. 识别受影响任务
    affectedJobs := dr.identifyAffectedJobs(event)
    
    // 3. 选择重规划策略
    strategy := dr.selectReplanStrategy(event, affectedJobs)
    
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
        ReplanID:     generateReplanID(),
        TriggerEvent: event,
        StartTime:    startTime,
        EndTime:      time.Now(),
        OldSchedule:  oldSchedule,
        NewSchedule:  newSchedule,
        ImpactMetrics: dr.calculateImpact(oldSchedule, newSchedule),
    }
    dr.replanHistory = append(dr.replanHistory, record)
    
    log.Printf("Replan completed: %s, Jobs rescheduled: %d, Makespan change: %v", 
        record.ReplanID, record.ImpactMetrics.JobsRescheduled, record.ImpactMetrics.MakespanChange)
    
    dr.currentSchedule = newSchedule
    return newSchedule, nil
}

type ReplanStrategy int
const (
    IncrementalReplan ReplanStrategy = iota  // 仅调整受影响任务
    GlobalReplan                              // 全局重优化
    RightShiftReplan                          // 简单右移（延后所有后续任务）
)

// selectReplanStrategy 选择重规划策略
func (dr *DynamicReplanner) selectReplanStrategy(event *models.DisruptionEvent, affectedJobs []string) ReplanStrategy {
    if len(affectedJobs) <= 3 {
        return IncrementalReplan
    }
    
    if event.Severity >= 9 {
        return GlobalReplan
    }
    
    if event.EventType == models.EquipmentFailure && event.EstimatedRecoveryTime != nil {
        return RightShiftReplan
    }
    
    return IncrementalReplan
}

// incrementalReplan 增量重规划（仅调整受影响任务）
func (dr *DynamicReplanner) incrementalReplan(affectedJobs []string, event *models.DisruptionEvent) (*models.ProductionSchedule, error) {
    newSchedule := dr.currentSchedule.Clone()
    
    // 1. 移除受影响任务的分配
    for _, jobID := range affectedJobs {
        newSchedule.RemoveAssignment(jobID)
    }
    
    // 2. 为受影响任务寻找新分配
    for _, jobID := range affectedJobs {
        job := dr.findJobByID(jobID)
        if job == nil {
            continue
        }
        
        // 寻找可用智能体（排除故障设备）
        availableAgents := dr.getAvailableAgents(event.AffectedAsset)
        
        // 使用启发式算法分配（例如：最早完成时间）
        bestAgent, bestStartTime := dr.findEarliestFinishAgent(job, availableAgents, newSchedule)
        
        if bestAgent != nil {
            assignment := &models.JobAssignment{
                JobID:     jobID,
                AgentID:   bestAgent.AgentID,
                StartTime: bestStartTime,
                EndTime:   bestStartTime.Add(job.EstimatedDuration),
            }
            newSchedule.AddAssignment(assignment)
        }
    }
    
    return newSchedule, nil
}

// globalReplan 全局重规划（所有未完成任务）
func (dr *DynamicReplanner) globalReplan(ctx context.Context) (*models.ProductionSchedule, error) {
    // TODO: 调用优化算法（例如：遗传算法、模拟退火）
    // 这里使用简化版本
    
    unfinishedJobs := dr.currentSchedule.GetUnfinishedJobs()
    
    // 使用贪心算法重新分配
    newSchedule := &models.ProductionSchedule{
        Assignments: make([]*models.JobAssignment, 0),
    }
    
    for _, job := range unfinishedJobs {
        availableAgents := dr.getAllAgents()
        bestAgent, bestStartTime := dr.findEarliestFinishAgent(job, availableAgents, newSchedule)
        
        if bestAgent != nil {
            assignment := &models.JobAssignment{
                JobID:     job.JobID,
                AgentID:   bestAgent.AgentID,
                StartTime: bestStartTime,
                EndTime:   bestStartTime.Add(job.EstimatedDuration),
            }
            newSchedule.AddAssignment(assignment)
        }
    }
    
    return newSchedule, nil
}

// rightShiftReplan 右移策略（延后所有受影响任务）
func (dr *DynamicReplanner) rightShiftReplan(affectedJobs []string, event *models.DisruptionEvent) (*models.ProductionSchedule, error) {
    newSchedule := dr.currentSchedule.Clone()
    
    if event.EstimatedRecoveryTime == nil {
        return nil, fmt.Errorf("recovery time not provided")
    }
    
    delay := *event.EstimatedRecoveryTime
    
    // 将所有受影响任务的开始时间延后
    for _, jobID := range affectedJobs {
        assignment := newSchedule.FindAssignment(jobID)
        if assignment != nil {
            assignment.StartTime = assignment.StartTime.Add(delay)
            assignment.EndTime = assignment.EndTime.Add(delay)
        }
    }
    
    return newSchedule, nil
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
    // 返回event中已标识的受影响任务
    return event.ImpactedJobs
}

func (dr *DynamicReplanner) validateSchedule(schedule *models.ProductionSchedule) bool {
    // TODO: 验证调度合法性（无资源冲突、满足约束等）
    return true
}

func (dr *DynamicReplanner) calculateImpact(old, new *models.ProductionSchedule) ReplanImpact {
    // TODO: 计算调度变化的影响
    return ReplanImpact{
        JobsRescheduled: len(new.Assignments),
    }
}

func (dr *DynamicReplanner) findJobByID(jobID string) *models.JobStep {
    // TODO: 从数据库查询
    return nil
}

func (dr *DynamicReplanner) getAvailableAgents(excludeAgentID string) []*models.ManufacturingAgent {
    // TODO: 查询可用智能体
    return nil
}

func (dr *DynamicReplanner) getAllAgents() []*models.ManufacturingAgent {
    // TODO: 查询所有智能体
    return nil
}

func (dr *DynamicReplanner) findEarliestFinishAgent(job *models.JobStep, agents []*models.ManufacturingAgent, schedule *models.ProductionSchedule) (*models.ManufacturingAgent, time.Time) {
    // TODO: 启发式算法找到最早完成的智能体
    return nil, time.Now()
}

func generateReplanID() string {
    return fmt.Sprintf("REPLAN-%d", time.Now().UnixNano())
}
```

#### 2.3 与反馈控制集成

**文件修改：** `go/orchestrator/internal/control/feedback.go`

```go
// 在ClosedLoopControl中添加重规划触发
func (clc *ClosedLoopControl) MonitorProgress(ctx context.Context) error {
    ticker := time.NewTicker(clc.MonitoringInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return nil
        case <-ticker.C:
            // 现有逻辑：检测偏差
            deviation := clc.detector.CheckDeviation(expected, actual)
            
            if deviation == DeviationCritical {
                // 新增：创建扰动事件
                event := &models.DisruptionEvent{
                    EventID:      generateEventID(),
                    EventType:    models.DeadlineAtRisk,
                    Severity:     8,
                    DetectedAt:   time.Now(),
                    ImpactedJobs: []string{currentJobID},
                }
                
                // 评估是否需要重规划
                needReplan, reason := clc.replanner.EvaluateReplanNeed(event)
                if needReplan {
                    log.Printf("Triggering replan: %s", reason)
                    newSchedule, err := clc.replanner.ExecuteReplan(ctx, event)
                    if err != nil {
                        log.Printf("Replan failed: %v", err)
                    } else {
                        clc.applyNewSchedule(newSchedule)
                    }
                }
            }
        }
    }
}
```

---

## 阶段2：具身智能感知-执行闭环 (Phase 2 - 4周)

### 优先级3: 多模态感知融合 (Multimodal Perception Fusion)

**业务价值：** 实现视觉+力觉融合，支持精密装配、bin picking等复杂任务

#### 3.1 传感器数据结构

**文件：** `rust/agent-core/src/perception/mod.rs`

```rust
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub timestamp: SystemTime,
    pub data: SensorData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorType {
    Camera,
    ForceTorqueSensor,
    LiDAR,
    IMU,
    JointEncoder,
    TactileSensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorData {
    Image(ImageData),
    ForceTorque(ForceTorqueData),
    PointCloud(PointCloudData),
    IMU(IMUData),
    JointState(JointStateData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub encoding: String,  // "rgb8", "bgr8", "mono8"
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceTorqueData {
    pub force: [f64; 3],   // Fx, Fy, Fz (N)
    pub torque: [f64; 3],  // Tx, Ty, Tz (Nm)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudData {
    pub points: Vec<Point3D>,
    pub colors: Option<Vec<RGB>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IMUData {
    pub linear_acceleration: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub orientation: Option<[f64; 4]>,  // Quaternion (w, x, y, z)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointStateData {
    pub positions: Vec<f64>,  // rad
    pub velocities: Vec<f64>, // rad/s
    pub efforts: Vec<f64>,    // Nm
}
```

#### 3.2 传感器融合引擎

**文件：** `rust/agent-core/src/perception/fusion.rs`

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use super::*;

pub struct SensorFusion {
    sensor_buffers: Arc<Mutex<HashMap<String, Vec<SensorReading>>>>,
    time_sync_tolerance: Duration,
    fusion_strategies: HashMap<String, FusionStrategy>,
}

pub enum FusionStrategy {
    VisionForceGrasp,       // 视觉定位 + 力反馈调整
    PointCloudRegistration, // 点云拼接（多相机）
    IMU_EncoderFusion,      // IMU + 编码器融合（轨迹估计）
}

impl SensorFusion {
    pub fn new() -> Self {
        Self {
            sensor_buffers: Arc::new(Mutex::new(HashMap::new())),
            time_sync_tolerance: Duration::from_millis(10),  // 10ms同步窗口
            fusion_strategies: HashMap::new(),
        }
    }
    
    /// 添加传感器读数到缓冲区
    pub fn add_reading(&self, reading: SensorReading) {
        let mut buffers = self.sensor_buffers.lock().unwrap();
        let buffer = buffers.entry(reading.sensor_id.clone()).or_insert(Vec::new());
        buffer.push(reading);
        
        // 保留最近1秒的数据
        let cutoff_time = SystemTime::now() - Duration::from_secs(1);
        buffer.retain(|r| r.timestamp > cutoff_time);
    }
    
    /// 视觉+力觉融合抓取
    pub fn fuse_vision_force_for_grasp(&self) -> Result<GraspPose, String> {
        let buffers = self.sensor_buffers.lock().unwrap();
        
        // 1. 获取最新的视觉数据
        let camera_readings = buffers.get("camera_wrist")
            .ok_or("No camera data")?;
        let latest_image = camera_readings.last()
            .ok_or("No recent image")?;
        
        // 2. 提取物体位姿（调用视觉模型）
        let object_pose = self.detect_object_pose(latest_image)?;
        
        // 3. 获取同步的力觉数据
        let force_readings = buffers.get("ft_sensor")
            .ok_or("No force/torque data")?;
        let synced_force = self.find_synced_reading(
            latest_image.timestamp,
            force_readings,
            self.time_sync_tolerance
        )?;
        
        // 4. 融合：如果检测到接触力，调整抓取姿态
        let adjusted_pose = if let SensorData::ForceTorque(ft) = &synced_force.data {
            let force_magnitude = (ft.force[0].powi(2) + ft.force[1].powi(2) + ft.force[2].powi(2)).sqrt();
            
            if force_magnitude > 1.0 {  // 1N阈值
                // 有接触力 → 视觉可能不准确，使用力反馈微调
                self.adjust_pose_by_force(object_pose, ft)
            } else {
                object_pose
            }
        } else {
            object_pose
        };
        
        Ok(adjusted_pose)
    }
    
    /// 点云配准（多相机融合）
    pub fn fuse_point_clouds(&self, camera_ids: Vec<String>) -> Result<PointCloudData, String> {
        let buffers = self.sensor_buffers.lock().unwrap();
        
        let mut all_points = Vec::new();
        
        for camera_id in camera_ids {
            let readings = buffers.get(&camera_id)
                .ok_or(format!("No data from {}", camera_id))?;
            let latest = readings.last()
                .ok_or(format!("No recent data from {}", camera_id))?;
            
            if let SensorData::PointCloud(pc) = &latest.data {
                // TODO: 应用TF变换（相机坐标系 → 机器人基坐标系）
                // TODO: ICP配准算法
                all_points.extend(pc.points.clone());
            }
        }
        
        Ok(PointCloudData {
            points: all_points,
            colors: None,
        })
    }
    
    // 辅助函数
    fn find_synced_reading(
        &self,
        target_time: SystemTime,
        readings: &[SensorReading],
        tolerance: Duration
    ) -> Result<&SensorReading, String> {
        readings.iter()
            .filter(|r| {
                let diff = r.timestamp.duration_since(target_time)
                    .or_else(|_| target_time.duration_since(r.timestamp))
                    .unwrap_or(Duration::from_secs(100));
                diff < tolerance
            })
            .min_by_key(|r| {
                r.timestamp.duration_since(target_time)
                    .or_else(|_| target_time.duration_since(r.timestamp))
                    .unwrap_or(Duration::from_secs(100))
            })
            .ok_or("No synced reading found".to_string())
    }
    
    fn detect_object_pose(&self, image_reading: &SensorReading) -> Result<GraspPose, String> {
        // TODO: 调用YOLO/Mask R-CNN检测物体边界框
        // TODO: 调用6D姿态估计网络（例如：DenseFusion, PVN3D）
        
        // 占位实现
        Ok(GraspPose {
            position: [0.5, 0.2, 0.1],
            orientation: [1.0, 0.0, 0.0, 0.0],
            confidence: 0.85,
        })
    }
    
    fn adjust_pose_by_force(&self, pose: GraspPose, ft: &ForceTorqueData) -> GraspPose {
        // 简化：根据力方向微调位置
        let force_norm = (ft.force[0].powi(2) + ft.force[1].powi(2) + ft.force[2].powi(2)).sqrt();
        let force_unit = [
            ft.force[0] / force_norm,
            ft.force[1] / force_norm,
            ft.force[2] / force_norm,
        ];
        
        // 沿力方向偏移5mm
        let offset = 0.005;
        GraspPose {
            position: [
                pose.position[0] + force_unit[0] * offset,
                pose.position[1] + force_unit[1] * offset,
                pose.position[2] + force_unit[2] * offset,
            ],
            ..pose
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraspPose {
    pub position: [f64; 3],       // xyz (m)
    pub orientation: [f64; 4],    // Quaternion (w, x, y, z)
    pub confidence: f64,          // 0-1
}
```

#### 3.3 ROS2集成

**文件修改：** `rust/agent-core/src/ros_bridge.rs`

```rust
// 在ROS2Bridge中添加多传感器订阅
impl ROS2Bridge {
    pub fn subscribe_sensors(&self, config: SensorConfig) -> Result<()> {
        // 订阅相机
        for camera_topic in config.camera_topics {
            self.subscribe_image(&camera_topic)?;
        }
        
        // 订阅力觉传感器
        self.subscribe_wrench(&config.ft_topic)?;
        
        // 订阅点云
        if let Some(pc_topic) = config.pointcloud_topic {
            self.subscribe_pointcloud(&pc_topic)?;
        }
        
        Ok(())
    }
    
    fn subscribe_image(&self, topic: &str) -> Result<()> {
        // 创建sensor_msgs/Image订阅者
        // 收到消息后转换为ImageData并发送到融合引擎
        Ok(())
    }
    
    fn subscribe_wrench(&self, topic: &str) -> Result<()> {
        // 创建geometry_msgs/WrenchStamped订阅者
        Ok(())
    }
    
    fn subscribe_pointcloud(&self, topic: &str) -> Result<()> {
        // 创建sensor_msgs/PointCloud2订阅者
        Ok(())
    }
}

pub struct SensorConfig {
    pub camera_topics: Vec<String>,
    pub ft_topic: String,
    pub pointcloud_topic: Option<String>,
}
```

---

### 优先级4: 具身动作空间建模 (Embodied Action Space)

**业务价值：** 统一异构机器人的动作表示，支持技能组合和轨迹学习

#### 4.1 动作原语定义

**文件：** `rust/agent-core/src/skills/primitives.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionPrimitive {
    MoveToCartesianPose(CartesianMove),
    MoveToJointState(JointMove),
    GraspObject(GraspAction),
    ReleaseObject(ReleaseAction),
    ApplyForce(ForceControl),
    FollowTrajectory(TrajectoryAction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartesianMove {
    pub target_pose: Pose,
    pub velocity_limit: Option<f64>,  // m/s
    pub acceleration_limit: Option<f64>,
    pub motion_profile: MotionProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    pub position: [f64; 3],
    pub orientation: Quaternion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointMove {
    pub target_joints: Vec<f64>,  // rad
    pub velocity_limits: Option<Vec<f64>>,
    pub motion_profile: MotionProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotionProfile {
    Trapezoidal,       // 梯形速度曲线
    SCurve,            // S曲线（平滑）
    Linear,            // 恒速
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraspAction {
    pub grasp_pose: Pose,
    pub pre_grasp_offset: f64,  // 预抓取偏移（m）
    pub gripper_width: f64,     // 夹爪开口（m）
    pub grasp_force: f64,       // 抓取力（N）
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAction {
    pub release_pose: Pose,
    pub post_release_offset: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceControl {
    pub target_force: [f64; 3],
    pub compliance_axes: [bool; 6],  // xyz, rpy
    pub duration: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAction {
    pub waypoints: Vec<Waypoint>,
    pub total_duration: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waypoint {
    pub pose: Pose,
    pub timestamp: f64,  // 相对起始时间（秒）
}
```

#### 4.2 技能库管理器

**文件：** `rust/agent-core/src/skills/skill_library.rs`

```rust
use std::collections::HashMap;
use super::primitives::*;

pub struct SkillLibrary {
    skills: HashMap<String, CompositeSkill>,
}

#[derive(Debug, Clone)]
pub struct CompositeSkill {
    pub skill_id: String,
    pub skill_name: String,
    pub parameters: Vec<SkillParameter>,
    pub primitives: Vec<ActionPrimitive>,
}

#[derive(Debug, Clone)]
pub struct SkillParameter {
    pub name: String,
    pub param_type: String,  // "pose", "float", "int"
    pub default_value: Option<String>,
}

impl SkillLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            skills: HashMap::new(),
        };
        
        // 预定义常用技能
        library.register_pick_and_place();
        library.register_screw_tightening();
        library.register_inspection();
        
        library
    }
    
    fn register_pick_and_place(&mut self) {
        let skill = CompositeSkill {
            skill_id: "pick_and_place".to_string(),
            skill_name: "Pick and Place".to_string(),
            parameters: vec![
                SkillParameter {
                    name: "object_pose".to_string(),
                    param_type: "pose".to_string(),
                    default_value: None,
                },
                SkillParameter {
                    name: "target_pose".to_string(),
                    param_type: "pose".to_string(),
                    default_value: None,
                },
            ],
            primitives: vec![
                // 1. 移动到预抓取位置
                ActionPrimitive::MoveToCartesianPose(CartesianMove {
                    target_pose: Pose {  // 占位符，运行时替换
                        position: [0.0, 0.0, 0.0],
                        orientation: Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
                    },
                    velocity_limit: Some(0.1),
                    acceleration_limit: Some(0.5),
                    motion_profile: MotionProfile::SCurve,
                }),
                // 2. 抓取
                ActionPrimitive::GraspObject(GraspAction {
                    grasp_pose: Pose {
                        position: [0.0, 0.0, 0.0],
                        orientation: Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
                    },
                    pre_grasp_offset: 0.05,
                    gripper_width: 0.08,
                    grasp_force: 20.0,
                }),
                // 3. 移动到目标位置
                ActionPrimitive::MoveToCartesianPose(CartesianMove {
                    target_pose: Pose {
                        position: [0.0, 0.0, 0.0],
                        orientation: Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
                    },
                    velocity_limit: Some(0.15),
                    acceleration_limit: Some(0.5),
                    motion_profile: MotionProfile::SCurve,
                }),
                // 4. 释放
                ActionPrimitive::ReleaseObject(ReleaseAction {
                    release_pose: Pose {
                        position: [0.0, 0.0, 0.0],
                        orientation: Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
                    },
                    post_release_offset: 0.05,
                }),
            ],
        };
        
        self.skills.insert(skill.skill_id.clone(), skill);
    }
    
    fn register_screw_tightening(&mut self) {
        // TODO: 螺丝拧紧技能（需要力控）
    }
    
    fn register_inspection(&mut self) {
        // TODO: 视觉检测技能
    }
    
    pub fn get_skill(&self, skill_id: &str) -> Option<&CompositeSkill> {
        self.skills.get(skill_id)
    }
    
    pub fn execute_skill(&self, skill_id: &str, params: HashMap<String, String>) -> Result<(), String> {
        let skill = self.get_skill(skill_id)
            .ok_or(format!("Skill {} not found", skill_id))?;
        
        // 参数替换
        let instantiated_primitives = self.instantiate_primitives(&skill.primitives, params)?;
        
        // 执行每个原语
        for primitive in instantiated_primitives {
            self.execute_primitive(primitive)?;
        }
        
        Ok(())
    }
    
    fn instantiate_primitives(
        &self,
        primitives: &[ActionPrimitive],
        params: HashMap<String, String>
    ) -> Result<Vec<ActionPrimitive>, String> {
        // TODO: 将params中的pose字符串解析并替换primitives中的占位符
        Ok(primitives.to_vec())
    }
    
    fn execute_primitive(&self, primitive: ActionPrimitive) -> Result<(), String> {
        // TODO: 调用ROS2 action server执行动作
        Ok(())
    }
}
```

---

## 阶段3：前沿AI技术集成 (Phase 3 - 3周)

### 优先级8: Diffusion Policy集成

**业务价值：** 已有diffusion_marl.py代码，集成到制造系统实现从演示学习复杂技能

#### 集成步骤

**文件：** `python/shannon/agents/diffusion_integration.py`

```python
import sys
sys.path.append('/path/to/shannon')

from diffusion_marl import DiffusionPolicyAgent, DiffusionConfig
from shannon.core import AgentBase
import numpy as np

class DiffusionSkillAgent(AgentBase):
    """
    集成Diffusion Policy到Shannon智能体框架
    """
    
    def __init__(self, agent_id: str, skill_name: str, model_checkpoint: str):
        super().__init__(agent_id)
        
        # 加载预训练的Diffusion Policy模型
        config = DiffusionConfig(
            action_dim=7,  # 6轴机械臂 + 1夹爪
            observation_dim=15,  # 关节角(7) + 物体位姿(6) + 夹爪状态(2)
            diffusion_steps=50,
            policy_hidden_dims=[256, 256],
        )
        
        self.diffusion_agent = DiffusionPolicyAgent(
            agent_id=f"diffusion_{agent_id}",
            config=config
        )
        
        # 加载模型权重
        self.diffusion_agent.load_checkpoint(model_checkpoint)
        
        self.skill_name = skill_name
    
    async def execute_skill(self, observation: np.ndarray) -> np.ndarray:
        """
        从观测生成动作序列
        
        Args:
            observation: 传感器观测 [关节角, 物体位姿, 夹爪状态]
        
        Returns:
            action: 机器人动作 [关节目标, 夹爪开合]
        """
        # Diffusion policy推理
        action = self.diffusion_agent.act(observation, deterministic=True)
        
        return action
    
    async def collect_demonstration(self, demo_observations: list, demo_actions: list):
        """
        从人工演示收集数据
        """
        for obs, act in zip(demo_observations, demo_actions):
            self.diffusion_agent.store_transition(
                observation=obs,
                action=act,
                reward=1.0,  # 成功演示
                next_observation=obs,  # 占位符
                done=False
            )
    
    async def fine_tune(self, epochs: int = 100):
        """
        在新演示数据上微调模型
        """
        for epoch in range(epochs):
            loss = self.diffusion_agent.train_step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        # 保存更新后的模型
        self.diffusion_agent.save_checkpoint(f"checkpoints/{self.skill_name}_epoch{epochs}.pt")
```

**ROS2桥接：**

**文件：** `rust/agent-core/src/skills/diffusion_bridge.rs`

```rust
// 调用Python Diffusion Policy的FFI桥
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub struct DiffusionBridge {
    py_module: Py<PyModule>,
}

impl DiffusionBridge {
    pub fn new(model_checkpoint: &str) -> Result<Self, String> {
        Python::with_gil(|py| {
            let code = include_str!("../../python/diffusion_wrapper.py");
            let module = PyModule::from_code(py, code, "diffusion_wrapper.py", "diffusion_wrapper")
                .map_err(|e| format!("Failed to load Python module: {}", e))?;
            
            // 初始化模型
            module.call_method1("initialize", (model_checkpoint,))
                .map_err(|e| format!("Failed to initialize model: {}", e))?;
            
            Ok(Self {
                py_module: module.into(),
            })
        })
    }
    
    pub fn infer_action(&self, observation: Vec<f64>) -> Result<Vec<f64>, String> {
        Python::with_gil(|py| {
            let module = self.py_module.as_ref(py);
            let result = module.call_method1("infer", (observation,))
                .map_err(|e| format!("Inference failed: {}", e))?;
            
            let action: Vec<f64> = result.extract()
                .map_err(|e| format!("Failed to extract action: {}", e))?;
            
            Ok(action)
        })
    }
}
```

---

## 阶段4：系统工程完善 (Phase 4 - 2周)

### 优先级9: 可观测性栈

**业务价值：** 生产环境监控、故障诊断、性能优化必需

#### Prometheus指标导出

**文件：** `go/observability/metrics/prometheus_exporter.go`

```go
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // 调度指标
    TasksScheduled = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "shannon_tasks_scheduled_total",
            Help: "Total number of tasks scheduled",
        },
        []string{"agent_type", "operation_type"},
    )
    
    TaskDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "shannon_task_duration_seconds",
            Help:    "Task execution duration",
            Buckets: prometheus.ExponentialBuckets(1, 2, 10),  // 1s到512s
        },
        []string{"agent_id", "operation_type"},
    )
    
    // KPI指标
    OEE = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "shannon_oee",
            Help: "Overall Equipment Effectiveness",
        },
        []string{"agent_id"},
    )
    
    AgentUtilization = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "shannon_agent_utilization",
            Help: "Agent utilization rate (0-1)",
        },
        []string{"agent_id"},
    )
    
    WIPLevel = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "shannon_wip_level",
            Help: "Work-in-progress buffer level",
        },
        []string{"buffer_id"},
    )
    
    // 重规划指标
    ReplansTriggered = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "shannon_replans_triggered_total",
            Help: "Total number of replans triggered",
        },
        []string{"trigger_type"},
    )
    
    ReplanDuration = promauto.NewHistogram(
        prometheus.HistogramOpts{
            Name:    "shannon_replan_duration_seconds",
            Help:    "Replan computation time",
            Buckets: []float64{0.1, 0.5, 1, 2, 5, 10},
        },
    )
)

// 在各模块中调用
func RecordTaskScheduled(agentType, operationType string) {
    TasksScheduled.WithLabelValues(agentType, operationType).Inc()
}

func RecordTaskDuration(agentID, operationType string, duration float64) {
    TaskDuration.WithLabelValues(agentID, operationType).Observe(duration)
}
```

#### Grafana Dashboard JSON

**文件：** `observability/grafana/shannon_manufacturing.json`

```json
{
  "dashboard": {
    "title": "Shannon Manufacturing System",
    "panels": [
      {
        "title": "Tasks Scheduled Rate",
        "targets": [
          {
            "expr": "rate(shannon_tasks_scheduled_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Agent Utilization",
        "targets": [
          {
            "expr": "shannon_agent_utilization"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "OEE by Agent",
        "targets": [
          {
            "expr": "shannon_oee"
          }
        ],
        "type": "bar"
      },
      {
        "title": "WIP Buffer Levels",
        "targets": [
          {
            "expr": "shannon_wip_level"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Replan Frequency",
        "targets": [
          {
            "expr": "rate(shannon_replans_triggered_total[1h])"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

---

## 总结：12个模块的完整实施清单

| # | 模块 | 文件路径 | 代码行数估计 | 工作量（天） | 依赖 |
|---|------|---------|------------|------------|------|
| 1 | 材料流管理 | `go/orchestrator/internal/workflows/` | 1200 | 10 | - |
| 2 | 动态重规划 | `go/orchestrator/internal/workflows/scheduling/` | 1500 | 12 | 1 |
| 3 | 感知融合 | `rust/agent-core/src/perception/` | 1000 | 14 | - |
| 4 | 动作建模 | `rust/agent-core/src/skills/` | 800 | 10 | 3 |
| 5 | 异构协调 | `python/shannon/agents/embodiment_coordinator.py` | 600 | 7 | 3, 4 |
| 6 | 质量闭环 | `go/orchestrator/internal/workflows/quality_recovery.go` | 900 | 10 | 1 |
| 7 | 多目标优化 | `python/shannon/optimization/multi_objective.py` | 700 | 8 | - |
| 8 | Diffusion集成 | `python/shannon/agents/diffusion_integration.py` | 400 | 5 | 4 |
| 9 | 可观测性 | `go/observability/` | 500 | 4 | - |
| 10 | 在线学习 | `python/shannon/learning/` | 1200 | 14 | 1, 2 |
| 11 | 仿真环境 | `python/shannon/simulation/` | 2000 | 21 | 所有 |
| 12 | 通信协议 | `protos/embodiment_communication.proto` | 300 | 3 | - |

**总工作量：** ~120人天 ≈ **4-5个月**（2-3人团队）

**关键路径：** 1 → 2 → 6 → 11（材料流 → 重规划 → 质量 → 仿真验证）
