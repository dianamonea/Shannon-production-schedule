package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ========== 调度指标 ==========

	// TasksScheduled 调度任务总数
	TasksScheduled = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_tasks_scheduled_total",
			Help: "Total number of tasks scheduled",
		},
		[]string{"agent_type", "operation_type"},
	)

	// TaskDuration 任务执行时长
	TaskDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "shannon_task_duration_seconds",
			Help:    "Task execution duration in seconds",
			Buckets: prometheus.ExponentialBuckets(1, 2, 10), // 1s到512s
		},
		[]string{"agent_id", "operation_type"},
	)

	// TasksFailed 失败任务数
	TasksFailed = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_tasks_failed_total",
			Help: "Total number of failed tasks",
		},
		[]string{"agent_id", "operation_type", "failure_reason"},
	)

	// ========== KPI指标 ==========

	// OEE Overall Equipment Effectiveness
	OEE = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_oee",
			Help: "Overall Equipment Effectiveness (0-1)",
		},
		[]string{"agent_id"},
	)

	// AgentUtilization 智能体利用率
	AgentUtilization = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_agent_utilization",
			Help: "Agent utilization rate (0-1)",
		},
		[]string{"agent_id"},
	)

	// FirstPassYield 一次通过率
	FirstPassYield = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_first_pass_yield",
			Help: "First pass yield rate (0-1)",
		},
		[]string{"agent_id"},
	)

	// OnTimeDelivery 准时交货率
	OnTimeDelivery = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "shannon_on_time_delivery",
			Help: "On-time delivery rate (0-1)",
		},
	)

	// ========== WIP指标 ==========

	// WIPLevel 在制品缓冲区水平
	WIPLevel = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_wip_level",
			Help: "Work-in-progress buffer level",
		},
		[]string{"buffer_id"},
	)

	// WIPBufferBlocked 缓冲区阻塞次数
	WIPBufferBlocked = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_wip_buffer_blocked_total",
			Help: "Total number of times WIP buffer became blocked",
		},
		[]string{"buffer_id"},
	)

	// ========== 库存指标 ==========

	// InventoryLevel 库存水平
	InventoryLevel = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_inventory_level",
			Help: "Current inventory level",
		},
		[]string{"material_id", "location"},
	)

	// MaterialIssued 发料量
	MaterialIssued = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_material_issued_total",
			Help: "Total material issued",
		},
		[]string{"material_id"},
	)

	// ReplenishRequests 补货请求次数
	ReplenishRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_replenish_requests_total",
			Help: "Total number of replenishment requests",
		},
		[]string{"material_id", "urgency"},
	)

	// ========== 重规划指标 ==========

	// ReplansTriggered 重规划触发次数
	ReplansTriggered = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_replans_triggered_total",
			Help: "Total number of replans triggered",
		},
		[]string{"trigger_type", "disruption_type"},
	)

	// ReplanDuration 重规划计算时长
	ReplanDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "shannon_replan_duration_seconds",
			Help:    "Replan computation time in seconds",
			Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 30},
		},
	)

	// ReplanImpact 重规划影响的任务数
	ReplanImpact = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "shannon_replan_impact_jobs",
			Help:    "Number of jobs affected by replan",
			Buckets: []float64{1, 5, 10, 20, 50, 100},
		},
		[]string{"strategy"},
	)

	// ========== 质量指标 ==========

	// QualityInspections 质量检测次数
	QualityInspections = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_quality_inspections_total",
			Help: "Total number of quality inspections",
		},
		[]string{"agent_id", "result"},
	)

	// DefectRate 缺陷率
	DefectRate = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_defect_rate",
			Help: "Defect rate (0-1)",
		},
		[]string{"agent_id", "operation_type"},
	)

	// ========== 成本指标 ==========

	// ProductionCost 生产成本
	ProductionCost = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_production_cost_total",
			Help: "Total production cost",
		},
		[]string{"cost_type"}, // material, labor, energy, overhead
	)

	// CostPerUnit 单位成本
	CostPerUnit = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "shannon_cost_per_unit",
			Help: "Cost per unit produced",
		},
		[]string{"product_type"},
	)

	// ========== 传感器指标 ==========

	// SensorReadings 传感器读数数量
	SensorReadings = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "shannon_sensor_readings_total",
			Help: "Total number of sensor readings",
		},
		[]string{"sensor_id", "sensor_type"},
	)

	// SensorFusionLatency 传感器融合延迟
	SensorFusionLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "shannon_sensor_fusion_latency_ms",
			Help: "Sensor fusion processing latency in milliseconds",
			Buckets: []float64{1, 5, 10, 20, 50, 100},
		},
	)
)

// ========== 便捷记录函数 ==========

// RecordTaskScheduled 记录任务调度
func RecordTaskScheduled(agentType, operationType string) {
	TasksScheduled.WithLabelValues(agentType, operationType).Inc()
}

// RecordTaskDuration 记录任务执行时长
func RecordTaskDuration(agentID, operationType string, durationSeconds float64) {
	TaskDuration.WithLabelValues(agentID, operationType).Observe(durationSeconds)
}

// RecordTaskFailed 记录任务失败
func RecordTaskFailed(agentID, operationType, reason string) {
	TasksFailed.WithLabelValues(agentID, operationType, reason).Inc()
}

// UpdateOEE 更新OEE指标
func UpdateOEE(agentID string, oee float64) {
	OEE.WithLabelValues(agentID).Set(oee)
}

// UpdateAgentUtilization 更新智能体利用率
func UpdateAgentUtilization(agentID string, utilization float64) {
	AgentUtilization.WithLabelValues(agentID).Set(utilization)
}

// UpdateWIPLevel 更新WIP缓冲区水平
func UpdateWIPLevel(bufferID string, level float64) {
	WIPLevel.WithLabelValues(bufferID).Set(level)
}

// RecordWIPBlocked 记录WIP缓冲区阻塞
func RecordWIPBlocked(bufferID string) {
	WIPBufferBlocked.WithLabelValues(bufferID).Inc()
}

// UpdateInventoryLevel 更新库存水平
func UpdateInventoryLevel(materialID, location string, level float64) {
	InventoryLevel.WithLabelValues(materialID, location).Set(level)
}

// RecordMaterialIssued 记录发料
func RecordMaterialIssued(materialID string, quantity float64) {
	MaterialIssued.WithLabelValues(materialID).Add(quantity)
}

// RecordReplenishRequest 记录补货请求
func RecordReplenishRequest(materialID string, urgency int) {
	urgencyStr := "normal"
	if urgency >= 8 {
		urgencyStr = "urgent"
	} else if urgency >= 5 {
		urgencyStr = "high"
	}
	ReplenishRequests.WithLabelValues(materialID, urgencyStr).Inc()
}

// RecordReplanTriggered 记录重规划触发
func RecordReplanTriggered(triggerType, disruptionType string) {
	ReplansTriggered.WithLabelValues(triggerType, disruptionType).Inc()
}

// RecordReplanDuration 记录重规划时长
func RecordReplanDuration(durationSeconds float64) {
	ReplanDuration.Observe(durationSeconds)
}

// RecordReplanImpact 记录重规划影响
func RecordReplanImpact(strategy string, jobsAffected float64) {
	ReplanImpact.WithLabelValues(strategy).Observe(jobsAffected)
}

// RecordQualityInspection 记录质量检测
func RecordQualityInspection(agentID string, passed bool) {
	result := "fail"
	if passed {
		result = "pass"
	}
	QualityInspections.WithLabelValues(agentID, result).Inc()
}

// UpdateDefectRate 更新缺陷率
func UpdateDefectRate(agentID, operationType string, rate float64) {
	DefectRate.WithLabelValues(agentID, operationType).Set(rate)
}

// RecordProductionCost 记录生产成本
func RecordProductionCost(costType string, cost float64) {
	ProductionCost.WithLabelValues(costType).Add(cost)
}

// UpdateCostPerUnit 更新单位成本
func UpdateCostPerUnit(productType string, cost float64) {
	CostPerUnit.WithLabelValues(productType).Set(cost)
}

// RecordSensorReading 记录传感器读数
func RecordSensorReading(sensorID, sensorType string) {
	SensorReadings.WithLabelValues(sensorID, sensorType).Inc()
}

// RecordSensorFusionLatency 记录传感器融合延迟
func RecordSensorFusionLatency(latencyMs float64) {
	SensorFusionLatency.Observe(latencyMs)
}
