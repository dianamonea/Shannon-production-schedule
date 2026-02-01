package scheduling

import (
	"context"
	"testing"
	"time"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/models"
	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/worldmodel"
	"github.com/stretchr/testify/assert"
)

func TestDynamicReplanner_EvaluateReplanNeed(t *testing.T) {
	dr := NewDynamicReplanner()

	tests := []struct {
		name           string
		event          *models.DisruptionEvent
		expectedNeed   bool
		expectedReason string
	}{
		{
			name: "Critical severity",
			event: &models.DisruptionEvent{
				EventType: models.EquipmentFailure,
				Severity:  9,
			},
			expectedNeed:   true,
			expectedReason: "Critical severity",
		},
		{
			name: "Many jobs impacted",
			event: &models.DisruptionEvent{
				EventType:    models.MaterialShortage,
				Severity:     5,
				ImpactedJobs: []string{"J1", "J2", "J3", "J4", "J5", "J6"},
			},
			expectedNeed:   true,
			expectedReason: "6 jobs impacted",
		},
		{
			name: "Urgent order",
			event: &models.DisruptionEvent{
				EventType: models.UrgentOrderInsert,
				Severity:  6,
			},
			expectedNeed:   true,
			expectedReason: "Urgent order",
		},
		{
			name: "Long recovery time",
			event: &models.DisruptionEvent{
				EventType:             models.EquipmentFailure,
				Severity:              6,
				EstimatedRecoveryTime: durationPtr(3 * time.Hour),
			},
			expectedNeed:   true,
			expectedReason: "Long recovery time",
		},
		{
			name: "Minor event - no replan",
			event: &models.DisruptionEvent{
				EventType:    models.ToolWearExcessive,
				Severity:     3,
				ImpactedJobs: []string{"J1"},
			},
			expectedNeed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			need, reason := dr.EvaluateReplanNeed(tt.event)
			assert.Equal(t, tt.expectedNeed, need)
			if tt.expectedNeed && tt.expectedReason != "" {
				assert.Contains(t, reason, tt.expectedReason)
			}
		})
	}
}

// 新增空间风险触发测试
func TestDynamicReplanner_EvaluateReplanNeed_SpatialRisk(t *testing.T) {
	replanner := NewDynamicReplanner()
	wm := worldmodel.NewWorldModel(nil, 5*time.Minute)
	replanner.SetWorldModel(wm)

	wm.UpdateObject(worldmodel.SensorObservation{
		ObjectID: "A",
		Pose:     worldmodel.Pose{X: 0, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now().Add(-1 * time.Second),
	})
	wm.UpdateObject(worldmodel.SensorObservation{
		ObjectID: "A",
		Pose:     worldmodel.Pose{X: 1, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})
	wm.UpdateObject(worldmodel.SensorObservation{
		ObjectID: "B",
		Pose:     worldmodel.Pose{X: 2, Y: 0, Z: 0, Qw: 1},
		Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		ObservedAt: time.Now(),
	})

	event := &models.DisruptionEvent{
		EventID:   "SPATIAL_1",
		EventType: models.AgentOverloaded,
		Severity:  3,
		DetectedAt: time.Now(),
		Metadata: map[string]interface{}{
			"object_ids":            []string{"A", "B"},
			"collision_horizon_sec": 2.0,
			"collision_threshold_m": 1.5,
		},
	}

	need, reason := replanner.EvaluateReplanNeed(event)
	assert.True(t, need)
	assert.Contains(t, reason, "Collision risk")
}
func TestDynamicReplanner_SelectStrategy(t *testing.T) {
	dr := NewDynamicReplanner()

	tests := []struct {
		name             string
		event            *models.DisruptionEvent
		affectedJobs     []string
		expectedStrategy ReplanStrategy
	}{
		{
			name: "Few jobs - incremental",
			event: &models.DisruptionEvent{
				EventType: models.MaterialShortage,
				Severity:  5,
			},
			affectedJobs:     []string{"J1", "J2"},
			expectedStrategy: IncrementalReplan,
		},
		{
			name: "High severity - global",
			event: &models.DisruptionEvent{
				EventType: models.EquipmentFailure,
				Severity:  9,
			},
			affectedJobs:     []string{"J1", "J2", "J3", "J4"},
			expectedStrategy: GlobalReplan,
		},
		{
			name: "Equipment failure with recovery time - right shift",
			event: &models.DisruptionEvent{
				EventType:             models.EquipmentFailure,
				Severity:              7,
				EstimatedRecoveryTime: durationPtr(2 * time.Hour),
			},
			affectedJobs:     []string{"J1", "J2", "J3", "J4"},
			expectedStrategy: RightShiftReplan,
		},
		{
			name: "Urgent order - global",
			event: &models.DisruptionEvent{
				EventType: models.UrgentOrderInsert,
				Severity:  7,
			},
			affectedJobs:     []string{"J1", "J2"},
			expectedStrategy: GlobalReplan,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := dr.selectReplanStrategy(tt.event, tt.affectedJobs)
			assert.Equal(t, tt.expectedStrategy, strategy)
		})
	}
}

func TestDynamicReplanner_RightShiftReplan(t *testing.T) {
	dr := NewDynamicReplanner()

	// 创建初始调度
	now := time.Now()
	initialSchedule := &models.ProductionSchedule{
		ScheduleID: "SCHEDULE_001",
		CreatedAt:  now,
		Assignments: []*models.JobAssignment{
			{
				AssignmentID: "A1",
				JobID:        "J1",
				AgentID:      "AGENT1",
				StartTime:    now.Add(1 * time.Hour),
				EndTime:      now.Add(2 * time.Hour),
			},
			{
				AssignmentID: "A2",
				JobID:        "J2",
				AgentID:      "AGENT1",
				StartTime:    now.Add(2 * time.Hour),
				EndTime:      now.Add(3 * time.Hour),
			},
		},
	}

	dr.SetCurrentSchedule(initialSchedule)

	// 创建扰动事件（设备故障，恢复需要30分钟）
	event := &models.DisruptionEvent{
		EventType:             models.EquipmentFailure,
		Severity:              7,
		AffectedAsset:         "AGENT1",
		EstimatedRecoveryTime: durationPtr(30 * time.Minute),
		ImpactedJobs:          []string{"J1", "J2"},
	}

	// 执行右移重规划
	newSchedule, err := dr.rightShiftReplan(event.ImpactedJobs, event)

	assert.NoError(t, err)
	assert.NotNil(t, newSchedule)

	// 验证任务被延后30分钟
	j1Assignment := newSchedule.FindAssignment("J1")
	assert.NotNil(t, j1Assignment)
	assert.Equal(t, now.Add(1*time.Hour+30*time.Minute), j1Assignment.StartTime)

	j2Assignment := newSchedule.FindAssignment("J2")
	assert.NotNil(t, j2Assignment)
	assert.Equal(t, now.Add(2*time.Hour+30*time.Minute), j2Assignment.StartTime)
}

func TestDynamicReplanner_ExecuteReplan(t *testing.T) {
	dr := NewDynamicReplanner()

	// 创建初始调度
	now := time.Now()
	initialSchedule := &models.ProductionSchedule{
		ScheduleID: "SCHEDULE_001",
		Assignments: []*models.JobAssignment{
			{JobID: "J1", AgentID: "AGENT1", StartTime: now, EndTime: now.Add(1 * time.Hour)},
		},
	}

	dr.SetCurrentSchedule(initialSchedule)

	// 创建扰动事件
	event := &models.DisruptionEvent{
		EventType:             models.EquipmentFailure,
		Severity:              8,
		AffectedAsset:         "AGENT1",
		EstimatedRecoveryTime: durationPtr(1 * time.Hour),
		ImpactedJobs:          []string{"J1"},
	}

	// 执行重规划
	ctx := context.Background()
	newSchedule, err := dr.ExecuteReplan(ctx, event)

	assert.NoError(t, err)
	assert.NotNil(t, newSchedule)

	// 验证重规划历史记录
	history := dr.GetReplanHistory()
	assert.Len(t, history, 1)
	assert.Equal(t, event, history[0].TriggerEvent)
	assert.NotNil(t, history[0].OldSchedule)
	assert.NotNil(t, history[0].NewSchedule)
}

func TestDynamicReplanner_CalculateImpact(t *testing.T) {
	dr := NewDynamicReplanner()

	now := time.Now()
	oldSchedule := &models.ProductionSchedule{
		Assignments: []*models.JobAssignment{
			{JobID: "J1", StartTime: now, EndTime: now.Add(1 * time.Hour)},
			{JobID: "J2", StartTime: now.Add(1 * time.Hour), EndTime: now.Add(2 * time.Hour)},
		},
	}

	newSchedule := &models.ProductionSchedule{
		Assignments: []*models.JobAssignment{
			{JobID: "J1", StartTime: now.Add(30 * time.Minute), EndTime: now.Add(1*time.Hour + 30*time.Minute)},
			{JobID: "J2", StartTime: now.Add(1*time.Hour + 30*time.Minute), EndTime: now.Add(2*time.Hour + 30*time.Minute)},
		},
	}

	impact := dr.calculateImpact(oldSchedule, newSchedule)

	assert.Equal(t, 2, impact.JobsRescheduled)
	assert.Equal(t, 30*time.Minute, impact.MakespanChange)
}

// 辅助函数
func durationPtr(d time.Duration) *time.Duration {
	return &d
}
