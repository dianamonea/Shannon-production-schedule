package cnp

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/worldmodel"
)

func TestCNPOrchestrator_AwardTask_SpatialBlocked(t *testing.T) {
	logger := zap.NewNop()
	orchestrator := NewCNPOrchestrator(logger)

	grid := worldmodel.NewOccupancyGrid(5, 5, 1.0, 0.0, 0.0)
	err := grid.UpdateCell(1.0, 1.0, 1.0)
	assert.NoError(t, err)

	wm := worldmodel.NewWorldModel(grid, 5*time.Minute)
	orchestrator.SetWorldModel(wm)

	task := TaskDescription{
		TaskID:        "TASK_BLOCKED",
		TaskType:      "MILLING",
		EstimatedTime: 10 * time.Minute,
		Deadline:      time.Now().Add(1 * time.Hour),
		ExtraParameters: map[string]interface{}{
			"target_pose": map[string]interface{}{
				"x": 1.0,
				"y": 1.0,
				"z": 0.0,
			},
		},
	}

	err = orchestrator.AdvertiseTask(context.Background(), task)
	assert.NoError(t, err)

	score := &ScoringResult{
		BidID:   "BID_1",
		AgentID: "AGENT_1",
		Score:   0.9,
	}

	err = orchestrator.AwardTask(context.Background(), task.TaskID, score)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "target pose occupied")
}

func TestCNPOrchestrator_AwardTask_PlacementStability(t *testing.T) {
	logger := zap.NewNop()
	orchestrator := NewCNPOrchestrator(logger)

	grid := worldmodel.NewOccupancyGrid(5, 5, 1.0, 0.0, 0.0)
	wm := worldmodel.NewWorldModel(grid, 5*time.Minute)
	wm.AddOrUpdateSurface(&worldmodel.SurfacePatch{
		SurfaceID: "TABLE_1",
		Center:    worldmodel.Pose{X: 1.0, Y: 1.0, Z: 0.75, Qw: 1},
		Normal:    [3]float64{0, 0, 1},
		Area:      1.0,
		Friction:  0.6,
		UpdatedAt: time.Now(),
	})
	orchestrator.SetWorldModel(wm)

	task := TaskDescription{
		TaskID:        "TASK_STABLE",
		TaskType:      "ASSEMBLY",
		EstimatedTime: 10 * time.Minute,
		Deadline:      time.Now().Add(1 * time.Hour),
		ExtraParameters: map[string]interface{}{
			"target_pose": map[string]interface{}{
				"x": 1.0,
				"y": 1.0,
				"z": 0.75,
			},
			"placement_max_tilt_deg": 15.0,
		},
	}

	err := orchestrator.AdvertiseTask(context.Background(), task)
	assert.NoError(t, err)

	score := &ScoringResult{
		BidID:   "BID_1",
		AgentID: "AGENT_1",
		Score:   0.9,
	}

	err = orchestrator.AwardTask(context.Background(), task.TaskID, score)
	assert.NoError(t, err)
}
