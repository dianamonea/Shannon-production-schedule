package cnp

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestCNPOrchestrator_ScoreBid_OverrideWeights(t *testing.T) {
	logger := zap.NewNop()
	orchestrator := NewCNPOrchestrator(logger)

	task := &TaskDescription{
		TaskID:        "TASK_1",
		TaskType:      "MILLING",
		EstimatedTime: 10 * time.Minute,
		Deadline:      time.Now().Add(1 * time.Hour),
		ExtraParameters: map[string]interface{}{
			"weight_duration": 0.1,
			"weight_cost":     0.6,
			"weight_quality":  0.1,
			"weight_load":     0.1,
			"weight_tool_health": 0.1,
		},
	}

	bid := &BidProposal{
		BidID:             "BID_1",
		TaskID:            task.TaskID,
		AgentID:           "AGENT_1",
		EstimatedDuration: 12 * time.Minute,
		EstimatedCost:     200.0,
		CurrentLoad:       0.2,
		EstimatedQuality:  0.9,
		ToolConditions:    map[string]float64{"T1": 0.9},
	}

	score := orchestrator.scoreBid(task, bid)
	assert.NotNil(t, score)
	assert.Greater(t, score.Score, 0.0)
	assert.Contains(t, score.Components, "cost")
}

func TestCNPOrchestrator_UpdateEvaluationCriteria(t *testing.T) {
	logger := zap.NewNop()
	orchestrator := NewCNPOrchestrator(logger)

	criteria := BidEvaluationCriteria{
		WeightDuration:     0.1,
		WeightCost:         0.5,
		WeightQuality:      0.2,
		WeightLoad:         0.1,
		WeightToolHealth:   0.1,
		LocalDisruptionPenalty: 0.05,
	}

	orchestrator.UpdateEvaluationCriteria(criteria)

	// Basic smoke: score with updated criteria should still be valid
	task := &TaskDescription{TaskID: "T1", EstimatedTime: 10 * time.Minute}
	bid := &BidProposal{BidID: "B1", TaskID: "T1", AgentID: "A1", EstimatedDuration: 10 * time.Minute, EstimatedCost: 100}
	score := orchestrator.scoreBid(task, bid)
	assert.NotNil(t, score)
}
