package cnp

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/worldmodel"
)

func TestCNPOrchestrator_UpdateWorldModelFromSemanticJSON(t *testing.T) {
	logger := zap.NewNop()
	orchestrator := NewCNPOrchestrator(logger)

	wm := worldmodel.NewWorldModel(nil, 5*time.Minute)
	orchestrator.SetWorldModel(wm)

	payload := []byte(`[
		{"object_id":"OBJ_1","pose":{"position":[0.1,0.2,0.3],"orientation":[1.0,0.0,0.0,0.0]}}
	]`)

	count := orchestrator.UpdateWorldModelFromSemanticJSON(payload)
	assert.Equal(t, 1, count)
	_, ok := wm.GetObject("OBJ_1")
	assert.True(t, ok)
}

func TestCNPOrchestrator_UpdateWorldModelFromSemanticObjectsJSON(t *testing.T) {
	logger := zap.NewNop()
	orchestrator := NewCNPOrchestrator(logger)

	wm := worldmodel.NewWorldModel(worldmodel.NewOccupancyGrid(5, 5, 1.0, 0.0, 0.0), 5*time.Minute)
	orchestrator.SetWorldModel(wm)

	payload := []byte(`[
		{"object_id":"TABLE_A","label":"table","pose":{"position":[1.0,1.0,0.7],"orientation":[1.0,0.0,0.0,0.0]}}
	]`)

	count := orchestrator.UpdateWorldModelFromSemanticObjectsJSON(payload)
	assert.Equal(t, 1, count)
	assert.True(t, wm.HasSurfaces())
}
