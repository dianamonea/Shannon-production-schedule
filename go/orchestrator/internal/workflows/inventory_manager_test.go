package workflows

import (
	"fmt"
	"testing"
	"time"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/models"
	"github.com/stretchr/testify/assert"
)

func TestInventoryManager_IssueMaterial(t *testing.T) {
	im := NewInventoryManager()
	defer im.Shutdown()

	// 添加测试库存
	im.AddInventoryItem(&models.InventoryItem{
		MaterialID:   "MAT001",
		MaterialType: "Steel Plate",
		Quantity:     100,
		Unit:         "kg",
		Location:     "WAREHOUSE_A",
		SafetyStock:  20,
		ReorderPoint: 30,
		LeadTimeDays: 5,
		UnitCost:     50.0,
	})

	// 测试正常发料
	err := im.IssueMaterial("MAT001", 50, "JOB001")
	assert.NoError(t, err)

	level, _ := im.GetInventoryLevel("MAT001")
	assert.Equal(t, 50, level)

	// 测试库存不足
	err = im.IssueMaterial("MAT001", 60, "JOB002")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "insufficient stock")

	// 验证库存未被错误扣减
	level, _ = im.GetInventoryLevel("MAT001")
	assert.Equal(t, 50, level)
}

func TestInventoryManager_ReorderTrigger(t *testing.T) {
	im := NewInventoryManager()
	defer im.Shutdown()

	im.AddInventoryItem(&models.InventoryItem{
		MaterialID:   "MAT002",
		Quantity:     35,
		SafetyStock:  20,
		ReorderPoint: 30,
		LeadTimeDays: 3,
	})

	// 发料后触发再订购点
	err := im.IssueMaterial("MAT002", 10, "JOB003")
	assert.NoError(t, err)

	// 等待补货请求处理
	time.Sleep(100 * time.Millisecond)

	// 验证库存降至再订购点以下
	level, _ := im.GetInventoryLevel("MAT002")
	assert.Equal(t, 25, level)
	assert.Less(t, level, 30) // 小于再订购点
}

func TestInventoryManager_WIPBuffer(t *testing.T) {
	im := NewInventoryManager()
	defer im.Shutdown()

	im.CreateWIPBuffer("BUF001", "WORKSTATION_A", 5)

	// 添加3个任务
	for i := 1; i <= 3; i++ {
		err := im.AddWIPToBuffer("BUF001", fmt.Sprintf("JOB%03d", i))
		assert.NoError(t, err)
	}

	level, capacity, _ := im.GetBufferStatus("BUF001")
	assert.Equal(t, 3, level)
	assert.Equal(t, 5, capacity)

	// 测试缓冲区满
	for i := 4; i <= 5; i++ {
		im.AddWIPToBuffer("BUF001", fmt.Sprintf("JOB%03d", i))
	}

	err := im.AddWIPToBuffer("BUF001", "JOB006")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "buffer")
	assert.Contains(t, err.Error(), "full")

	// 测试移除任务
	err = im.RemoveFromWIPBuffer("BUF001", "JOB001")
	assert.NoError(t, err)

	level, _, _ = im.GetBufferStatus("BUF001")
	assert.Equal(t, 4, level)

	// 现在可以再添加一个
	err = im.AddWIPToBuffer("BUF001", "JOB006")
	assert.NoError(t, err)
}

func TestInventoryManager_Kanban(t *testing.T) {
	im := NewInventoryManager()
	defer im.Shutdown()

	// 发出Kanban卡片
	kanbanID := im.IssueKanban("PART123", "WORKSTATION_A", "WORKSTATION_B", 10)
	assert.NotEmpty(t, kanbanID)

	// 更新Kanban状态
	err := im.UpdateKanbanStatus(kanbanID, models.KanbanInTransit)
	assert.NoError(t, err)

	err = im.UpdateKanbanStatus(kanbanID, models.KanbanConsumed)
	assert.NoError(t, err)

	// 验证消耗时间已设置
	im.mu.RLock()
	kanban := im.kanbanCards[kanbanID]
	im.mu.RUnlock()

	assert.NotNil(t, kanban.ConsumedAt)
	assert.Equal(t, models.KanbanConsumed, kanban.Status)
}

func TestInventoryManager_EventLog(t *testing.T) {
	im := NewInventoryManager()
	defer im.Shutdown()

	im.AddInventoryItem(&models.InventoryItem{
		MaterialID:  "MAT003",
		Quantity:    100,
		SafetyStock: 10,
	})

	before := time.Now()
	time.Sleep(10 * time.Millisecond)

	// 执行发料
	im.IssueMaterial("MAT003", 20, "JOB100")

	time.Sleep(10 * time.Millisecond)

	// 获取事件日志
	events := im.GetEventLog(before)
	assert.Len(t, events, 1)
	assert.Equal(t, models.MaterialIssued, events[0].EventType)
	assert.Equal(t, "MAT003", events[0].MaterialID)
	assert.Equal(t, 20, events[0].Quantity)
	assert.Equal(t, "JOB100", events[0].RelatedJobID)
}

func TestInventoryManager_CheckAvailability(t *testing.T) {
	im := NewInventoryManager()
	defer im.Shutdown()

	im.AddInventoryItem(&models.InventoryItem{
		MaterialID:  "MAT004",
		Quantity:    50,
		SafetyStock: 15,
	})

	// 可用数量 = 50 - 15 = 35
	available, qty := im.CheckAvailability("MAT004", 30)
	assert.True(t, available)
	assert.Equal(t, 35, qty)

	// 超过可用数量
	available, qty = im.CheckAvailability("MAT004", 40)
	assert.False(t, available)
	assert.Equal(t, 35, qty)

	// 不存在的材料
	available, qty = im.CheckAvailability("MAT999", 10)
	assert.False(t, available)
	assert.Equal(t, 0, qty)
}
