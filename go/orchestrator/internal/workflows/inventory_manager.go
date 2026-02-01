package workflows

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/Kocoro-lab/Shannon/go/orchestrator/internal/models"
)

type InventoryManager struct {
	mu               sync.RWMutex
	inventory        map[string]*models.InventoryItem  // MaterialID -> Item
	wipBuffers       map[string]*models.WIPBuffer      // BufferID -> Buffer
	kanbanCards      map[string]*models.KanbanCard     // KanbanID -> Card
	eventLog         []models.MaterialFlowEvent        // 物料流事件日志
	replenishChannel chan ReplenishRequest
	ctx              context.Context
	cancel           context.CancelFunc
}

type ReplenishRequest struct {
	MaterialID string
	Quantity   int
	Urgency    int // 1-10
}

func NewInventoryManager() *InventoryManager {
	ctx, cancel := context.WithCancel(context.Background())
	im := &InventoryManager{
		inventory:        make(map[string]*models.InventoryItem),
		wipBuffers:       make(map[string]*models.WIPBuffer),
		kanbanCards:      make(map[string]*models.KanbanCard),
		eventLog:         make([]models.MaterialFlowEvent, 0, 1000),
		replenishChannel: make(chan ReplenishRequest, 100),
		ctx:              ctx,
		cancel:           cancel,
	}

	// 启动补货处理协程
	go im.processReplenishRequests()

	return im
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
		select {
		case im.replenishChannel <- ReplenishRequest{
			MaterialID: materialID,
			Quantity:   item.ReorderPoint - item.Quantity + item.SafetyStock,
			Urgency:    10, // 紧急补货
		}:
		default:
			// Channel满了，记录警告
			fmt.Printf("WARNING: Replenish channel full for material %s\n", materialID)
		}
	}

	// 记录物料流事件
	event := models.MaterialFlowEvent{
		EventID:      im.generateEventID(),
		EventType:    models.MaterialIssued,
		MaterialID:   materialID,
		Quantity:     qty,
		FromLocation: item.Location,
		ToLocation:   "WORKSTATION",
		Timestamp:    time.Now(),
		RelatedJobID: jobID,
	}
	im.eventLog = append(im.eventLog, event)

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
		now := time.Now()
		buffer.BlockedSince = &now
		return fmt.Errorf("buffer %s is full (capacity %d)", bufferID, buffer.Capacity)
	}

	buffer.Jobs = append(buffer.Jobs, jobID)
	buffer.CurrentLevel++
	buffer.BlockedSince = nil

	return nil
}

// RemoveFromWIPBuffer 从缓冲区移除任务
func (im *InventoryManager) RemoveFromWIPBuffer(bufferID string, jobID string) error {
	im.mu.Lock()
	defer im.mu.Unlock()

	buffer, exists := im.wipBuffers[bufferID]
	if !exists {
		return fmt.Errorf("buffer %s not found", bufferID)
	}

	// 查找并移除JobID
	for i, id := range buffer.Jobs {
		if id == jobID {
			buffer.Jobs = append(buffer.Jobs[:i], buffer.Jobs[i+1:]...)
			buffer.CurrentLevel--
			buffer.BlockedSince = nil
			return nil
		}
	}

	return fmt.Errorf("job %s not found in buffer %s", jobID, bufferID)
}

// IssueKanban 发出Kanban卡片（下游消耗触发）
func (im *InventoryManager) IssueKanban(partNumber string, fromStation, toStation string, qty int) string {
	im.mu.Lock()
	defer im.mu.Unlock()

	kanban := &models.KanbanCard{
		KanbanID:          im.generateKanbanID(),
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
	fmt.Printf("Kanban issued: %s for %d units of %s from %s to %s\n",
		kanban.KanbanID, qty, partNumber, fromStation, toStation)

	return kanban.KanbanID
}

// UpdateKanbanStatus 更新Kanban状态
func (im *InventoryManager) UpdateKanbanStatus(kanbanID string, status models.KanbanStatus) error {
	im.mu.Lock()
	defer im.mu.Unlock()

	kanban, exists := im.kanbanCards[kanbanID]
	if !exists {
		return fmt.Errorf("kanban %s not found", kanbanID)
	}

	kanban.Status = status
	if status == models.KanbanConsumed {
		now := time.Now()
		kanban.ConsumedAt = &now
	}

	return nil
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

// AddInventoryItem 添加库存项目
func (im *InventoryManager) AddInventoryItem(item *models.InventoryItem) {
	im.mu.Lock()
	defer im.mu.Unlock()

	item.LastUpdated = time.Now()
	im.inventory[item.MaterialID] = item
}

// CreateWIPBuffer 创建在制品缓冲区
func (im *InventoryManager) CreateWIPBuffer(bufferID, workstationID string, capacity int) {
	im.mu.Lock()
	defer im.mu.Unlock()

	im.wipBuffers[bufferID] = &models.WIPBuffer{
		BufferID:      bufferID,
		WorkstationID: workstationID,
		Capacity:      capacity,
		CurrentLevel:  0,
		Jobs:          make([]string, 0, capacity),
		FIFO:          true,
		BlockedSince:  nil,
	}
}

// GetInventoryLevel 获取库存水平
func (im *InventoryManager) GetInventoryLevel(materialID string) (int, error) {
	im.mu.RLock()
	defer im.mu.RUnlock()

	item, exists := im.inventory[materialID]
	if !exists {
		return 0, fmt.Errorf("material %s not found", materialID)
	}

	return item.Quantity, nil
}

// GetEventLog 获取物料流事件日志
func (im *InventoryManager) GetEventLog(since time.Time) []models.MaterialFlowEvent {
	im.mu.RLock()
	defer im.mu.RUnlock()

	filtered := make([]models.MaterialFlowEvent, 0)
	for _, event := range im.eventLog {
		if event.Timestamp.After(since) {
			filtered = append(filtered, event)
		}
	}

	return filtered
}

// processReplenishRequests 处理补货请求
func (im *InventoryManager) processReplenishRequests() {
	for {
		select {
		case <-im.ctx.Done():
			return
		case req := <-im.replenishChannel:
			im.handleReplenish(req)
		}
	}
}

func (im *InventoryManager) handleReplenish(req ReplenishRequest) {
	im.mu.RLock()
	item, exists := im.inventory[req.MaterialID]
	im.mu.RUnlock()

	if !exists {
		fmt.Printf("ERROR: Cannot replenish unknown material %s\n", req.MaterialID)
		return
	}

	fmt.Printf("REPLENISH REQUEST: Material=%s, Qty=%d, Urgency=%d, LeadTime=%d days\n",
		req.MaterialID, req.Quantity, req.Urgency, item.LeadTimeDays)

	// TODO: 集成到采购系统/ERP
	// TODO: 创建采购订单
}

// Shutdown 关闭库存管理器
func (im *InventoryManager) Shutdown() {
	im.cancel()
	close(im.replenishChannel)
}

func (im *InventoryManager) generateEventID() string {
	return fmt.Sprintf("EVT-%d", time.Now().UnixNano())
}

func (im *InventoryManager) generateKanbanID() string {
	return fmt.Sprintf("KAN-%d", time.Now().UnixNano())
}
