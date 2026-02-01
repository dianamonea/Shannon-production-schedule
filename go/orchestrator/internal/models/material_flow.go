package models

import "time"

// InventoryItem 库存项目
type InventoryItem struct {
	MaterialID    string
	MaterialType  string
	Quantity      int
	Unit          string     // kg, pcs, m
	Location      string     // 仓库位置
	Batch         string     // 批次号
	ExpiryDate    *time.Time // 保质期（可选）
	SafetyStock   int        // 安全库存
	ReorderPoint  int        // 再订购点
	LeadTimeDays  int        // 供应商交货期
	UnitCost      float64
	LastUpdated   time.Time
}

// WIPBuffer 在制品缓冲区
type WIPBuffer struct {
	BufferID      string
	WorkstationID string     // 缓冲区所属工位
	Capacity      int        // 最大容量
	CurrentLevel  int        // 当前数量
	Jobs          []string   // 等待的JobID列表
	FIFO          bool       // 先进先出 vs 优先级队列
	BlockedSince  *time.Time // 阻塞时间（满载时）
}

// KanbanCard Kanban卡片
type KanbanCard struct {
	KanbanID          string
	PartNumber        string
	Quantity          int
	SourceWorkstation string       // 上游工位
	SinkWorkstation   string       // 下游工位
	Status            KanbanStatus // WAITING, IN_TRANSIT, CONSUMED
	IssuedAt          time.Time
	ConsumedAt        *time.Time
	Priority          int
}

type KanbanStatus string

const (
	KanbanWaiting   KanbanStatus = "WAITING"
	KanbanInTransit KanbanStatus = "IN_TRANSIT"
	KanbanConsumed  KanbanStatus = "CONSUMED"
)

// MaterialFlowEvent 物料流事件
type MaterialFlowEvent struct {
	EventID       string
	EventType     MaterialEventType
	MaterialID    string
	Quantity      int
	FromLocation  string
	ToLocation    string
	TransportedBy string // AGV ID
	Timestamp     time.Time
	RelatedJobID  string
}

type MaterialEventType string

const (
	MaterialArrival     MaterialEventType = "ARRIVAL"      // 入库
	MaterialIssued      MaterialEventType = "ISSUED"       // 发料
	MaterialConsumed    MaterialEventType = "CONSUMED"     // 消耗
	MaterialReplenished MaterialEventType = "REPLENISHED" // 补货
	MaterialDefective   MaterialEventType = "DEFECTIVE"   // 不良品
)

// MaterialRequirement 物料需求
type MaterialRequirement struct {
	MaterialID string
	Quantity   int
}
