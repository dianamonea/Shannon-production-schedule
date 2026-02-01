package cnp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// MessageType defines types of messages in extended protocol
type MessageType string

const (
	// FIPA Contract Net Protocol (existing)
	MsgCallForProposal MessageType = "CALL_FOR_PROPOSAL"
	MsgProposal        MessageType = "PROPOSAL"
	MsgAcceptProposal  MessageType = "ACCEPT_PROPOSAL"
	MsgRejectProposal  MessageType = "REJECT_PROPOSAL"
	MsgConfirm         MessageType = "CONFIRM"
	
	// Extended: Negotiation
	MsgCounterProposal MessageType = "COUNTER_PROPOSAL"
	MsgNegotiate       MessageType = "NEGOTIATE"
	
	// Extended: Coordination
	MsgCoordinationRequest MessageType = "COORDINATION_REQUEST"
	MsgCoordinationAck     MessageType = "COORDINATION_ACK"
	MsgHandoffRequest      MessageType = "HANDOFF_REQUEST"
	MsgHandoffComplete     MessageType = "HANDOFF_COMPLETE"
	
	// Extended: Monitoring
	MsgStatusUpdate    MessageType = "STATUS_UPDATE"
	MsgHeartbeat       MessageType = "HEARTBEAT"
	MsgAlert           MessageType = "ALERT"
	
	// Extended: Knowledge Sharing
	MsgKnowledgeShare  MessageType = "KNOWLEDGE_SHARE"
	MsgQueryKnowledge  MessageType = "QUERY_KNOWLEDGE"
	MsgKnowledgeResponse MessageType = "KNOWLEDGE_RESPONSE"
)

// Message represents a communication message between agents
type Message struct {
	MessageID   string                 `json:"message_id"`
	MessageType MessageType            `json:"message_type"`
	SenderID    string                 `json:"sender_id"`
	ReceiverID  string                 `json:"receiver_id"`
	Timestamp   time.Time              `json:"timestamp"`
	ConversationID string              `json:"conversation_id"`
	InReplyTo   string                 `json:"in_reply_to,omitempty"`
	Content     map[string]interface{} `json:"content"`
	Priority    int                    `json:"priority"` // 1-10
	TTL         time.Duration          `json:"ttl"`      // Time to live
}

// NegotiationOffer represents a negotiation counter-offer
type NegotiationOffer struct {
	Cost         float64           `json:"cost"`
	Duration     time.Duration     `json:"duration"`
	Quality      float64           `json:"quality"` // 0.0-1.0
	Constraints  map[string]string `json:"constraints"`
	Justification string           `json:"justification"`
}

// CoordinationRequest represents a request for multi-agent coordination
type CoordinationRequest struct {
	TaskID          string            `json:"task_id"`
	ParticipantIDs  []string          `json:"participant_ids"`
	CoordinationType string           `json:"coordination_type"` // HANDOFF, SYNCHRONIZED, SEQUENTIAL
	Constraints     map[string]string `json:"constraints"`
	Deadline        time.Time         `json:"deadline"`
}

// KnowledgeItem represents shared knowledge between agents
type KnowledgeItem struct {
	ItemID      string                 `json:"item_id"`
	Category    string                 `json:"category"` // BEST_PRACTICE, DEFECT_PATTERN, OPTIMIZATION_TIP
	Content     map[string]interface{} `json:"content"`
	Confidence  float64                `json:"confidence"` // 0.0-1.0
	Source      string                 `json:"source"`
	Timestamp   time.Time              `json:"timestamp"`
	UsageCount  int                    `json:"usage_count"`
}

// MessageRouter handles message routing and delivery
type MessageRouter struct {
	mu sync.RWMutex
	
	// Message queues per agent
	queues map[string]chan *Message
	
	// Conversation tracking
	conversations map[string][]*Message
	
	// Knowledge base
	knowledgeBase map[string]*KnowledgeItem
	
	// Message history
	messageHistory []*Message
	maxHistory     int

	// Heartbeat tracking
	heartbeats map[string]time.Time
}

// NewMessageRouter creates a new message router
func NewMessageRouter() *MessageRouter {
	return &MessageRouter{
		queues:         make(map[string]chan *Message),
		conversations:  make(map[string][]*Message),
		knowledgeBase:  make(map[string]*KnowledgeItem),
		messageHistory: make([]*Message, 0),
		maxHistory:     10000,
		heartbeats:     make(map[string]time.Time),
	}
}

// RegisterAgent registers an agent's message queue
func (mr *MessageRouter) RegisterAgent(agentID string, queueSize int) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	if _, exists := mr.queues[agentID]; exists {
		log.Printf("[ROUTER] Agent %s already registered", agentID)
		return
	}
	
	mr.queues[agentID] = make(chan *Message, queueSize)
	mr.heartbeats[agentID] = time.Now()
	log.Printf("[ROUTER] Registered agent %s with queue size %d", agentID, queueSize)
}

// RecordHeartbeat updates heartbeat timestamp for an agent
func (mr *MessageRouter) RecordHeartbeat(agentID string) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	mr.heartbeats[agentID] = time.Now()
}

// GetStaleAgents returns agents not seen within threshold
func (mr *MessageRouter) GetStaleAgents(threshold time.Duration) []string {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	stale := make([]string, 0)
	cutoff := time.Now().Add(-threshold)
	for agentID, last := range mr.heartbeats {
		if last.Before(cutoff) {
			stale = append(stale, agentID)
		}
	}
	return stale
}

// SendMessage sends a message to an agent
func (mr *MessageRouter) SendMessage(msg *Message) error {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	// Validate message
	if msg.SenderID == "" || msg.ReceiverID == "" {
		return fmt.Errorf("sender and receiver required")
	}
	
	// Check if receiver exists
	queue, exists := mr.queues[msg.ReceiverID]
	if !exists {
		return fmt.Errorf("receiver %s not registered", msg.ReceiverID)
	}
	
	// Set timestamp if not set
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now()
	}

	// Update heartbeat on heartbeat messages
	if msg.MessageType == MsgHeartbeat {
		mr.heartbeats[msg.SenderID] = msg.Timestamp
	}
	
	// Generate message ID if not set
	if msg.MessageID == "" {
		msg.MessageID = fmt.Sprintf("MSG_%d", time.Now().UnixNano())
	}
	
	// Add to conversation history
	if msg.ConversationID != "" {
		mr.conversations[msg.ConversationID] = append(
			mr.conversations[msg.ConversationID],
			msg,
		)
	}
	
	// Add to message history
	mr.messageHistory = append(mr.messageHistory, msg)
	if len(mr.messageHistory) > mr.maxHistory {
		mr.messageHistory = mr.messageHistory[len(mr.messageHistory)-mr.maxHistory:]
	}
	
	// Send to queue (non-blocking)
	select {
	case queue <- msg:
		log.Printf("[ROUTER] Sent %s from %s to %s", msg.MessageType, msg.SenderID, msg.ReceiverID)
		return nil
	default:
		return fmt.Errorf("receiver %s message queue full", msg.ReceiverID)
	}
}

// ReceiveMessage receives a message for an agent (blocking with timeout)
func (mr *MessageRouter) ReceiveMessage(agentID string, timeout time.Duration) (*Message, error) {
	mr.mu.RLock()
	queue, exists := mr.queues[agentID]
	mr.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("agent %s not registered", agentID)
	}
	
	deadline := time.Now().Add(timeout)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return nil, fmt.Errorf("timeout waiting for message")
		}
		select {
		case msg := <-queue:
			if msg.TTL > 0 && msg.Timestamp.Add(msg.TTL).Before(time.Now()) {
				// Drop expired message and continue
				continue
			}
			return msg, nil
		case <-time.After(remaining):
			return nil, fmt.Errorf("timeout waiting for message")
		}
	}
}

// Broadcast sends a message to multiple agents
func (mr *MessageRouter) Broadcast(msg *Message, receiverIDs []string) error {
	errors := make([]error, 0)
	
	for _, receiverID := range receiverIDs {
		msgCopy := *msg
		msgCopy.ReceiverID = receiverID
		msgCopy.MessageID = fmt.Sprintf("MSG_%s_%d", receiverID, time.Now().UnixNano())
		
		err := mr.SendMessage(&msgCopy)
		if err != nil {
			errors = append(errors, err)
		}
	}
	
	if len(errors) > 0 {
		return fmt.Errorf("broadcast failed for %d receivers", len(errors))
	}
	
	return nil
}

// GetConversationHistory retrieves message history for a conversation
func (mr *MessageRouter) GetConversationHistory(conversationID string) []*Message {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	
	if messages, exists := mr.conversations[conversationID]; exists {
		return messages
	}
	
	return []*Message{}
}

// ShareKnowledge adds knowledge to the shared knowledge base
func (mr *MessageRouter) ShareKnowledge(item *KnowledgeItem) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	
	if item.ItemID == "" {
		item.ItemID = fmt.Sprintf("KNOW_%d", time.Now().UnixNano())
	}
	
	if item.Timestamp.IsZero() {
		item.Timestamp = time.Now()
	}
	
	mr.knowledgeBase[item.ItemID] = item
	log.Printf("[ROUTER] Shared knowledge: %s (%s) with confidence %.2f",
		item.ItemID, item.Category, item.Confidence)
}

// QueryKnowledge retrieves knowledge items matching category
func (mr *MessageRouter) QueryKnowledge(category string, minConfidence float64) []*KnowledgeItem {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	
	results := make([]*KnowledgeItem, 0)
	
	for _, item := range mr.knowledgeBase {
		if item.Category == category && item.Confidence >= minConfidence {
			results = append(results, item)
		}
	}
	
	return results
}

// NegotiationSession manages multi-round negotiation between agents
type NegotiationSession struct {
	SessionID      string
	InitiatorID    string
	ResponderID    string
	TaskID         string
	
	Rounds         []*NegotiationRound
	MaxRounds      int
	CurrentRound   int
	
	Status         string // ACTIVE, ACCEPTED, REJECTED, TIMEOUT
	FinalOffer     *NegotiationOffer
	
	StartTime      time.Time
	EndTime        *time.Time
}

// NegotiationRound represents one round of negotiation
type NegotiationRound struct {
	RoundNumber    int
	InitiatorOffer *NegotiationOffer
	ResponderOffer *NegotiationOffer
	Timestamp      time.Time
}

// NegotiationManager manages negotiation sessions
type NegotiationManager struct {
	mu sync.RWMutex
	
	sessions map[string]*NegotiationSession
	router   *MessageRouter
}

// NewNegotiationManager creates a new negotiation manager
func NewNegotiationManager(router *MessageRouter) *NegotiationManager {
	return &NegotiationManager{
		sessions: make(map[string]*NegotiationSession),
		router:   router,
	}
}

// StartNegotiation initiates a negotiation session
func (nm *NegotiationManager) StartNegotiation(initiatorID, responderID, taskID string,
	initialOffer *NegotiationOffer) (*NegotiationSession, error) {
	
	nm.mu.Lock()
	defer nm.mu.Unlock()
	
	session := &NegotiationSession{
		SessionID:    fmt.Sprintf("NEG_%d", time.Now().UnixNano()),
		InitiatorID:  initiatorID,
		ResponderID:  responderID,
		TaskID:       taskID,
		Rounds:       make([]*NegotiationRound, 0),
		MaxRounds:    5,
		CurrentRound: 1,
		Status:       "ACTIVE",
		StartTime:    time.Now(),
	}
	
	// First round
	round := &NegotiationRound{
		RoundNumber:    1,
		InitiatorOffer: initialOffer,
		Timestamp:      time.Now(),
	}
	
	session.Rounds = append(session.Rounds, round)
	nm.sessions[session.SessionID] = session
	
	// Send negotiation message
	msg := &Message{
		MessageType:    MsgNegotiate,
		SenderID:       initiatorID,
		ReceiverID:     responderID,
		ConversationID: session.SessionID,
		Priority:       5,
		Content: map[string]interface{}{
			"session_id": session.SessionID,
			"task_id":    taskID,
			"offer":      initialOffer,
			"round":      1,
		},
	}
	
	err := nm.router.SendMessage(msg)
	if err != nil {
		return nil, err
	}
	
	log.Printf("[NEGOTIATION] Started session %s between %s and %s",
		session.SessionID, initiatorID, responderID)
	
	return session, nil
}

// RespondToNegotiation responds to a negotiation offer
func (nm *NegotiationManager) RespondToNegotiation(sessionID string, 
	counterOffer *NegotiationOffer, accept bool) error {
	
	nm.mu.Lock()
	defer nm.mu.Unlock()
	
	session, exists := nm.sessions[sessionID]
	if !exists {
		return fmt.Errorf("negotiation session %s not found", sessionID)
	}
	
	if session.Status != "ACTIVE" {
		return fmt.Errorf("session %s not active", sessionID)
	}
	
	currentRound := session.Rounds[len(session.Rounds)-1]
	currentRound.ResponderOffer = counterOffer
	
	if accept {
		// Accept offer
		session.Status = "ACCEPTED"
		session.FinalOffer = counterOffer
		now := time.Now()
		session.EndTime = &now
		
		msg := &Message{
			MessageType:    MsgAcceptProposal,
			SenderID:       session.ResponderID,
			ReceiverID:     session.InitiatorID,
			ConversationID: sessionID,
			Priority:       8,
			Content: map[string]interface{}{
				"session_id": sessionID,
				"final_offer": counterOffer,
			},
		}
		
		return nm.router.SendMessage(msg)
	}
	
	// Counter-offer
	session.CurrentRound++
	
	if session.CurrentRound > session.MaxRounds {
		// Max rounds reached
		session.Status = "REJECTED"
		now := time.Now()
		session.EndTime = &now
		
		return fmt.Errorf("negotiation failed: max rounds reached")
	}
	
	// Create new round
	newRound := &NegotiationRound{
		RoundNumber:    session.CurrentRound,
		ResponderOffer: counterOffer,
		Timestamp:      time.Now(),
	}
	
	session.Rounds = append(session.Rounds, newRound)
	
	msg := &Message{
		MessageType:    MsgCounterProposal,
		SenderID:       session.ResponderID,
		ReceiverID:     session.InitiatorID,
		ConversationID: sessionID,
		Priority:       7,
		Content: map[string]interface{}{
			"session_id": sessionID,
			"offer":      counterOffer,
			"round":      session.CurrentRound,
		},
	}
	
	return nm.router.SendMessage(msg)
}

// GetSession retrieves a negotiation session
func (nm *NegotiationManager) GetSession(sessionID string) (*NegotiationSession, error) {
	nm.mu.RLock()
	defer nm.mu.RUnlock()
	
	session, exists := nm.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session %s not found", sessionID)
	}
	
	return session, nil
}

// CoordinationManager manages multi-agent coordination
type CoordinationManager struct {
	mu sync.RWMutex
	
	activeCoordinations map[string]*CoordinationRequest
	router              *MessageRouter
}

// NewCoordinationManager creates a new coordination manager
func NewCoordinationManager(router *MessageRouter) *CoordinationManager {
	return &CoordinationManager{
		activeCoordinations: make(map[string]*CoordinationRequest),
		router:              router,
	}
}

// RequestCoordination initiates a coordination request
func (cm *CoordinationManager) RequestCoordination(ctx context.Context, req *CoordinationRequest) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	cm.activeCoordinations[req.TaskID] = req
	
	// Broadcast coordination request to all participants
	msg := &Message{
		MessageType:    MsgCoordinationRequest,
		SenderID:       "ORCHESTRATOR",
		ConversationID: req.TaskID,
		Priority:       9,
		Content: map[string]interface{}{
			"task_id":           req.TaskID,
			"coordination_type": req.CoordinationType,
			"constraints":       req.Constraints,
			"deadline":          req.Deadline,
		},
	}
	
	err := cm.router.Broadcast(msg, req.ParticipantIDs)
	if err != nil {
		return err
	}
	
	log.Printf("[COORDINATION] Requested %s coordination for task %s with %d participants",
		req.CoordinationType, req.TaskID, len(req.ParticipantIDs))
	
	return nil
}

// AcknowledgeCoordination acknowledges participation in coordination
func (cm *CoordinationManager) AcknowledgeCoordination(taskID, agentID string, ready bool) error {
	msg := &Message{
		MessageType:    MsgCoordinationAck,
		SenderID:       agentID,
		ReceiverID:     "ORCHESTRATOR",
		ConversationID: taskID,
		Priority:       8,
		Content: map[string]interface{}{
			"task_id": taskID,
			"ready":   ready,
		},
	}
	
	return cm.router.SendMessage(msg)
}

// MarshalJSON custom JSON marshaling for Message
func (m *Message) MarshalJSON() ([]byte, error) {
	type Alias Message
	return json.Marshal(&struct {
		Timestamp string `json:"timestamp"`
		TTL       string `json:"ttl"`
		*Alias
	}{
		Timestamp: m.Timestamp.Format(time.RFC3339),
		TTL:       m.TTL.String(),
		Alias:     (*Alias)(m),
	})
}
