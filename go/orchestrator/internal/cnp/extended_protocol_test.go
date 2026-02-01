package cnp

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestMessageRouter_SendReceive(t *testing.T) {
	router := NewMessageRouter()

	// Register agents
	router.RegisterAgent("AGENT_001", 10)
	router.RegisterAgent("AGENT_002", 10)

	// Send message
	msg := &Message{
		MessageType:    MsgCallForProposal,
		SenderID:       "AGENT_001",
		ReceiverID:     "AGENT_002",
		ConversationID: "CONV_001",
		Priority:       5,
		Content: map[string]interface{}{
			"task": "MILLING",
			"cost": 100.0,
		},
	}

	err := router.SendMessage(msg)
	assert.NoError(t, err)

	// Receive message
	received, err := router.ReceiveMessage("AGENT_002", 1*time.Second)
	assert.NoError(t, err)
	assert.Equal(t, MsgCallForProposal, received.MessageType)
	assert.Equal(t, "AGENT_001", received.SenderID)
	assert.Equal(t, "CONV_001", received.ConversationID)
}

func TestMessageRouter_Broadcast(t *testing.T) {
	router := NewMessageRouter()

	// Register multiple agents
	agentIDs := []string{"AGENT_001", "AGENT_002", "AGENT_003"}
	for _, id := range agentIDs {
		router.RegisterAgent(id, 10)
	}

	// Broadcast message
	msg := &Message{
		MessageType:    MsgStatusUpdate,
		SenderID:       "ORCHESTRATOR",
		ConversationID: "BROADCAST_001",
		Priority:       3,
		Content: map[string]interface{}{
			"status": "MAINTENANCE_SCHEDULED",
		},
	}

	err := router.Broadcast(msg, agentIDs)
	assert.NoError(t, err)

	// Each agent should receive a message
	for _, id := range agentIDs {
		received, err := router.ReceiveMessage(id, 1*time.Second)
		assert.NoError(t, err)
		assert.Equal(t, MsgStatusUpdate, received.MessageType)
		assert.Equal(t, id, received.ReceiverID)
	}
}

func TestMessageRouter_ConversationHistory(t *testing.T) {
	router := NewMessageRouter()

	router.RegisterAgent("AGENT_001", 10)
	router.RegisterAgent("AGENT_002", 10)

	conversationID := "CONV_TEST"

	// Send multiple messages in conversation
	for i := 0; i < 3; i++ {
		msg := &Message{
			MessageType:    MsgProposal,
			SenderID:       "AGENT_001",
			ReceiverID:     "AGENT_002",
			ConversationID: conversationID,
			Priority:       5,
			Content: map[string]interface{}{
				"round": i + 1,
			},
		}
		router.SendMessage(msg)
	}

	// Get conversation history
	history := router.GetConversationHistory(conversationID)
	assert.Len(t, history, 3)
	assert.Equal(t, conversationID, history[0].ConversationID)
}

func TestMessageRouter_KnowledgeSharing(t *testing.T) {
	router := NewMessageRouter()

	// Share knowledge
	item1 := &KnowledgeItem{
		Category:   "BEST_PRACTICE",
		Content: map[string]interface{}{
			"tip": "Use coolant for better surface finish",
		},
		Confidence: 0.95,
		Source:     "AGENT_001",
	}

	item2 := &KnowledgeItem{
		Category:   "DEFECT_PATTERN",
		Content: map[string]interface{}{
			"pattern": "Chatter marks appear at high RPM",
		},
		Confidence: 0.80,
		Source:     "AGENT_002",
	}

	router.ShareKnowledge(item1)
	router.ShareKnowledge(item2)

	// Query knowledge
	bestPractices := router.QueryKnowledge("BEST_PRACTICE", 0.9)
	assert.Len(t, bestPractices, 1)
	assert.Equal(t, "AGENT_001", bestPractices[0].Source)

	defectPatterns := router.QueryKnowledge("DEFECT_PATTERN", 0.7)
	assert.Len(t, defectPatterns, 1)
	assert.Equal(t, 0.80, defectPatterns[0].Confidence)
}

func TestNegotiationManager_FullNegotiation(t *testing.T) {
	router := NewMessageRouter()
	router.RegisterAgent("INITIATOR", 10)
	router.RegisterAgent("RESPONDER", 10)

	nm := NewNegotiationManager(router)

	// Start negotiation
	initialOffer := &NegotiationOffer{
		Cost:     1000.0,
		Duration: 30 * time.Minute,
		Quality:  0.95,
		Justification: "Standard pricing",
	}

	session, err := nm.StartNegotiation("INITIATOR", "RESPONDER", "TASK_001", initialOffer)
	assert.NoError(t, err)
	assert.Equal(t, "ACTIVE", session.Status)
	assert.Equal(t, 1, session.CurrentRound)

	// Responder receives negotiation message
	msg, err := router.ReceiveMessage("RESPONDER", 1*time.Second)
	assert.NoError(t, err)
	assert.Equal(t, MsgNegotiate, msg.MessageType)

	// Responder makes counter-offer
	counterOffer := &NegotiationOffer{
		Cost:     900.0,
		Duration: 25 * time.Minute,
		Quality:  0.98,
		Justification: "Higher quality, lower cost",
	}

	err = nm.RespondToNegotiation(session.SessionID, counterOffer, false)
	assert.NoError(t, err)

	// Check session updated
	updatedSession, _ := nm.GetSession(session.SessionID)
	assert.Equal(t, 2, updatedSession.CurrentRound)
	assert.Len(t, updatedSession.Rounds, 2)

	// Accept final offer
	err = nm.RespondToNegotiation(session.SessionID, counterOffer, true)
	assert.NoError(t, err)

	finalSession, _ := nm.GetSession(session.SessionID)
	assert.Equal(t, "ACCEPTED", finalSession.Status)
	assert.NotNil(t, finalSession.FinalOffer)
	assert.Equal(t, 900.0, finalSession.FinalOffer.Cost)
}

func TestNegotiationManager_MaxRounds(t *testing.T) {
	router := NewMessageRouter()
	router.RegisterAgent("INITIATOR", 10)
	router.RegisterAgent("RESPONDER", 10)

	nm := NewNegotiationManager(router)

	initialOffer := &NegotiationOffer{
		Cost:     1000.0,
		Duration: 30 * time.Minute,
		Quality:  0.95,
	}

	session, err := nm.StartNegotiation("INITIATOR", "RESPONDER", "TASK_002", initialOffer)
	assert.NoError(t, err)

	// Exceed max rounds
	counterOffer := &NegotiationOffer{Cost: 950.0, Duration: 28 * time.Minute, Quality: 0.96}

	for i := 0; i < session.MaxRounds; i++ {
		err = nm.RespondToNegotiation(session.SessionID, counterOffer, false)
	}

	// Should fail on last iteration
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "max rounds")

	finalSession, _ := nm.GetSession(session.SessionID)
	assert.Equal(t, "REJECTED", finalSession.Status)
}

func TestCoordinationManager_RequestCoordination(t *testing.T) {
	router := NewMessageRouter()
	router.RegisterAgent("AGENT_001", 10)
	router.RegisterAgent("AGENT_002", 10)
	router.RegisterAgent("AGENT_003", 10)

	cm := NewCoordinationManager(router)

	req := &CoordinationRequest{
		TaskID:          "COORD_TASK_001",
		ParticipantIDs:  []string{"AGENT_001", "AGENT_002", "AGENT_003"},
		CoordinationType: "HANDOFF",
		Constraints: map[string]string{
			"sync_tolerance": "5s",
		},
		Deadline: time.Now().Add(1 * time.Hour),
	}

	err := cm.RequestCoordination(nil, req)
	assert.NoError(t, err)

	// Each participant should receive coordination request
	for _, agentID := range req.ParticipantIDs {
		msg, err := router.ReceiveMessage(agentID, 1*time.Second)
		assert.NoError(t, err)
		assert.Equal(t, MsgCoordinationRequest, msg.MessageType)
		assert.Equal(t, "COORD_TASK_001", msg.Content["task_id"])
		assert.Equal(t, "HANDOFF", msg.Content["coordination_type"])
	}
}

func TestCoordinationManager_AcknowledgeCoordination(t *testing.T) {
	router := NewMessageRouter()
	router.RegisterAgent("ORCHESTRATOR", 10)
	router.RegisterAgent("AGENT_001", 10)

	cm := NewCoordinationManager(router)

	err := cm.AcknowledgeCoordination("TASK_001", "AGENT_001", true)
	assert.NoError(t, err)

	// Orchestrator receives acknowledgment
	msg, err := router.ReceiveMessage("ORCHESTRATOR", 1*time.Second)
	assert.NoError(t, err)
	assert.Equal(t, MsgCoordinationAck, msg.MessageType)
	assert.Equal(t, true, msg.Content["ready"])
}

func TestMessageRouter_Heartbeat(t *testing.T) {
	router := NewMessageRouter()
	router.RegisterAgent("AGENT_001", 10)
	router.RegisterAgent("AGENT_002", 10)

	// Simulate stale agent
	router.heartbeats["AGENT_002"] = time.Now().Add(-10 * time.Minute)

	stale := router.GetStaleAgents(5 * time.Minute)
	assert.Len(t, stale, 1)
	assert.Equal(t, "AGENT_002", stale[0])
}
