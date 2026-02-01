package cnp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// P2PBidingHandler manages bidding round-trips between orchestrator and agents
type P2PBiddingHandler struct {
	logger       *zap.Logger
	cnpOrch      *CNPOrchestrator
	taskAds      chan TaskAdvertisement  // broadcast channel to agents
	bidResponses chan BidResponse        // channel from agents
	awardMsgs    chan AwardMessage       // channel to agents
	confirmMsgs  chan ConfirmationMessage // channel from agents

	ongoingBidRounds map[string]*BiddingRound
	mu               sync.RWMutex

	maxBidWaitTime   time.Duration
	maxConfirmTime   time.Duration
}

// TaskAdvertisement sent to all agents
type TaskAdvertisement struct {
	AdID         string
	Task         TaskDescription
	AdvertiseTime time.Time
	BidDeadline   time.Time
}

// BidResponse with optional constraints from agent
type BidResponse struct {
	Bid              BidProposal
	ConstraintChanges map[string]interface{} // local constraint updates
	ResponseTime      time.Time
}

// AwardMessage sent to winning agent
type AwardMessage struct {
	Award              AwardNotification
	MessageID          string
	ConfirmationDeadline time.Time
}

// ConfirmationMessage from agent after receiving award
type ConfirmationMessage struct {
	BidID               string
	TaskID              string
	AgentID             string
	Status              string  // CONFIRMED, REJECTED, FAILED
	ConfirmationTime    time.Time
	LocalScheduleUpdate map[string]interface{}
}

// BiddingRound tracks one complete bidding cycle
type BiddingRound struct {
	RoundID            string
	TaskID             string
	StartTime          time.Time
	BidDeadline        time.Time
	ConfirmationDeadline time.Time
	ReceivedBids       []BidProposal
	WinnerBid          *ScoringResult
	FinalStatus        string // ONGOING, COMPLETED, FAILED
	mu                 sync.RWMutex
}

// NewP2PBiddingHandler creates new handler
func NewP2PBiddingHandler(logger *zap.Logger, cnpOrch *CNPOrchestrator) *P2PBiddingHandler {
	return &P2PBiddingHandler{
		logger:           logger,
		cnpOrch:          cnpOrch,
		taskAds:          make(chan TaskAdvertisement, 100),
		bidResponses:     make(chan BidResponse, 1000),
		awardMsgs:        make(chan AwardMessage, 100),
		confirmMsgs:      make(chan ConfirmationMessage, 100),
		ongoingBidRounds: make(map[string]*BiddingRound),
		maxBidWaitTime:   3 * time.Second,
		maxConfirmTime:   2 * time.Second,
	}
}

// StartBiddingRound initiates a complete bidding cycle for a task
func (h *P2PBiddingHandler) StartBiddingRound(ctx context.Context, task TaskDescription) (*BiddingRound, error) {
	round := &BiddingRound{
		RoundID:            fmt.Sprintf("round_%s_%d", task.TaskID, time.Now().UnixNano()),
		TaskID:             task.TaskID,
		StartTime:          time.Now(),
		BidDeadline:        time.Now().Add(h.maxBidWaitTime),
		ConfirmationDeadline: time.Now().Add(h.maxBidWaitTime + h.maxConfirmTime),
		ReceivedBids:       make([]BidProposal, 0),
		FinalStatus:        "ONGOING",
	}

	h.mu.Lock()
	h.ongoingBidRounds[round.RoundID] = round
	h.mu.Unlock()

	// Broadcast task advertisement
	ad := TaskAdvertisement{
		AdID:         fmt.Sprintf("ad_%s_%d", task.TaskID, time.Now().UnixNano()),
		Task:         task,
		AdvertiseTime: time.Now(),
		BidDeadline:   round.BidDeadline,
	}

	h.logger.Info("Bidding round started",
		zap.String("round_id", round.RoundID),
		zap.String("task_id", task.TaskID),
		zap.Time("bid_deadline", round.BidDeadline),
	)

	// Send advertisement to agents (in production via pub/sub or broadcast RPC)
	select {
	case h.taskAds <- ad:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Start background goroutine to collect bids and evaluate
	go h.conductBiddingPhase(ctx, round)

	return round, nil
}

// conductBiddingPhase waits for bids until deadline, then evaluates
func (h *P2PBiddingHandler) conductBiddingPhase(ctx context.Context, round *BiddingRound) {
	biddingCtx, cancel := context.WithDeadline(ctx, round.BidDeadline)
	defer cancel()

	for {
		select {
		case bidResp := <-h.bidResponses:
			// Verify bid belongs to this round
			if bidResp.Bid.TaskID == round.TaskID {
				round.mu.Lock()
				round.ReceivedBids = append(round.ReceivedBids, bidResp.Bid)
				round.mu.Unlock()

				h.logger.Debug("Bid received in round",
					zap.String("round_id", round.RoundID),
					zap.String("bid_id", bidResp.Bid.BidID),
					zap.String("agent_id", bidResp.Bid.AgentID),
				)

				// Handle constraint updates from agent if present
				if bidResp.ConstraintChanges != nil {
					h.logger.Debug("Constraint changes from agent",
						zap.String("agent_id", bidResp.Bid.AgentID),
						zap.Any("changes", bidResp.ConstraintChanges),
					)
				}
			}

		case <-biddingCtx.Done():
			// Bidding phase ended
			round.mu.Lock()
			bidCount := len(round.ReceivedBids)
			round.mu.Unlock()

			h.logger.Info("Bidding phase ended",
				zap.String("round_id", round.RoundID),
				zap.Int("bids_received", bidCount),
			)

			// Proceed to evaluation
			h.evaluateAndAward(ctx, round)
			return
		}
	}
}

// evaluateAndAward selects winner and conducts confirmation phase
func (h *P2PBiddingHandler) evaluateAndAward(ctx context.Context, round *BiddingRound) {
	round.mu.RLock()
	if len(round.ReceivedBids) == 0 {
		round.mu.RUnlock()
		h.logger.Warn("No bids received, task failed",
			zap.String("round_id", round.RoundID),
			zap.String("task_id", round.TaskID),
		)
		round.mu.Lock()
		round.FinalStatus = "FAILED"
		round.mu.Unlock()
		return
	}

	// Register all bids with CNP orchestrator
	for _, bid := range round.ReceivedBids {
		if err := h.cnpOrch.RegisterBid(ctx, bid); err != nil {
			h.logger.Error("Failed to register bid",
				zap.Error(err),
				zap.String("bid_id", bid.BidID),
			)
		}
	}
	round.mu.RUnlock()

	// Evaluate and find winner
	score, err := h.cnpOrch.EvaluateBids(ctx, round.TaskID)
	if err != nil {
		h.logger.Error("Bid evaluation failed",
			zap.Error(err),
			zap.String("task_id", round.TaskID),
		)
		round.mu.Lock()
		round.FinalStatus = "FAILED"
		round.mu.Unlock()
		return
	}

	round.mu.Lock()
	round.WinnerBid = score
	round.mu.Unlock()

	// Send award to winner
	award := AwardNotification{
		BidID:              score.BidID,
		TaskID:             round.TaskID,
		AgentID:            score.AgentID,
		Award:              true,
		ConfirmationDeadline: round.ConfirmationDeadline,
	}

	awardMsg := AwardMessage{
		Award:                award,
		MessageID:            fmt.Sprintf("award_%s_%d", score.BidID, time.Now().UnixNano()),
		ConfirmationDeadline: round.ConfirmationDeadline,
	}

	h.logger.Info("Sending award to winner",
		zap.String("round_id", round.RoundID),
		zap.String("agent_id", score.AgentID),
		zap.Float64("score", score.Score),
	)

	// Send award (in production via RPC)
	select {
	case h.awardMsgs <- awardMsg:
	case <-ctx.Done():
		return
	}

	// Start confirmation phase
	h.conductConfirmationPhase(ctx, round, awardMsg)
}

// conductConfirmationPhase waits for agent confirmation
func (h *P2PBiddingHandler) conductConfirmationPhase(ctx context.Context, round *BiddingRound, awardMsg AwardMessage) {
	confirmCtx, cancel := context.WithDeadline(ctx, awardMsg.ConfirmationDeadline)
	defer cancel()

	confirmed := false

	for {
		select {
		case confirmMsg := <-h.confirmMsgs:
			if confirmMsg.BidID == awardMsg.Award.BidID {
				confirmed = true

				h.logger.Info("Award confirmed by agent",
					zap.String("round_id", round.RoundID),
					zap.String("agent_id", confirmMsg.AgentID),
					zap.String("status", confirmMsg.Status),
				)

				if confirmMsg.Status == "CONFIRMED" {
					round.mu.Lock()
					round.FinalStatus = "COMPLETED"
					round.mu.Unlock()

					// Handle schedule update if provided
					if confirmMsg.LocalScheduleUpdate != nil {
						h.logger.Debug("Local schedule update from agent",
							zap.String("agent_id", confirmMsg.AgentID),
							zap.Any("update", confirmMsg.LocalScheduleUpdate),
						)
					}
				} else {
					round.mu.Lock()
					round.FinalStatus = "FAILED"
					round.mu.Unlock()

					// Award was rejected, retry with next bidder
					h.retryWithNextBidder(ctx, round)
				}
				return
			}

		case <-confirmCtx.Done():
			// Confirmation timeout
			if !confirmed {
				h.logger.Warn("Confirmation timeout, winner failed to respond",
					zap.String("round_id", round.RoundID),
					zap.String("agent_id", awardMsg.Award.AgentID),
				)
				round.mu.Lock()
				round.FinalStatus = "FAILED"
				round.mu.Unlock()

				// Retry with next bidder
				h.retryWithNextBidder(ctx, round)
			}
			return
		}
	}
}

// retryWithNextBidder offers task to runner-up if winner rejects or times out
func (h *P2PBiddingHandler) retryWithNextBidder(ctx context.Context, round *BiddingRound) {
	round.mu.RLock()
	bids := round.ReceivedBids
	round.mu.RUnlock()

	if len(bids) <= 1 {
		h.logger.Warn("No alternative bidders, task failed",
			zap.String("round_id", round.RoundID),
			zap.String("task_id", round.TaskID),
		)
		return
	}

	h.logger.Info("Retrying with next best bidder",
		zap.String("round_id", round.RoundID),
		zap.Int("alternatives", len(bids)-1),
	)

	// In production, evaluate remaining bids and award to next best
	// For now, just mark as retry needed
}

// SubmitBid registers an agent's bid response
func (h *P2PBiddingHandler) SubmitBid(ctx context.Context, bidResp BidResponse) error {
	select {
	case h.bidResponses <- bidResp:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ConfirmAward registers agent's confirmation
func (h *P2PBiddingHandler) ConfirmAward(ctx context.Context, confirm ConfirmationMessage) error {
	select {
	case h.confirmMsgs <- confirm:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// GetBiddingRound returns status of a bidding round
func (h *P2PBiddingHandler) GetBiddingRound(roundID string) *BiddingRound {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if round, ok := h.ongoingBidRounds[roundID]; ok {
		// Return a copy
		copy := *round
		copy.ReceivedBids = make([]BidProposal, len(round.ReceivedBids))
		copy(copy.ReceivedBids, round.ReceivedBids)
		return &copy
	}
	return nil
}
