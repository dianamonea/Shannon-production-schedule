package control

import (
	"math"
	"testing"
	"time"
)

// ========== PID Controller Unit Tests ==========

// TestPIDStepResponse tests response to step input (simulating sudden delay)
func TestPIDStepResponse(t *testing.T) {
	controller := NewAdaptivePIDController(10 * time.Minute)

	tests := []struct {
		name              string
		completionPercent float64
		actualElapsedTime time.Duration
		expectedAction    string
		expectedSpeedMin  float64
		expectedSpeedMax  float64
	}{
		{
			name:              "On schedule at 50%",
			completionPercent: 0.5,
			actualElapsedTime: 5 * time.Minute,
			expectedAction:    "CONTINUE",
			expectedSpeedMin:  0.8,
			expectedSpeedMax:  1.2,
		},
		{
			name:              "Falling behind early (simulating delay)",
			completionPercent: 0.3,
			actualElapsedTime: 4 * time.Minute,
			expectedAction:    "ACCELERATE",
			expectedSpeedMin:  1.0,
			expectedSpeedMax:  1.5,
		},
		{
			name:              "Recovering from delay at 70%",
			completionPercent: 0.7,
			actualElapsedTime: 6 * time.Minute,
			expectedAction:    "CONTINUE",
			expectedSpeedMin:  1.0,
			expectedSpeedMax:  1.3,
		},
		{
			name:              "Severely behind (should trigger replan)",
			completionPercent: 0.2,
			actualElapsedTime: 8 * time.Minute,
			expectedAction:    "REPLAN",
			expectedSpeedMin:  0.5,
			expectedSpeedMax:  1.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := controller.CalculateControl("TEST_JOB", tt.actualElapsedTime, tt.completionPercent)

			if output.Action != tt.expectedAction {
				t.Errorf("Expected action %s, got %s", tt.expectedAction, output.Action)
			}

			if output.RecommendedSpeed < tt.expectedSpeedMin || output.RecommendedSpeed > tt.expectedSpeedMax {
				t.Errorf("Speed out of range [%.2f, %.2f]: got %.2f",
					tt.expectedSpeedMin, tt.expectedSpeedMax, output.RecommendedSpeed)
			}
		})
	}
}

// TestPIDParameterTuning tests parameter adjustment
func TestPIDParameterTuning(t *testing.T) {
	controller := NewAdaptivePIDController(10 * time.Minute)

	// Test initial parameters
	if controller.Kp != 0.5 || controller.Ki != 0.1 || controller.Kd != 0.05 {
		t.Errorf("Initial parameters incorrect")
	}

	// Test tuning
	controller.TuneParameters(0.7, 0.15, 0.1)

	if controller.Kp != 0.7 || controller.Ki != 0.15 || controller.Kd != 0.1 {
		t.Errorf("Tuning failed")
	}
}

// TestPIDAntiWindup tests integral windup prevention
func TestPIDAntiWindup(t *testing.T) {
	controller := NewAdaptivePIDController(10 * time.Minute)

	// Simulate sustained large error
	for i := 0; i < 100; i++ {
		output := controller.CalculateControl("TEST_JOB",
			time.Duration(10+i)*time.Second,
			float64(i)*0.001)

		// Integral term should not exceed limit
		if math.Abs(output.IntegralTerm) > controller.integralWindupLimit*controller.Ki {
			t.Errorf("Integral windup detected: %.2f", output.IntegralTerm)
		}
	}
}

// TestDeviationDetectorThreshold tests alert triggering
func TestDeviationDetectorThreshold(t *testing.T) {
	detector := NewDeviationDetector(15.0, 25.0) // 15% alert, 25% replan

	detector.RegisterJob("JOB_001", "AGENT_01", 10*time.Minute)

	tests := []struct {
		name            string
		completionPct   float64
		actualElapsed   time.Duration
		shouldAlert     bool
		shouldReplan    bool
	}{
		{
			name:          "On schedule",
			completionPct: 0.5,
			actualElapsed: 5 * time.Minute,
			shouldAlert:   false,
			shouldReplan:  false,
		},
		{
			name:          "15% behind (should alert)",
			completionPct: 0.5,
			actualElapsed: 5*time.Minute + 45*time.Second, // 15% delay
			shouldAlert:   true,
			shouldReplan:  false,
		},
		{
			name:          "30% behind (should replan)",
			completionPct: 0.5,
			actualElapsed: 6*time.Minute + 30*time.Second, // 30% delay
			shouldAlert:   true,
			shouldReplan:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector.RegisterJob("JOB_"+tt.name, "AGENT_01", 10*time.Minute)
			_ = detector.UpdateJobProgress("JOB_"+tt.name, tt.completionPct)

			alerts := detector.GetDeviationAlerts("JOB_" + tt.name)
			hasAlert := len(alerts) > 0
			hasReplan := false
			if len(alerts) > 0 {
				hasReplan = alerts[len(alerts)-1].RequiresReplanning
			}

			if hasAlert != tt.shouldAlert {
				t.Errorf("Alert expectation mismatch: expected %v, got %v", tt.shouldAlert, hasAlert)
			}

			if hasReplan != tt.shouldReplan {
				t.Errorf("Replan expectation mismatch: expected %v, got %v", tt.shouldReplan, hasReplan)
			}
		})
	}
}

// TestControlHistoryTracking tests history accumulation
func TestControlHistoryTracking(t *testing.T) {
	controller := NewAdaptivePIDController(10 * time.Minute)

	// Generate 50 control updates
	for i := 0; i < 50; i++ {
		controller.CalculateControl("JOB_001",
			time.Duration(i)*time.Second,
			float64(i)*0.01)
	}

	history := controller.GetControlHistory(10)
	if len(history) != 10 {
		t.Errorf("Expected 10 history entries, got %d", len(history))
	}

	// Verify chronological order
	for i := 1; i < len(history); i++ {
		if history[i].Timestamp.Before(history[i-1].Timestamp) {
			t.Errorf("History not in chronological order")
		}
	}
}

// ========== Integration Tests ==========

// TestClosedLoopSimulation simulates a complete closed-loop control cycle
func TestClosedLoopSimulation(t *testing.T) {
	activity := NewClosedLoopControlActivity()
	
	jobID := "SIM_JOB_001"
	agentID := "AGENT_ROBOT_01"
	estimatedDuration := 5 * time.Minute

	activity.devDetector.RegisterJob(jobID, agentID, estimatedDuration)
	activity.pidControl = NewAdaptivePIDController(estimatedDuration)

	// Simulate job execution with variance
	scenarioSteps := []struct {
		completionPercent float64
		delay             time.Duration // additional delay beyond expected
	}{
		{0.1, 0},         // 10% done, on time
		{0.2, 0},         // 20% done, on time
		{0.3, 30 * time.Second},  // 30% done, 30 sec late
		{0.5, 45 * time.Second},  // 50% done, 45 sec late
		{0.7, 1 * time.Minute},   // 70% done, 1 min late
		{0.9, 30 * time.Second},  // 90% done, recovered somewhat
		{1.0, 0},         // 100% done
	}

	for _, step := range scenarioSteps {
		expectedTime := time.Duration(float64(estimatedDuration) * step.completionPercent)
		actualTime := expectedTime + step.delay

		// Update deviation detector
		_ = activity.devDetector.UpdateJobProgress(jobID, step.completionPercent)

		// Calculate control output
		controlOutput := activity.pidControl.CalculateControl(jobID, actualTime, step.completionPercent)

		t.Logf("Progress: %.0f%% | Expected: %v | Actual: %v | Error: %.1fs | Action: %s | Speed: %.2fx",
			step.completionPercent*100,
			expectedTime,
			actualTime,
			controlOutput.Error,
			controlOutput.Action,
			controlOutput.RecommendedSpeed)

		// Verify control is within bounds
		if controlOutput.RecommendedSpeed < 0.5 || controlOutput.RecommendedSpeed > 1.5 {
			t.Errorf("Control output out of bounds: %.2f", controlOutput.RecommendedSpeed)
		}
	}

	alerts := activity.devDetector.GetDeviationAlerts(jobID)
	if len(alerts) == 0 {
		t.Errorf("Expected deviation alerts but got none")
	}

	t.Logf("Generated %d deviation alerts", len(alerts))
}

// TestEdgeDistillation tests rule extraction
func TestEdgeDistillation(t *testing.T) {
	controller := NewAdaptivePIDController(10 * time.Minute)
	rules := DistillComplexPolicyToEdgeRules(controller)

	if len(rules) < 5 {
		t.Errorf("Expected at least 5 distilled rules, got %d", len(rules))
	}

	for _, rule := range rules {
		if rule.RuleID == "" {
			t.Errorf("Rule has empty ID")
		}
		if rule.ConfidenceScore < 0 || rule.ConfidenceScore > 1.0 {
			t.Errorf("Invalid confidence score: %.2f", rule.ConfidenceScore)
		}
	}

	t.Logf("Distilled %d rules for edge execution", len(rules))
}

// BenchmarkPIDCalculation benchmarks control calculation speed
func BenchmarkPIDCalculation(b *testing.B) {
	controller := NewAdaptivePIDController(10 * time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		controller.CalculateControl("BENCH_JOB",
			time.Duration(i%600)*time.Second,
			float64(i%100)/100.0)
	}
}

// BenchmarkDeviationDetection benchmarks deviation calculation
func BenchmarkDeviationDetection(b *testing.B) {
	detector := NewDeviationDetector(15.0, 25.0)
	detector.RegisterJob("BENCH_JOB", "BENCH_AGENT", 10*time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = detector.UpdateJobProgress("BENCH_JOB", float64(i%100)/100.0)
	}
}
