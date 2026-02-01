package workflows

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestQualityRecovery_RecordInspection(t *testing.T) {
	qr := NewQualityRecovery()

	// Test passing inspection
	inspection1 := &QualityInspection{
		InspectionID:  "INSP_001",
		JobID:         "JOB_001",
		AgentID:       "AGENT_001",
		Metric:        DimensionalAccuracy,
		MeasuredValue: 10.05,
		TargetValue:   10.0,
		Tolerance:     0.1,
		DetectedAt:    time.Now(),
	}

	qr.RecordInspection(inspection1)

	assert.False(t, inspection1.IsDefective, "Should pass inspection")
	assert.Len(t, qr.defects, 0, "Should have no defects")

	// Test failing inspection
	inspection2 := &QualityInspection{
		InspectionID:  "INSP_002",
		JobID:         "JOB_002",
		AgentID:       "AGENT_001",
		Metric:        DimensionalAccuracy,
		MeasuredValue: 10.5,
		TargetValue:   10.0,
		Tolerance:     0.1,
		DetectedAt:    time.Now(),
	}

	qr.RecordInspection(inspection2)

	assert.True(t, inspection2.IsDefective, "Should fail inspection")
	assert.Len(t, qr.defects, 1, "Should have 1 defect")
}

func TestQualityRecovery_ReworkAction(t *testing.T) {
	qr := NewQualityRecovery()

	// Create defective inspection
	inspection := &QualityInspection{
		InspectionID:  "INSP_003",
		JobID:         "JOB_003",
		AgentID:       "AGENT_002",
		Metric:        SurfaceFinish,
		MeasuredValue: 5.0,
		TargetValue:   1.0,
		Tolerance:     0.5,
		DetectedAt:    time.Now(),
	}

	qr.RecordInspection(inspection)

	// Check rework queue
	reworkQueue := qr.GetReworkQueue()
	assert.Len(t, reworkQueue, 1, "Should have 1 rework action")
	assert.Equal(t, "REPOLISH", reworkQueue[0].Action)
	assert.Equal(t, "PENDING", reworkQueue[0].Status)

	// Complete rework
	err := qr.CompleteReworkAction(reworkQueue[0].ActionID, true)
	assert.NoError(t, err)

	// Check status changed
	updatedQueue := qr.GetReworkQueue()
	assert.Len(t, updatedQueue, 0, "Should have no pending rework actions")
}

func TestQualityRecovery_SPCChart(t *testing.T) {
	qr := NewQualityRecovery()

	// Add samples to build SPC chart
	for i := 0; i < 25; i++ {
		inspection := &QualityInspection{
			InspectionID:  fmt.Sprintf("INSP_%d", i),
			JobID:         fmt.Sprintf("JOB_%d", i),
			AgentID:       "AGENT_001",
			Metric:        DimensionalAccuracy,
			MeasuredValue: 10.0 + float64(i)*0.01,
			TargetValue:   10.0,
			Tolerance:     0.5,
			DetectedAt:    time.Now(),
		}
		qr.RecordInspection(inspection)
	}

	// Check SPC chart exists
	chart, exists := qr.GetSPCChart(DimensionalAccuracy)
	assert.True(t, exists, "SPC chart should exist")
	assert.GreaterOrEqual(t, len(chart.Samples), 20, "Should have sufficient samples")
	assert.Greater(t, chart.UCL, chart.Mean, "UCL should be above mean")
	assert.Less(t, chart.LCL, chart.Mean, "LCL should be below mean")

	// Add out-of-control sample
	outOfControl := &QualityInspection{
		InspectionID:  "INSP_OUT",
		JobID:         "JOB_OUT",
		AgentID:       "AGENT_001",
		Metric:        DimensionalAccuracy,
		MeasuredValue: 15.0, // Way out of spec
		TargetValue:   10.0,
		Tolerance:     0.5,
		DetectedAt:    time.Now(),
	}
	qr.RecordInspection(outOfControl)

	// Check chart flagged as out of control
	chart, _ = qr.GetSPCChart(DimensionalAccuracy)
	assert.True(t, chart.OutOfControl, "Chart should be out of control")
}

func TestQualityRecovery_DefectRate(t *testing.T) {
	qr := NewQualityRecovery()

	// Add 10 inspections, 2 defective
	for i := 0; i < 10; i++ {
		value := 10.0
		if i < 2 {
			value = 12.0 // Out of tolerance
		}

		inspection := &QualityInspection{
			InspectionID:  fmt.Sprintf("INSP_%d", i),
			JobID:         fmt.Sprintf("JOB_%d", i),
			AgentID:       "AGENT_001",
			Metric:        DimensionalAccuracy,
			MeasuredValue: value,
			TargetValue:   10.0,
			Tolerance:     0.5,
			DetectedAt:    time.Now(),
		}
		qr.RecordInspection(inspection)
	}

	defectRate := qr.CalculateDefectRate()
	assert.Equal(t, 0.2, defectRate, "Defect rate should be 20%")
}

func TestQualityRecovery_RootCauseAnalysis(t *testing.T) {
	qr := NewQualityRecovery()

	// Create multiple defects from same agent
	for i := 0; i < 5; i++ {
		defect := &Defect{
			DefectID:   fmt.Sprintf("DEFECT_%d", i),
			JobID:      fmt.Sprintf("JOB_%d", i),
			Type:       DefectDimensional,
			Severity:   SeverityModerate,
			DetectedAt: time.Now(),
			AgentID:    "AGENT_BAD",
			ToolID:     "TOOL_001",
		}
		qr.defects[defect.DefectID] = defect
	}

	// Perform root cause analysis
	analysis, err := qr.PerformRootCauseAnalysis("DEFECT_0")
	assert.NoError(t, err)
	assert.Equal(t, "AGENT_MALFUNCTION", analysis.ProbableRoot)
	assert.Greater(t, analysis.Confidence, 0.7)
	assert.Contains(t, analysis.RecommendedAction, "AGENT_BAD")
}

func TestQualityRecovery_DefectSeverity(t *testing.T) {
	qr := NewQualityRecovery()

	// Minor deviation
	minor := qr.calculateSeverity(0.1, 0.1)
	assert.Equal(t, SeverityMinor, minor)

	// Moderate deviation
	moderate := qr.calculateSeverity(0.2, 0.1)
	assert.Equal(t, SeverityModerate, moderate)

	// Critical deviation
	critical := qr.calculateSeverity(0.5, 0.1)
	assert.Equal(t, SeverityCritical, critical)
}

func TestQualityRecovery_ShouldTriggerInvestigation(t *testing.T) {
	qr := NewQualityRecovery()

	// Add inspections to exceed defect rate threshold
	for i := 0; i < 20; i++ {
		value := 10.0
		if i < 2 {
			value = 12.0 // Defective
		}

		inspection := &QualityInspection{
			InspectionID:  fmt.Sprintf("INSP_INV_%d", i),
			JobID:         fmt.Sprintf("JOB_INV_%d", i),
			AgentID:       "AGENT_001",
			Metric:        DimensionalAccuracy,
			MeasuredValue: value,
			TargetValue:   10.0,
			Tolerance:     0.5,
			DetectedAt:    time.Now(),
		}
		qr.RecordInspection(inspection)
	}

	assert.True(t, qr.ShouldTriggerInvestigation())
}

func TestQualityRecovery_PredictDefectRisk(t *testing.T) {
	qr := NewQualityRecovery()

	// Build SPC chart
	for i := 0; i < 25; i++ {
		inspection := &QualityInspection{
			InspectionID:  fmt.Sprintf("INSP_RISK_%d", i),
			JobID:         fmt.Sprintf("JOB_RISK_%d", i),
			AgentID:       "AGENT_001",
			Metric:        SurfaceFinish,
			MeasuredValue: 1.0 + float64(i)*0.01,
			TargetValue:   1.0,
			Tolerance:     0.5,
			DetectedAt:    time.Now(),
		}
		qr.RecordInspection(inspection)
	}

	risk := qr.PredictDefectRisk(SurfaceFinish)
	assert.GreaterOrEqual(t, risk, 0.1)
}

func TestQualityRecovery_BuildQualityDisruptionEvent(t *testing.T) {
	qr := NewQualityRecovery()
	meta := map[string]interface{}{"metric": "SURFACE_FINISH"}
	de := qr.BuildQualityDisruptionEvent("JOB_123", 7, meta)
	assert.Equal(t, "JOB_123", de.AffectedAsset)
	assert.Equal(t, 7, de.Severity)
	assert.Equal(t, "SURFACE_FINISH", de.Metadata["metric"])
}

func TestQualityRecovery_InvestigationCallback(t *testing.T) {
	qr := NewQualityRecovery()
	qr.SetInvestigationCooldown(0)

	triggered := make(chan float64, 1)
	qr.RegisterInvestigationCallback(func(rate float64) {
		triggered <- rate
	})

	for i := 0; i < 20; i++ {
		value := 10.0
		if i < 2 {
			value = 12.0
		}
		inspection := &QualityInspection{
			InspectionID:  fmt.Sprintf("INSP_CB_%d", i),
			JobID:         fmt.Sprintf("JOB_CB_%d", i),
			AgentID:       "AGENT_001",
			Metric:        DimensionalAccuracy,
			MeasuredValue: value,
			TargetValue:   10.0,
			Tolerance:     0.5,
			DetectedAt:    time.Now(),
		}
		qr.RecordInspection(inspection)
	}

	select {
	case rate := <-triggered:
		assert.Greater(t, rate, 0.0)
	case <-time.After(500 * time.Millisecond):
		assert.Fail(t, "investigation callback not triggered")
	}
}
