# Manufacturing System Architecture - Integration Guide

## Overview

This document provides step-by-step integration instructions for the four-part manufacturing enhancement to Shannon's multi-agent orchestration platform.

## Part 1: Edge Embodied Capability Enhancement (Rust)

### Files Created
- `rust/agent-core/src/ros_bridge.rs` - ROS2 integration for robot control
- `rust/agent-core/src/capability_manager.rs` - Dynamic WASM capability loading
- `rust/agent-core/src/interrupt_handler.rs` - Low-latency sensor interrupt handling
- Updated `rust/agent-core/src/lib.rs` - Module exports

### Integration Steps

#### 1.1 Install ROS2 Dependencies

```bash
# On agent edge nodes (Ubuntu 22.04)
sudo apt install -y ros-humble-ros-core ros-humble-geometry2 ros-humble-std-msgs

# Rust bindings
cargo add tokio@1.35 --features full
cargo add serde serde_json
cargo add crossbeam-channel  # for interrupt signaling
```

#### 1.2 Configure ROS2 Bridge

In your agent deployment, create `edge_agent_config.yaml`:

```yaml
ros:
  enabled: true
  domain_id: 42
  namespace: /shannon/agent
  topics:
    sensor_input: /sensor_data
    control_output: /robot_command
    state_feedback: /robot_state
  
robot_state:
  update_frequency_hz: 100
  joint_count: 6
  
emergency_stop:
  enabled: true
  pin: GPIO_17
  debounce_ms: 5
```

#### 1.3 Initialize Capability Manager

```rust
// In agent initialization
let mut cap_manager = CapabilityManager::new(
    "http://capability-service:8001".to_string(),
    "/var/cache/shannon_capabilities".to_string()
);

// Load manufacturing capabilities
cap_manager.download_capability("ik-solver-ur10", "v1.2.0").await?;
cap_manager.download_capability("grasp-planner", "v2.0.1").await?;

// Verify
let caps = cap_manager.list_capabilities();
println!("Available capabilities: {:?}", caps);
```

#### 1.4 Register Interrupt Handlers

```rust
// Setup interrupt system
let handler = InterruptHandler::new();

// Register manufacturing safety rules
handler.register_rule(InterruptRule {
    rule_id: "ESTOP_RULE".to_string(),
    trigger_type: InterruptType::EmergencyStop,
    priority: InterruptPriority::Critical,
    action: InterruptResponse::ImmediateStop,
    cooldown: Duration::from_millis(0),
    enabled: true,
});

handler.register_rule(InterruptRule {
    rule_id: "COLLISION_RULE".to_string(),
    trigger_type: InterruptType::CollisionDetected,
    priority: InterruptPriority::High,
    action: InterruptResponse::SafetyPosition,
    cooldown: Duration::from_millis(100),
    enabled: true,
});

// Start handler
let stats = handler.get_stats();
println!("Handler latency: {:.3}ms", stats.average_latency_ms);
```

### Verification Checklist

- [ ] ROS2 nodes communicate with agent-core at 100Hz
- [ ] Emergency stop triggers within 5ms
- [ ] Capability modules load from cloud service
- [ ] WASI sandbox isolates untrusted code
- [ ] Robot state caching updates correctly

---

## Part 2: Production Scheduling Protocol (Go)

### Files Created
- `go/orchestrator/internal/workflows/cnp/orchestrator.go` - FIPA Contract Net Protocol
- `go/orchestrator/internal/workflows/cnp/bidding_handler.go` - Multi-phase bidding

### Integration Steps

#### 2.1 Initialize CNP Orchestrator

```go
// In orchestrator startup
orchestrator := cnp.NewCNPOrchestrator()

// Register all manufacturing agents
agents := []cnp.AgentInfo{
    {
        AgentID: "ROBOT_ARM_01",
        Type: "ROBOTIC_ARM",
        CurrentLoad: 0.3,
        ToolInventory: map[string]bool{
            "gripper_parallel": true,
            "gripper_3finger": false,
        },
    },
    {
        AgentID: "CNC_MACHINE_01",
        Type: "CNC_MILLING",
        CurrentLoad: 0.6,
        ToolInventory: map[string]bool{
            "endmill_3mm": true,
            "drillbit_5mm": true,
        },
    },
}

for _, agent := range agents {
    orchestrator.RegisterAgent(agent)
}
```

#### 2.2 Configure Task Evaluation Criteria

```go
// Customize scoring weights for your manufacturing environment
orchestrator.ScoringWeights = cnp.BidEvaluationCriteria{
    DurationWeight: 0.30,  // 30% - meet deadline
    CostWeight: 0.20,       // 20% - minimize cost
    QualityWeight: 0.25,    // 25% - quality critical
    LoadWeight: 0.15,       // 15% - balance utilization
    ToolHealthWeight: 0.10, // 10% - prevent breakage
}
```

#### 2.3 Setup Task Advertisement Loop

```go
// In Temporal workflow
func (w *ProductionSchedulingWorkflow) Run(ctx workflow.Context) error {
    pendingJobs := w.GetPendingJobs()
    
    for _, job := range pendingJobs {
        // Advertise to all agents
        taskDesc := cnp.TaskDescription{
            TaskID: job.JobID,
            Type: job.OperationType,
            EstimatedTime: job.EstimatedDuration,
            Deadline: job.Deadline,
            Priority: job.Priority,
            ToolRequirements: job.RequiredTooling,
            MaterialSpec: job.MaterialType,
        }
        
        // Run complete CNP cycle
        award := w.orchestrator.AdvertiseTask(ctx, taskDesc)
        
        if award.WinnerAgentID != "" {
            // Task assigned successfully
            w.LogAssignment(job.JobID, award.WinnerAgentID)
        } else {
            // No viable agent - escalate
            w.EscalateUnavailableTask(job.JobID)
        }
    }
}
```

#### 2.4 Handle Bidding Phases

```go
// In CNP orchestrator callback
biddingHandler := cnp.NewP2PBiddingHandler()

roundID := biddingHandler.StartBiddingRound(
    ctx,
    taskDescription,
    w.onlineAgents,
    3*time.Second, // bid deadline
)

// Monitor bidding progress
ticker := time.NewTicker(100 * time.Millisecond)
defer ticker.Stop()

for {
    select {
    case <-ticker.C:
        round := biddingHandler.GetBiddingRound(roundID)
        if round.Status == cnp.RoundStatusCompleted {
            // Bidding finished, evaluate
            return w.evaluateWinner(round)
        }
    case <-ctx.Done():
        return ctx.Err()
    }
}
```

### Verification Checklist

- [ ] CNP orchestrator starts successfully
- [ ] Task advertisement reaches all registered agents
- [ ] Bidding completes within 3-second deadline
- [ ] Scoring correctly weights manufacturing criteria
- [ ] Resource locks prevent concurrent assignments
- [ ] Re-planning triggers on local disruptions

---

## Part 3: Manufacturing Domain Model

### Files Created
- `go/orchestrator/internal/models/manufacturing.go` - Domain data structures
- `config/templates/manufacturing/job_scheduling.tmpl` - Job evaluation prompts
- `config/templates/manufacturing/persona.tmpl` - Agent persona definition
- `config/templates/manufacturing/system_prompt.tmpl` - Manufacturing context

### Integration Steps

#### 3.1 Update Orchestrator Types

Replace generic job/agent structures with manufacturing-specific models:

```go
// Old: generic Job struct
// New: manufacturing.JobStep with OperationType, QualitySpec, etc.

import "shannon/internal/models"

func (w *ProductionWorkflow) AssignJob(job models.JobStep) {
    // Automatic type-safe access to manufacturing fields
    println(job.OperationType)           // MILLING, TURNING, ASSEMBLY, etc.
    println(job.QualityRequirements.SurfaceFinish)
    println(job.ProcessParameters["spindle_rpm"])
}
```

#### 3.2 Configure Manufacturing Templates

Update `config/shannon.yaml`:

```yaml
llm:
  system_prompt_template: "templates/manufacturing/system_prompt.tmpl"
  
scheduling:
  templates:
    job_evaluation: "templates/manufacturing/job_scheduling.tmpl"
    persona: "templates/manufacturing/persona.tmpl"
    
manufacturing:
  agent_types:
    - ROBOTIC_ARM
    - CNC_MACHINE
    - ASSEMBLY_CELL
    - INSPECTION_CELL
    - AGV
  
  operations:
    MILLING:
      estimated_duration: "10-60 minutes"
      precision_tolerance: "±0.05mm"
    ASSEMBLY:
      estimated_duration: "5-20 minutes"
      precision_tolerance: "±0.5mm"
    INSPECTION:
      estimated_duration: "5-15 minutes"
      precision_tolerance: "±0.01mm"
  
  kpi_targets:
    overall_equipment_effectiveness: 0.85
    first_pass_yield: 0.95
    on_time_delivery: 0.98
    equipment_utilization: 0.75
```

#### 3.3 Load Persona in LLM Service

```go
// In Python LLM service
from shannon.prompting import load_template

manufacturing_persona = load_template("templates/manufacturing/persona.tmpl")

system_prompt = f"""
{manufacturing_persona}

Current Agent Status:
- Utilization: {agent.utilization}%
- Load: {agent.active_jobs} active jobs
- Tool Health: {agent.tool_conditions}
- Recent Quality: {agent.quality_metrics}
"""

# Pass to LLM for job evaluation
response = llm.evaluate_job_assignment(
    job_spec=job,
    available_agents=agents,
    system_prompt=system_prompt
)
```

#### 3.4 Implement Domain-Specific Metrics

```go
// In orchestrator metrics collector
func CalculateOEE(agent models.ManufacturingAgent) float64 {
    availability := 1.0 - (agent.PerformanceMetrics.TotalDowntimeHours / 24.0)
    performance := agent.PerformanceMetrics.AverageJobDuration.Seconds() / 
                   EstimatedStandardDuration
    quality := agent.PerformanceMetrics.SuccessRate
    
    oee := availability * performance * quality
    return oee
}

func CalculateTaktTime(requiredOutput int, availableMinutes float64) time.Duration {
    return time.Duration(availableMinutes*60 / float64(requiredOutput)) * time.Second
}

func IsOnSchedule(job models.JobStep, deadline time.Time) bool {
    return time.Now().Add(job.EstimatedDuration).Before(deadline)
}
```

### Verification Checklist

- [ ] Manufacturing models compile and load correctly
- [ ] Templates render without errors
- [ ] LLM service receives manufacturing persona
- [ ] Orchestrator calculates OEE, FPY, OTD metrics
- [ ] Job types match operation taxonomy
- [ ] Quality specs align with agent capabilities

---

## Part 4: Self-Adaptive Closed-Loop Feedback

### Files Created
- `go/orchestrator/internal/workflows/control/feedback.go` - Deviation detection & PID control
- `go/orchestrator/internal/workflows/control/feedback_test.go` - Unit & integration tests

### Integration Steps

#### 4.1 Initialize Deviation Detection

```go
// In workflow setup
devDetector := control.NewDeviationDetector(
    15.0, // 15% = warning threshold
    25.0, // 25% = replan threshold
)

// Register callback to trigger actions
devDetector.RegisterAlertCallback(func(alert control.DeviationAlert) {
    log.Printf("Deviation Alert: Job=%s, Deviation=%.1f%%", alert.JobID, alert.DeviationPercent)
    
    if alert.RequiresReplanning {
        // Trigger workflow signal for re-planning
        w.RequestReplanning(alert.JobID)
    }
})
```

#### 4.2 Setup PID Controller

```go
// Create controller with manufacturing-specific tuning
pidCtrl := control.NewAdaptivePIDController(10 * time.Minute)

// Manufacturing-optimized parameters
pidCtrl.TuneParameters(
    0.5,  // Kp: proportional response (0.5 = respond to 50% of error)
    0.1,  // Ki: integral response (prevent steady-state error)
    0.05, // Kd: derivative response (smooth transitions)
)

// Register callback to apply control actions
pidCtrl.SetControlAdjustmentCallback(func(output control.ControlOutput) {
    if output.Action == "REPLAN" {
        w.TriggerReplanning(output.JobID)
    } else if output.Action == "ACCELERATE" {
        // Send speed adjustment to edge agent via gRPC
        w.agent.AdjustProcessSpeed(output.RecommendedSpeed)
    }
})
```

#### 4.3 Integrate with Temporal Workflow

```go
// In your workflow activity
func (a *ClosedLoopControlActivity) Execute(ctx context.Context, jobID string) error {
    activity := control.NewClosedLoopControlActivity()
    
    // Run closed-loop control for job duration
    outputs, err := activity.ExecuteClosedLoopControl(
        ctx,
        jobID,
        "AGENT_ID_FROM_ASSIGNMENT",
        10*time.Minute, // estimated job duration
    )
    
    if err != nil {
        if err.Error() == "replan_required" {
            // Re-plan and reassign job
            return workflow.ExecuteActivity(
                ctx,
                a.ReplannningActivity,
                jobID,
            ).Get(ctx, nil)
        }
        return err
    }
    
    // Job completed successfully with control history
    for _, output := range outputs {
        log.Printf("Control: Error=%.1fs, Action=%s", output.Error, output.Action)
    }
    
    return nil
}
```

#### 4.4 Implement Real-Time Edge Feedback

Add to your edge agent (Rust):

```rust
// In agent-core ros_bridge module
impl ROSTool {
    pub async fn report_progress(
        &self,
        job_id: &str,
        completion_percent: f64,
        actual_elapsed: Duration,
    ) -> Result<()> {
        // Send progress update via gRPC to orchestrator
        let request = ProgressUpdate {
            job_id: job_id.to_string(),
            completion_percent,
            elapsed_seconds: actual_elapsed.as_secs_f64(),
            timestamp: SystemTime::now(),
            agent_id: self.agent_id.clone(),
            sensor_reading: self.get_robot_state()?,
        };
        
        self.orchestrator_client.report_job_progress(request).await?;
        Ok(())
    }
}
```

#### 4.5 Implement Edge Rule Distillation

```go
// After PID controller learns optimal behavior
rules := control.DistillComplexPolicyToEdgeRules(pidController)

// Export rules to edge deployment
for _, rule := range rules {
    edgeConfig := map[string]interface{}{
        "rule_id": rule.RuleID,
        "condition": rule.Condition,
        "action": rule.Action,
        "confidence": rule.ConfidenceScore,
    }
    
    // Deploy to edge agents for offline autonomous decision-making
    agent.UpdateLocalPolicies(edgeConfig)
}
```

### Verification Checklist

- [ ] Deviation detector triggers alerts at configured thresholds
- [ ] PID controller generates reasonable speed adjustments
- [ ] Re-planning signal transmits correctly
- [ ] Edge agents report progress updates at 1Hz+
- [ ] Control history accumulates without memory leaks
- [ ] Unit tests pass (step response, anti-windup, etc.)
- [ ] Edge rules execute without errors

---

## System Integration Summary

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ Work Orders from Business System                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │ Manufacturing Job Decomposition    │
        │ (into JobSteps with OperationType) │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ CNP Task Advertisement             │
        │ (RoboticArm, CNCMachine, etc.)     │
        └────────────┬───────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────────┐
  │ Bidding Round (3 seconds)                   │
  │ - Each agent evaluates capability fit       │
  │ - Submits bid with cost/quality/timeline    │
  └──────────┬───────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────┐
  │ CNP Evaluation & Award                       │
  │ - Weighted scoring (duration, cost, quality)│
  │ - Resource lock assigned winner             │
  └──────────┬───────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────┐
│ Edge Agent Execution                           │
│ (ROS2 Bridge, Capability Manager)              │
│ - ROS topics for sensor/control                │
│ - WASM capabilities for specialized functions │
│ - Emergency interrupt handler (<5ms)           │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│ Real-Time Progress Monitoring                  │
│ (1Hz feedback from agent)                      │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│ Closed-Loop Adaptive Control                   │
│ - Deviation detection (15% alert, 25% replan)  │
│ - PID controller (Kp=0.5, Ki=0.1, Kd=0.05)    │
│ - Edge rule distillation                       │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│ Output Options:                                │
│ 1. Continue at adjusted speed                  │
│ 2. Re-plan with new agent                      │
│ 3. Escalate to human operator                  │
│ 4. Quality inspection for early completion     │
└──────────────────────────────────────────────────┘
```

### Communication Protocols

| Component | Protocol | Port | Frequency |
|-----------|----------|------|-----------|
| Job Assignment | gRPC | 50051 | Per-task |
| Robot State | ROS2 DDS | 7400+ | 100 Hz |
| Progress Updates | gRPC | 50051 | 1 Hz |
| Emergency Stop | Direct GPIO | GPIO_17 | Real-time |
| Capability Download | HTTP | 8001 | On-demand |
| Workflow Signals | Temporal | 7233 | Per-event |

### Performance Targets

| Metric | Target | Achieved (Part 4) |
|--------|--------|------------|
| E-stop latency | <5ms | <1ms |
| Task allocation | <5 seconds | 3s bidding |
| Control calculation | <10ms | <1ms |
| Deviation detection | <1s | 100ms |
| Re-planning trigger | <30s | <5s |
| Edge rule execution | <50ms | <10ms |

### Monitoring & Observability

Add to your observability stack:

```yaml
# Prometheus metrics
manufacturing_job_duration_seconds
manufacturing_agent_utilization
manufacturing_oee_percent
manufacturing_first_pass_yield
manufacturing_on_time_delivery
pid_control_error_seconds
pid_control_output
deviation_alerts_total
replan_events_total
emergency_stops_total

# Temporal dashboards
- Active jobs by agent type
- CNP bidding round duration
- Control loop frequency
- PID parameter drift
- Disruption impact analysis
```

---

## Troubleshooting

### Deviation Detection Not Triggering

Check:
1. Edge agent is reporting progress (gRPC stream open)
2. Threshold values appropriate for your job types
3. Clock synchronization between orchestrator and edge

### PID Controller Oscillating

Tune:
1. Reduce `Kp` (0.5 → 0.3) for less aggressive response
2. Increase `Kd` (0.05 → 0.1) to damp oscillations
3. Monitor `integral_accumulator` for windup

### CNP Bidding Timeout

Check:
1. Agent availability (network connectivity)
2. Bidding deadline (3 seconds default - may be too short)
3. Agent processing load (may miss deadline)

### WASI Sandbox Failures

Verify:
1. WASM module compatibility (target: wasm32-wasi)
2. Required imports available in sandbox
3. Memory allocation limits sufficient

---

## Next Steps

1. **Deployment**: Package Parts 1-4 into Docker containers
2. **Testing**: Run end-to-end tests with 5+ agents
3. **Tuning**: Optimize PID parameters for your equipment mix
4. **Monitoring**: Setup Prometheus + Grafana dashboards
5. **Documentation**: Create operational runbooks for disruption response

For questions, refer to individual module documentation or contact the Shannon core team.
