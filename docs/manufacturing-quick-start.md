# Manufacturing System Quick Start Guide

## 5-Minute Setup

### 1. Verify Code Files Are in Place

```bash
# Rust components
ls -la rust/agent-core/src/ros_bridge.rs
ls -la rust/agent-core/src/capability_manager.rs
ls -la rust/agent-core/src/interrupt_handler.rs

# Go components
ls -la go/orchestrator/internal/models/manufacturing.go
ls -la go/orchestrator/internal/workflows/cnp/orchestrator.go
ls -la go/orchestrator/internal/workflows/cnp/bidding_handler.go
ls -la go/orchestrator/internal/workflows/control/feedback.go

# Configuration
ls -la config/templates/manufacturing/
```

### 2. Compile Rust Components

```bash
cd rust/agent-core

# Build with manufacturing features
cargo build --release --features "ros2,wasi,manufacturing"

# Run tests
cargo test --lib interrupt_handler
cargo test --lib capability_manager
```

### 3. Compile Go Components

```bash
cd go/orchestrator

# Build orchestrator with manufacturing models
go build -o bin/orchestrator ./cmd/orchestrator

# Run tests
go test ./internal/workflows/control -v
go test ./internal/models -v
```

### 4. Load Manufacturing Templates

```bash
# Verify templates load without errors
grep -r "system_prompt_template" config/shannon.yaml
grep -r "job_evaluation" config/shannon.yaml

# Update config if needed
cat > config/shannon-manufacturing.yaml << 'EOF'
manufacturing:
  enabled: true
  
llm:
  system_prompt_template: "templates/manufacturing/system_prompt.tmpl"
  
templates:
  job_evaluation: "templates/manufacturing/job_scheduling.tmpl"
  persona: "templates/manufacturing/persona.tmpl"
EOF
```

### 5. Test Closed-Loop Control

```bash
cd go/orchestrator

# Run unit tests
go test -run TestPIDStepResponse ./internal/workflows/control -v
go test -run TestClosedLoopSimulation ./internal/workflows/control -v

# Expected output:
# Progress: 10% | Expected: 30s | Error: +5s | Action: ACCELERATE | Speed: 1.20x
# Progress: 50% | Expected: 150s | Error: -10s | Action: CONTINUE | Speed: 0.95x
# ...test passes...
```

---

## Testing the Four Components

### Test Part 1: ROS2 Bridge

```bash
# Requires ROS2 environment
source /opt/ros/humble/setup.bash

# Start ROS2 simulator
ros2 launch ros2_control_demo_example_13 diffbot_system.launch.py &

# Connect agent-core
cd rust/agent-core
cargo run --example ros_bridge_test

# Expected: Joint states subscribe/publish at 100Hz
```

### Test Part 2: CNP Orchestrator

```bash
cd go/orchestrator

# Create test fixture
cat > test_jobs.json << 'EOF'
[
  {
    "job_id": "JOB_001",
    "operation_type": "MILLING",
    "material": "STEEL",
    "estimated_duration": "30m",
    "deadline": "2024-01-15T14:00:00Z",
    "priority": 8
  },
  {
    "job_id": "JOB_002",
    "operation_type": "ASSEMBLY",
    "estimated_duration": "15m",
    "deadline": "2024-01-15T15:00:00Z",
    "priority": 5
  }
]
EOF

# Run CNP test
go test -run TestCNPAdvertisement ./internal/workflows/cnp -v

# Expected: Task advertised to all agents, bidding completes in 3 seconds
```

### Test Part 3: Manufacturing Models

```bash
cd go/orchestrator

# Verify type safety
go run -tags=manufactuing_models ./cmd/test_models

# Check output:
# ✓ WorkOrder type loaded
# ✓ JobStep with OperationType
# ✓ BillOfMaterials structure
# ✓ QualitySpec validation
# ✓ ManufacturingAgent metrics
```

### Test Part 4: PID Control Loop

```bash
cd go/orchestrator

# Run closed-loop simulation
go test -run TestClosedLoopSimulation ./internal/workflows/control -v -count=1

# Examine output:
# Time=0: Deviation=0%, Action=CONTINUE
# Time=1: Deviation=+15%, Action=ACCELERATE, Speed=1.20x
# Time=2: Deviation=+8%, Action=CONTINUE, Speed=1.05x
# Time=3: Deviation=0%, Action=CONTINUE, Speed=1.00x

# Verify no latency violations
go test -bench BenchmarkPID ./internal/workflows/control
# Expected: ~1ms per calculation
```

---

## Configuration Checklist

### Environment Variables

```bash
# Edge agent
export ROS_DOMAIN_ID=42
export SHANNON_AGENT_ID=ROBOT_001
export SHANNON_AGENT_TYPE=ROBOTIC_ARM
export CAPABILITY_SERVICE_URL=http://capability-service:8001

# Orchestrator
export TEMPORAL_SERVER=localhost:7233
export TEMPORAL_NAMESPACE=manufacturing
export LLM_SERVICE_URL=http://llm-service:8000
export ORCHESTRATOR_PORT=50051

# Manufacturing
export MFG_DEVIATION_THRESHOLD=15.0
export MFG_REPLAN_THRESHOLD=25.0
```

### Port Configuration

```yaml
Orchestrator:
  gRPC: 50051
  HTTP: 8080
  Temporal: 7233

Edge Agents:
  ROS2 DDS: 7400-7410 (dynamic)
  gRPC: 50052+ (per agent)

LLM Service:
  HTTP: 8000
  Inference: 8001

Capability Service:
  HTTP: 8001
```

### Database Setup

```sql
-- Add manufacturing tables if not present
CREATE TABLE manufacturing_agents (
    agent_id VARCHAR(255) PRIMARY KEY,
    agent_type VARCHAR(50),
    current_load FLOAT,
    status VARCHAR(20),
    last_heartbeat TIMESTAMP
);

CREATE TABLE job_assignments (
    job_id VARCHAR(255),
    agent_id VARCHAR(255),
    assigned_at TIMESTAMP,
    estimated_completion TIMESTAMP,
    PRIMARY KEY (job_id)
);

CREATE TABLE production_metrics (
    metric_time TIMESTAMP,
    agent_id VARCHAR(255),
    oee_percent FLOAT,
    utilization FLOAT,
    first_pass_yield FLOAT
);

CREATE INDEX idx_agent_assignments ON job_assignments(agent_id);
CREATE INDEX idx_metric_time ON production_metrics(metric_time DESC);
```

---

## Common Operations

### Launch Manufacturing System

```bash
# 1. Start Temporal
docker run -d --name temporal temporal:latest

# 2. Start orchestrator
cd go/orchestrator
./bin/orchestrator \
  --temporal-address localhost:7233 \
  --manufacturing-mode=true \
  --port 50051

# 3. Start edge agents (per physical robot)
cd rust/agent-core
./target/release/shannon-agent \
  --agent-id ROBOT_001 \
  --agent-type ROBOTIC_ARM \
  --ros-enabled=true \
  --capability-service http://localhost:8001

# 4. Start LLM service
cd python
python -m shannon.llm_service \
  --templates-dir ../config/templates/manufacturing \
  --port 8000
```

### Submit a Manufacturing Job

```bash
# Create work order
curl -X POST http://localhost:8080/work-orders \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "ACME_CORP",
    "required_date": "2024-01-15T16:00:00Z",
    "jobs": [
      {
        "operation_type": "MILLING",
        "material": "STEEL",
        "quantity": 5,
        "precision": 0.05,
        "deadline": "2024-01-15T14:00:00Z",
        "priority": 8
      }
    ]
  }'

# Returns: order_id, estimated_completion, assigned_agent
```

### Monitor Production in Real-Time

```bash
# Watch CNP bidding rounds
watch -n 1 'curl -s http://localhost:50051/cnp/active-biddings | jq'

# Check agent utilization
curl -s http://localhost:50051/agents/metrics | jq '.[] | {agent_id, utilization, oee}'

# View deviation alerts
curl -s http://localhost:50051/control/alerts?last_hours=1 | jq '.[] | {job_id, deviation_percent, action}'

# Get PID control history
curl -s http://localhost:50051/control/history?job_id=JOB_001 | jq '.[] | {timestamp, error, action, speed}'
```

### Trigger Manual Re-Planning

```bash
# Signal orchestrator to re-plan
curl -X POST http://localhost:50051/workflows/replan \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "MATERIAL_SHORTAGE",
    "affected_jobs": ["JOB_001", "JOB_002"],
    "priority": "HIGH"
  }'
```

### Emergency Stop (All Agents)

```bash
# Immediate shutdown - triggers interrupt handler
curl -X POST http://localhost:50051/safety/emergency-stop

# Or via GPIO on edge agent
echo 1 > /sys/class/gpio/gpio17/value
```

---

## Troubleshooting

### ROS2 Connection Failed

```bash
# Check domain ID
echo $ROS_DOMAIN_ID  # should be 42

# Verify topic availability
ros2 topic list  # should show /sensor_data, /robot_command, /robot_state

# Check network
ros2 doctor
```

### CNP Bidding Timeout

```bash
# Agents not responding?
curl http://localhost:50051/agents  # check online status

# Increase bidding timeout
export CNP_BID_TIMEOUT=5000  # 5 seconds

# Check agent logs
docker logs shannon-agent-robot-001 | grep -i "error\|timeout"
```

### PID Control Oscillating

```bash
# Check parameters
curl http://localhost:50051/control/parameters | jq '.pid'

# Reduce proportional gain
curl -X PATCH http://localhost:50051/control/parameters \
  -d '{
    "kp": 0.3,  # reduce from 0.5
    "ki": 0.1,
    "kd": 0.1   # increase from 0.05
  }'

# Monitor next job
curl http://localhost:50051/control/history?job_id=JOB_003 | head -20
```

### Deviation Detector Not Alerting

```bash
# Verify thresholds
echo $MFG_DEVIATION_THRESHOLD  # should be 15

# Check if agent reports progress
docker logs shannon-agent-robot-001 | grep "report_job_progress"

# Manually trigger test alert
curl -X POST http://localhost:50051/control/test-deviation \
  -d '{
    "job_id": "TEST_JOB",
    "completion_percent": 0.5,
    "simulated_delay": 30  # 30 seconds
  }'
```

### WASM Capability Load Failure

```bash
# Check capability service
curl http://capability-service:8001/health

# List available capabilities
curl http://capability-service:8001/capabilities | jq '.[] | .name'

# Manual download & verify
wget http://capability-service:8001/capabilities/ik-solver-ur10/v1.2.0/module.wasm
file module.wasm  # should be "WebAssembly (wasm) binary module"
```

---

## Performance Verification

```bash
# Benchmark PID calculation latency
go test -bench BenchmarkPID ./internal/workflows/control -benchmem -count 5

# Expected:
# BenchmarkPIDCalculation-8       1000000    956 ns/op    ~1 microsecond

# Check deviation detection speed
go test -bench BenchmarkDeviation ./internal/workflows/control -benchmem -count 5

# Expected:
# BenchmarkDeviationDetection-8   1000000    987 ns/op    ~1 microsecond

# Monitor real-time performance
watch -n 1 'curl -s http://localhost:50051/metrics/performance | jq'
```

---

## Success Criteria

Your manufacturing system is ready when:

- ✅ All 4 Rust modules compile without warnings
- ✅ All 4 Go packages compile and tests pass
- ✅ ROS2 agents connect at 100Hz (verified with `ros2 topic hz`)
- ✅ CNP bidding completes in <3 seconds
- ✅ Deviation detector triggers alerts at configured thresholds
- ✅ PID controller adjusts speed within 1ms
- ✅ OEE tracking shows >85% utilization
- ✅ First-pass yield tracking > 90%
- ✅ LLM persona responses reference manufacturing concepts
- ✅ E-stop activates within 5ms

---

## Next: Production Deployment

See `docs/manufacturing-integration-guide.md` for:
- Comprehensive 7-section integration guide
- Data flow architecture diagrams
- Communication protocol specifications
- Monitoring setup with Prometheus/Grafana
- Operational runbooks
- Troubleshooting procedures

**Status**: Manufacturing system ready for pilot deployment. 🚀
