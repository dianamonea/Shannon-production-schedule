"""
单元测试和端到端演示
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch

# 添加到路径
sys.path.insert(0, str(Path(__file__).parent))

from common.schemas import (
    RobotState, JobSpec, StationState, StepObservation,
    RobotAction, PolicyActRequest, RobotStatus, StationType,
)
from common.vectorizer import StateVectorizer, ActionVectorizer
from training.model import DecisionTransformer, MADTLoss
from app import PolicyService, PolicyServiceConfig


# ============================================================
# 1. Schema 验证测试
# ============================================================

def test_schemas():
    """测试 Pydantic schema 验证"""
    print("\n=== Test 1: Schema Validation ===")
    
    # 创建有效的观测
    robot = RobotState(
        robot_id="robot_0",
        position={"x": 10.0, "y": 20.0},
        status=RobotStatus.IDLE,
        battery_level=85.5,
    )
    
    job = JobSpec(
        job_id="job_0",
        job_type="assembly",
        source_station_id="station_0",
        target_station_id="station_1",
        deadline=100.0,
        priority=75,
    )
    
    station = StationState(
        station_id="station_0",
        station_type=StationType.ASSEMBLY,
        position={"x": 0.0, "y": 0.0},
        is_available=True,
    )
    
    obs = StepObservation(
        t=0,
        robots=[robot],
        jobs=[job],
        stations=[station],
    )
    
    print("✓ Created valid StepObservation")
    print(f"  - Robots: {[r.robot_id for r in obs.robots]}")
    print(f"  - Jobs: {[j.job_id for j in obs.jobs]}")
    print(f"  - Stations: {[s.station_id for s in obs.stations]}")
    
    # 测试无效的 battery_level（应该在 0-100）
    try:
        invalid_robot = RobotState(
            robot_id="robot_bad",
            battery_level=150.0,  # 无效
        )
        print("✗ Failed to catch invalid battery_level")
    except ValueError as e:
        print(f"✓ Correctly caught validation error: {e}")
    
    return obs


def test_vectorizer(obs: StepObservation):
    """测试向量化器"""
    print("\n=== Test 2: Vectorizer ===")
    
    vectorizer = StateVectorizer(
        max_robots=10,
        max_jobs=50,
        max_stations=20,
        embed_dim=128,
    )
    
    # 向量化单步观测
    vec_state = vectorizer.vectorize_step(obs)
    
    print("✓ Vectorized step observation")
    print(f"  - Robot embeddings shape: {len(vec_state.robot_embeddings)} x {len(vec_state.robot_embeddings[0])}")
    print(f"  - Robot mask shape: {len(vec_state.robot_mask)}")
    print(f"  - Job embeddings shape: {len(vec_state.job_embeddings)} x {len(vec_state.job_embeddings[0])}")
    print(f"  - Time embedding shape: {len(vec_state.time_embedding)}")
    
    # 验证 mask
    assert vec_state.robot_mask[0] == 1, "First robot should have mask=1"
    assert vec_state.robot_mask[-1] == 0, "Last robot (padding) should have mask=0"
    print("✓ Robot mask correctly applied")
    
    # 向量化轨迹（多步）
    trajectory = [obs] * 4  # K=4 步
    state_vectors = vectorizer.vectorize_trajectory(trajectory)
    print(f"✓ Vectorized trajectory: shape {state_vectors.shape}")
    assert state_vectors.shape[0] == 4, f"Expected 4 steps, got {state_vectors.shape[0]}"


def test_action_vectorizer():
    """测试动作向量化"""
    print("\n=== Test 3: Action Vectorizer ===")
    
    action_vec = ActionVectorizer()
    
    actions = [
        RobotAction(robot_id="robot_0", action_type="assign_job", assign_job_id="job_0"),
        RobotAction(robot_id="robot_1", action_type="idle"),
    ]
    
    available_jobs = ["job_0", "job_1"]
    
    # 动作转目标索引
    targets = action_vec.actions_to_targets(actions, available_jobs)
    print(f"✓ Action targets: {targets}")
    assert targets[0] == 0, "robot_0 should have target=0 (job_0)"
    assert targets[1] == 2, "robot_1 should have target=2 (idle)"
    
    # 生成虚拟 logits 并转换回动作
    logits = np.array([
        [1.0, 0.5, 0.2],  # robot_0: job_0
        [0.1, 0.3, 0.9],  # robot_1: idle
    ])
    
    recovered_actions = action_vec.logits_to_actions(
        logits,
        ["robot_0", "robot_1"],
        available_jobs,
    )
    
    print("✓ Recovered actions from logits:")
    for act in recovered_actions:
        print(f"  - {act.robot_id}: {act.action_type} (job={act.assign_job_id})")


def test_model_forward():
    """测试模型前向传播"""
    print("\n=== Test 4: Model Forward Pass ===")
    
    batch_size = 2
    seq_len = 4
    state_vec_dim = 256 + 256 + 256 + 256  # 简化
    max_robots = 10
    max_actions = 51
    
    model = DecisionTransformer(
        state_vec_dim=state_vec_dim,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        max_robots=max_robots,
        max_actions=max_actions,
    )
    
    # 虚拟输入
    state_seq = torch.randn(batch_size, seq_len, state_vec_dim)
    robot_mask = torch.ones(batch_size, max_robots)
    robot_mask[:, 5:] = 0  # 只有前 5 个机器人有效
    
    # 前向传播
    logits = model(state_seq, robot_mask)
    
    print(f"✓ Model forward pass successful")
    print(f"  - Input shape: {state_seq.shape}")
    print(f"  - Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, max_robots, max_actions), "Logits shape mismatch"
    
    # 测试采样
    actions = model.sample_action(logits)
    print(f"✓ Sampled actions: {actions.shape}")
    assert actions.shape == (batch_size, max_robots), "Actions shape mismatch"
    
    # 测试损失
    targets = torch.randint(0, max_actions, (batch_size, max_robots))
    loss_fn = MADTLoss()
    loss, metrics = loss_fn(logits, targets, robot_mask)
    print(f"✓ Loss computation: loss={loss.item():.4f}, accuracy={metrics['accuracy']:.4f}")


def test_api_end_to_end():
    """测试 API 端到端"""
    print("\n=== Test 5: API End-to-End ===")
    
    # 创建测试请求
    robots = [
        RobotState(
            robot_id=f"robot_{i}",
            position={"x": float(i * 10), "y": float(i * 5)},
            status=RobotStatus.IDLE,
            battery_level=80.0,
        )
        for i in range(3)
    ]
    
    jobs = [
        JobSpec(
            job_id=f"job_{i}",
            job_type="assembly",
            source_station_id="station_0",
            target_station_id="station_1",
            deadline=float(100 + i * 10),
            priority=50 + i * 10,
        )
        for i in range(5)
    ]
    
    stations = [
        StationState(
            station_id=f"station_{i}",
            station_type=StationType.ASSEMBLY,
            position={"x": float(i * 20), "y": 0.0},
        )
        for i in range(2)
    ]
    
    # K 步轨迹
    trajectory = [
        StepObservation(
            t=t,
            robots=robots,
            jobs=jobs,
            stations=stations,
            global_time=float(t),
        )
        for t in range(4)
    ]
    
    request = PolicyActRequest(
        trajectory=trajectory,
        return_logits=True,
    )
    
    print("✓ Created PolicyActRequest")
    print(f"  - Trajectory length: {len(request.trajectory)}")
    print(f"  - Robots: {len(request.trajectory[0].robots)}")
    print(f"  - Jobs: {len(request.trajectory[0].jobs)}")
    
    # 初始化策略服务
    config = PolicyServiceConfig(
        checkpoint_path="./checkpoints/best_model.pt",
        device="cpu",
        version="v1.0-test",
    )
    
    service = PolicyService(config)
    print("✓ Initialized PolicyService")
    
    # 推理
    response = service.act(request)
    print("✓ Policy inference successful")
    print(f"  - Number of actions: {len(response.actions)}")
    print(f"  - Actions:")
    for action in response.actions:
        print(f"    - {action.robot_id}: {action.action_type} (job={action.assign_job_id})")
    
    if response.action_distributions:
        print(f"  - Action distributions (first robot):")
        dist = response.action_distributions[0]
        if dist.logits:
            top_jobs = sorted(dist.logits.items(), key=lambda x: x[1], reverse=True)[:3]
            for job, logit in top_jobs:
                print(f"    - {job}: {logit:.4f}")


def test_heuristic_baseline():
    """测试启发式基准（最早截止时间/最近距离）"""
    print("\n=== Test 6: Heuristic Baseline ===")
    
    robots = [
        RobotState(robot_id="robot_0", position={"x": 0.0, "y": 0.0}),
        RobotState(robot_id="robot_1", position={"x": 10.0, "y": 10.0}),
    ]
    
    jobs = [
        JobSpec(
            job_id="job_0",
            job_type="assembly",
            source_station_id="s0",
            target_station_id="s1",
            deadline=50.0,
            priority=100,
        ),
        JobSpec(
            job_id="job_1",
            job_type="assembly",
            source_station_id="s0",
            target_station_id="s2",
            deadline=100.0,
            priority=50,
        ),
    ]
    
    # 最早截止时间启发式 (EDF)
    sorted_jobs = sorted(jobs, key=lambda j: j.deadline)
    print("✓ Earliest Deadline First (EDF):")
    for job in sorted_jobs:
        print(f"  - {job.job_id}: deadline={job.deadline}")
    
    # 最近距离启发式
    def distance(r1_pos, j_source):
        return ((r1_pos["x"] - j_source["x"]) ** 2 + (r1_pos["y"] - j_source["y"]) ** 2) ** 0.5
    
    print("✓ Nearest Distance assignment:")
    for robot in robots:
        # 假设工作站位置
        station_pos = {"x": 5.0, "y": 5.0}
        dist = distance(robot.position, station_pos)
        print(f"  - {robot.robot_id}: distance to station={dist:.2f}")


# ============================================================
# 主测试函数
# ============================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("MADT Policy Service - Unit Tests")
    print("="*60)
    
    try:
        # 1. Schema 测试
        obs = test_schemas()
        
        # 2. 向量化器测试
        test_vectorizer(obs)
        
        # 3. 动作向量化测试
        test_action_vectorizer()
        
        # 4. 模型前向传播测试
        test_model_forward()
        
        # 5. API 端到端测试
        test_api_end_to_end()
        
        # 6. Baseline 测试
        test_heuristic_baseline()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
