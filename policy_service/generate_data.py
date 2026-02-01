"""
生成合成训练数据的脚本
用于演示和测试目的
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from common.schemas import (
    StepObservation, RobotState, JobSpec, StationState,
    RobotAction, TrajectoryStep, Episode,
    RobotStatus, StationType,
)


def generate_synthetic_episode(episode_id: str, steps: int = 50) -> Episode:
    """生成合成 Episode"""
    
    trajectory_steps = []
    
    # 固定资源配置
    num_robots = random.randint(3, 7)
    num_jobs = random.randint(10, 30)
    num_stations = random.randint(3, 6)
    
    robot_ids = [f"robot_{i}" for i in range(num_robots)]
    job_ids = [f"job_{i}" for i in range(num_jobs)]
    station_ids = [f"station_{i}" for i in range(num_stations)]
    
    total_reward = 0.0
    
    for t in range(steps):
        # 生成观测
        robots = [
            RobotState(
                robot_id=rid,
                position={
                    "x": random.uniform(0, 100),
                    "y": random.uniform(0, 100)
                },
                status=random.choice(list(RobotStatus)),
                battery_level=random.uniform(20, 100),
                load_capacity=random.uniform(0, 50),
            )
            for rid in robot_ids
        ]
        
        # 动态生成待完成的任务
        num_pending = random.randint(5, min(15, len(job_ids)))
        pending_jobs = random.sample(job_ids, num_pending)
        
        jobs = [
            JobSpec(
                job_id=jid,
                job_type=random.choice(["assembly", "packaging", "quality_check"]),
                source_station_id=random.choice(station_ids),
                target_station_id=random.choice(station_ids),
                deadline=float(t + random.randint(10, 100)),
                priority=random.randint(10, 100),
            )
            for jid in pending_jobs
        ]
        
        stations = [
            StationState(
                station_id=sid,
                station_type=random.choice(list(StationType)),
                position={
                    "x": float(i * 30),
                    "y": float(i % 3 * 30),
                },
                is_available=random.random() > 0.1,
                queued_jobs=random.sample(job_ids, random.randint(0, 3)),
            )
            for i, sid in enumerate(station_ids)
        ]
        
        observation = StepObservation(
            t=t,
            robots=robots,
            jobs=jobs,
            stations=stations,
            global_time=float(t),
        )
        
        # 生成"最优"动作（基于启发式）
        actions = []
        
        # 最早截止时间优先
        sorted_jobs = sorted(jobs, key=lambda j: j.deadline) if jobs else []
        
        for i, robot in enumerate(robots):
            if sorted_jobs and random.random() > 0.2:  # 80% 概率分配任务
                job = sorted_jobs[i % len(sorted_jobs)]
                action = RobotAction(
                    robot_id=robot.robot_id,
                    action_type="assign_job",
                    assign_job_id=job.job_id,
                )
            else:
                action = RobotAction(
                    robot_id=robot.robot_id,
                    action_type="idle",
                )
            actions.append(action)
        
        # 奖励函数（简化）
        reward = 0.0
        # 有活跃机器人得分
        for action in actions:
            if action.action_type == "assign_job":
                reward += 0.5
        # 任务少了得分
        reward += max(0, (10 - len(jobs)) / 10.0)
        
        total_reward += reward
        
        # 创建训练步
        step = TrajectoryStep(
            obs=observation,
            action=actions,
            reward=reward,
            done=t == steps - 1,
            info={"assigned_jobs": len([a for a in actions if a.action_type == "assign_job"])},
        )
        
        trajectory_steps.append(step)
    
    episode = Episode(
        episode_id=episode_id,
        steps=trajectory_steps,
        total_reward=total_reward,
        metadata={
            "generated_at": datetime.now().isoformat(),
            "num_robots": num_robots,
            "num_jobs": num_jobs,
            "num_stations": num_stations,
        },
    )
    
    return episode


def save_episodes(episodes: List[Episode], output_dir: str, format: str = "jsonl"):
    """保存 Episode 到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        file_path = output_path / "episodes.jsonl"
        with open(file_path, 'w') as f:
            for episode in episodes:
                # 手动序列化（因为 Pydantic 的复杂嵌套结构）
                episode_dict = {
                    "episode_id": episode.episode_id,
                    "steps": [
                        {
                            "obs": step.obs.model_dump(),
                            "action": [a.model_dump() for a in step.action],
                            "reward": step.reward,
                            "done": step.done,
                            "info": step.info,
                        }
                        for step in episode.steps
                    ],
                    "total_reward": episode.total_reward,
                    "metadata": episode.metadata,
                }
                f.write(json.dumps(episode_dict, default=str) + "\n")
        
        print(f"✓ Saved {len(episodes)} episodes to {file_path}")
        print(f"  File size: {file_path.stat().st_size / 1024:.1f} KB")
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    """生成合成数据"""
    import sys
    
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/episodes"
    
    print(f"Generating {num_episodes} synthetic episodes...")
    
    episodes = []
    for i in range(num_episodes):
        episode = generate_synthetic_episode(
            episode_id=f"episode_{i:04d}",
            steps=random.randint(30, 100),
        )
        episodes.append(episode)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_episodes} episodes")
    
    save_episodes(episodes, output_dir, format="jsonl")
    
    print(f"\n✓ Generated {num_episodes} episodes in {output_dir}/episodes.jsonl")
    print(f"  Use for training: python -m training.train --config configs/v1_bc.yaml")


if __name__ == '__main__':
    main()
