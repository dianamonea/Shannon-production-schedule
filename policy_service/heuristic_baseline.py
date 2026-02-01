"""
启发式策略生成器（Baseline）
用于生成初始训练数据
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random


class HeuristicDispatcher:
    """
    启发式调度器
    支持多种策略：
    - earliest_deadline: 优先分配最早截止时间的任务
    - nearest_distance: 优先分配距离最近的任务
    - min_completion_time: 优先分配预计完成时间最短的任务
    """
    
    def __init__(self, strategy: str = "earliest_deadline"):
        """
        Args:
            strategy: 启发式策略名称
        """
        self.strategy = strategy
        self.strategies = {
            'earliest_deadline': self._earliest_deadline,
            'nearest_distance': self._nearest_distance,
            'min_completion_time': self._min_completion_time,
        }
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
    
    def dispatch(self, state: Dict) -> List[Dict]:
        """
        根据当前状态生成调度动作
        
        Args:
            state: {
                "robots": [{"robot_id": ..., "position": {...}, "status": ..., ...}, ...],
                "jobs": [{"job_id": ..., "source_station_id": ..., "target_station_id": ..., "deadline": ..., ...}, ...],
                "stations": [{"station_id": ..., "position": {...}, ...}, ...],
            }
        
        Returns:
            actions: [{"robot_id": ..., "action_type": ..., "assign_job_id": ...}, ...]
        """
        return self.strategies[self.strategy](state)
    
    def _earliest_deadline(self, state: Dict) -> List[Dict]:
        """最早截止时间优先"""
        robots = state['robots']
        jobs = state['jobs']
        stations = state['stations']
        
        # 创建 station_id -> position 映射
        station_pos = {s['station_id']: s['position'] for s in stations}
        
        # 过滤可用机器人和任务
        available_robots = [r for r in robots if r['status'] == 'idle']
        pending_jobs = [j for j in jobs if j.get('status', 'pending') == 'pending']
        
        # 按 deadline 排序任务
        pending_jobs = sorted(pending_jobs, key=lambda j: j.get('deadline', float('inf')))
        
        actions = []
        assigned_jobs = set()
        
        # 贪心分配
        for robot in available_robots:
            best_job = None
            for job in pending_jobs:
                if job['job_id'] not in assigned_jobs:
                    best_job = job
                    break
            
            if best_job:
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'assign_job',
                    'assign_job_id': best_job['job_id'],
                })
                assigned_jobs.add(best_job['job_id'])
            else:
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'idle',
                    'assign_job_id': None,
                })
        
        # 已占用的机器人保持当前任务
        for robot in robots:
            if robot['status'] != 'idle':
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'working',
                    'assign_job_id': robot.get('current_job_id'),
                })
        
        return actions
    
    def _nearest_distance(self, state: Dict) -> List[Dict]:
        """最近距离优先"""
        robots = state['robots']
        jobs = state['jobs']
        stations = state['stations']
        
        station_pos = {s['station_id']: s['position'] for s in stations}
        
        available_robots = [r for r in robots if r['status'] == 'idle']
        pending_jobs = [j for j in jobs if j.get('status', 'pending') == 'pending']
        
        actions = []
        assigned_jobs = set()
        
        for robot in available_robots:
            robot_pos = robot['position']
            
            # 计算到每个任务起点的距离
            best_job = None
            min_dist = float('inf')
            
            for job in pending_jobs:
                if job['job_id'] in assigned_jobs:
                    continue
                
                source_pos = station_pos.get(job['source_station_id'], {'x': 0, 'y': 0})
                dist = np.sqrt(
                    (robot_pos['x'] - source_pos['x'])**2 +
                    (robot_pos['y'] - source_pos['y'])**2
                )
                
                if dist < min_dist:
                    min_dist = dist
                    best_job = job
            
            if best_job:
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'assign_job',
                    'assign_job_id': best_job['job_id'],
                })
                assigned_jobs.add(best_job['job_id'])
            else:
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'idle',
                    'assign_job_id': None,
                })
        
        for robot in robots:
            if robot['status'] != 'idle':
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'working',
                    'assign_job_id': robot.get('current_job_id'),
                })
        
        return actions
    
    def _min_completion_time(self, state: Dict) -> List[Dict]:
        """最小预计完成时间优先"""
        robots = state['robots']
        jobs = state['jobs']
        stations = state['stations']
        
        station_pos = {s['station_id']: s['position'] for s in stations}
        
        available_robots = [r for r in robots if r['status'] == 'idle']
        pending_jobs = [j for j in jobs if j.get('status', 'pending') == 'pending']
        
        actions = []
        assigned_jobs = set()
        
        for robot in available_robots:
            robot_pos = robot['position']
            robot_speed = 1.0  # 假设恒定速度
            
            best_job = None
            min_time = float('inf')
            
            for job in pending_jobs:
                if job['job_id'] in assigned_jobs:
                    continue
                
                source_pos = station_pos.get(job['source_station_id'], {'x': 0, 'y': 0})
                target_pos = station_pos.get(job['target_station_id'], {'x': 0, 'y': 0})
                
                # 预计完成时间 = 到起点距离 + 起点到终点距离
                dist_to_source = np.sqrt(
                    (robot_pos['x'] - source_pos['x'])**2 +
                    (robot_pos['y'] - source_pos['y'])**2
                )
                dist_source_to_target = np.sqrt(
                    (source_pos['x'] - target_pos['x'])**2 +
                    (source_pos['y'] - target_pos['y'])**2
                )
                
                completion_time = (dist_to_source + dist_source_to_target) / robot_speed
                
                if completion_time < min_time:
                    min_time = completion_time
                    best_job = job
            
            if best_job:
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'assign_job',
                    'assign_job_id': best_job['job_id'],
                })
                assigned_jobs.add(best_job['job_id'])
            else:
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'idle',
                    'assign_job_id': None,
                })
        
        for robot in robots:
            if robot['status'] != 'idle':
                actions.append({
                    'robot_id': robot['robot_id'],
                    'action_type': 'working',
                    'assign_job_id': robot.get('current_job_id'),
                })
        
        return actions


def generate_baseline_data(
    num_episodes: int,
    num_steps_per_episode: int,
    num_robots: int = 5,
    num_jobs: int = 20,
    num_stations: int = 10,
    strategy: str = "earliest_deadline",
    output_dir: str = "./data/baseline",
    seed: int = 42,
):
    """
    生成启发式基线数据
    
    Args:
        num_episodes: 生成 episode 数量
        num_steps_per_episode: 每个 episode 的步数
        num_robots: 机器人数量
        num_jobs: 任务数量
        num_stations: 工作站数量
        strategy: 启发式策略
        output_dir: 输出目录
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dispatcher = HeuristicDispatcher(strategy=strategy)
    
    for ep in range(num_episodes):
        episode_steps = []
        
        # 初始化状态（简化版）
        robots = [
            {
                'robot_id': f"r{i}",
                'position': {'x': random.uniform(0, 100), 'y': random.uniform(0, 100)},
                'status': 'idle',
                'current_job_id': None,
                'battery_level': 100.0,
                'load_capacity': 0.0,
            }
            for i in range(num_robots)
        ]
        
        stations = [
            {
                'station_id': f"s{i}",
                'station_type': random.choice(['assembly', 'quality_check', 'packaging', 'storage']),
                'position': {'x': random.uniform(0, 100), 'y': random.uniform(0, 100)},
                'is_available': True,
                'queued_jobs': [],
            }
            for i in range(num_stations)
        ]
        
        jobs = [
            {
                'job_id': f"j{i}",
                'job_type': 'transport',
                'source_station_id': f"s{random.randint(0, num_stations-1)}",
                'target_station_id': f"s{random.randint(0, num_stations-1)}",
                'deadline': random.uniform(10, 100),
                'priority': random.randint(0, 100),
                'required_capacity': 0.0,
                'status': 'pending',
            }
            for i in range(num_jobs)
        ]
        
        for t in range(num_steps_per_episode):
            # 当前状态
            obs = {
                't': t,
                'robots': robots,
                'jobs': jobs,
                'stations': stations,
                'global_time': float(t),
            }
            
            # 启发式动作
            actions = dispatcher.dispatch(obs)
            
            # 简单奖励（完成任务数）
            reward = sum(1 for j in jobs if j['status'] == 'completed')
            
            # 记录 step
            step = {
                'obs': obs,
                'action': actions,
                'reward': reward,
                'done': t == num_steps_per_episode - 1,
                'info': {},
            }
            episode_steps.append(step)
            
            # 简单状态转移（占位符，实际应有完整仿真）
            # ...
        
        # 保存 episode
        episode_file = output_path / f"episode_{ep}.jsonl"
        with open(episode_file, 'w') as f:
            for step in episode_steps:
                f.write(json.dumps(step) + '\n')
        
        print(f"Generated episode {ep+1}/{num_episodes}")
    
    print(f"Baseline data saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate baseline heuristic data')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--strategy', type=str, default='earliest_deadline',
                        choices=['earliest_deadline', 'nearest_distance', 'min_completion_time'])
    parser.add_argument('--output_dir', type=str, default='./data/baseline')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    generate_baseline_data(
        num_episodes=args.num_episodes,
        num_steps_per_episode=args.num_steps,
        strategy=args.strategy,
        output_dir=args.output_dir,
        seed=args.seed,
    )
