"""
强化学习调度器 - 用于动态车间调度问题
Reinforcement Learning Scheduler for Dynamic Job Shop Scheduling

支持的算法：
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Advantage Actor-Critic)
- 自定义算法

作者：Shannon 团队
日期：2026-01-29
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time


# ============================================================
# 状态空间定义
# ============================================================

@dataclass
class JobShopState:
    """车间调度状态"""
    # 零件队列状态
    parts_queue: List[Dict[str, Any]] = field(default_factory=list)
    
    # 机床状态
    machines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # AGV 状态
    agvs: List[Dict[str, Any]] = field(default_factory=list)
    
    # 机器人状态
    robots: List[Dict[str, Any]] = field(default_factory=list)
    
    # 当前扰动
    active_disturbances: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间信息
    current_time: float = 0.0
    
    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """将状态转换为向量表示（用于神经网络输入）"""
        features = []
        
        # 零件队列特征 (假设最多处理10个零件)
        max_parts = 10
        for i in range(max_parts):
            if i < len(self.parts_queue):
                part = self.parts_queue[i]
                features.extend([
                    part.get('priority_score', 0) / 100.0,  # 归一化
                    len(part.get('process', [])) / 5.0,     # 工序数
                    1.0 if part.get('urgent', False) else 0.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # 机床特征
        for machine_id in ['cnc_1', 'cnc_2', 'cnc_3']:
            if machine_id in self.machines:
                m = self.machines[machine_id]
                features.extend([
                    1.0 if m.get('status') == 'operational' else 0.0,
                    m.get('utilization', 0.0),
                    m.get('queue_length', 0) / 10.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # AGV 特征
        max_agvs = 5
        for i in range(max_agvs):
            if i < len(self.agvs):
                agv = self.agvs[i]
                features.extend([
                    1.0 if agv.get('status') == 'idle' else 0.0,
                    agv.get('battery_level', 100.0) / 100.0
                ])
            else:
                features.extend([0.0, 0.0])
        
        # 扰动特征
        features.append(len(self.active_disturbances) / 10.0)
        features.append(1.0 if any(d.get('severity') == 'high' for d in self.active_disturbances) else 0.0)
        
        # 时间特征
        features.append(self.current_time / 1000.0)  # 归一化时间
        
        return np.array(features, dtype=np.float32)
    
    def get_state_dim(self) -> int:
        """获取状态向量维度"""
        return len(self.to_vector())


# ============================================================
# 动作空间定义
# ============================================================

class ActionType(Enum):
    """调度动作类型"""
    ASSIGN_PART_TO_MACHINE = "assign_part_to_machine"
    DISPATCH_AGV = "dispatch_agv"
    ASSIGN_ROBOT_TASK = "assign_robot_task"
    ADJUST_PRIORITY = "adjust_priority"
    ACTIVATE_BACKUP_RESOURCE = "activate_backup_resource"
    NO_ACTION = "no_action"


@dataclass
class SchedulingAction:
    """调度动作"""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_index(self, action_space_size: int) -> int:
        """将动作转换为离散索引（用于DQN）"""
        # 简化版本：将动作类型和参数编码为索引
        base_idx = list(ActionType).index(self.action_type) * 100
        
        if self.action_type == ActionType.ASSIGN_PART_TO_MACHINE:
            part_idx = self.parameters.get('part_idx', 0)
            machine_idx = self.parameters.get('machine_idx', 0)
            return base_idx + part_idx * 3 + machine_idx
        
        return base_idx


# ============================================================
# 奖励函数
# ============================================================

class RewardCalculator:
    """奖励函数计算器"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # 默认权重
        self.weights = weights or {
            'completion_time': -1.0,      # 完工时间（负奖励）
            'utilization': 2.0,            # 设备利用率（正奖励）
            'tardiness': -5.0,             # 延期惩罚
            'disturbance_handling': 3.0,   # 扰动处理奖励
            'energy_efficiency': 1.0,      # 能源效率
            'quality': 2.0                 # 质量指标
        }
    
    def calculate(self, 
                  prev_state: JobShopState, 
                  action: SchedulingAction, 
                  next_state: JobShopState) -> float:
        """计算奖励值"""
        reward = 0.0
        
        # 1. 完工时间改进
        prev_completion = prev_state.metrics.get('total_completion_time', 0)
        next_completion = next_state.metrics.get('total_completion_time', 0)
        if next_completion > 0:
            reward += self.weights['completion_time'] * (next_completion - prev_completion) / 60.0
        
        # 2. 设备利用率提升
        prev_util = prev_state.metrics.get('average_utilization', 0)
        next_util = next_state.metrics.get('average_utilization', 0)
        reward += self.weights['utilization'] * (next_util - prev_util)
        
        # 3. 延期惩罚
        tardiness = next_state.metrics.get('tardiness', 0)
        if tardiness > 0:
            reward += self.weights['tardiness'] * (tardiness / 60.0)
        
        # 4. 扰动处理奖励
        disturbances_handled = len(prev_state.active_disturbances) - len(next_state.active_disturbances)
        if disturbances_handled > 0:
            reward += self.weights['disturbance_handling'] * disturbances_handled
        
        # 5. 能源效率
        energy_saved = prev_state.metrics.get('energy_consumption', 0) - next_state.metrics.get('energy_consumption', 0)
        if energy_saved > 0:
            reward += self.weights['energy_efficiency'] * energy_saved / 100.0
        
        # 6. 质量奖励
        quality_improvement = next_state.metrics.get('quality_score', 0) - prev_state.metrics.get('quality_score', 0)
        reward += self.weights['quality'] * quality_improvement
        
        return reward


# ============================================================
# 强化学习算法基类
# ============================================================

class RLScheduler(ABC):
    """强化学习调度器基类"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reward_calculator = RewardCalculator()
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
    
    @abstractmethod
    def select_action(self, state: JobShopState, epsilon: float = 0.0) -> SchedulingAction:
        """选择动作（需要子类实现）"""
        pass
    
    @abstractmethod
    def update(self, 
               state: JobShopState, 
               action: SchedulingAction, 
               reward: float, 
               next_state: JobShopState, 
               done: bool):
        """更新模型（需要子类实现）"""
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """加载模型"""
        pass


# ============================================================
# DQN 实现（示例）
# ============================================================

class DQNScheduler(RLScheduler):
    """DQN 调度器（Deep Q-Network）"""
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        
        # Q 网络（这里使用简单的表格方法作为演示）
        # 实际应用中应该使用深度神经网络（如 PyTorch 或 TensorFlow）
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        print(f"✓ 初始化 DQN 调度器")
        print(f"  状态维度: {state_dim}")
        print(f"  动作维度: {action_dim}")
    
    def _state_to_key(self, state: JobShopState) -> str:
        """将状态转换为字典键（简化版本）"""
        # 实际应用中应该使用神经网络而非表格
        return str(hash(state.to_vector().tobytes()))
    
    def select_action(self, state: JobShopState, epsilon: float = None) -> SchedulingAction:
        """ε-贪心策略选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        
        # ε-贪心探索
        if np.random.random() < epsilon:
            # 随机动作
            action_type = np.random.choice(list(ActionType))
            parameters = self._generate_random_parameters(state, action_type)
            return SchedulingAction(action_type, parameters)
        else:
            # 贪心动作（选择Q值最大的动作）
            return self._greedy_action(state)
    
    def _greedy_action(self, state: JobShopState) -> SchedulingAction:
        """贪心选择动作"""
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table:
            # 未见过的状态，随机动作
            action_type = ActionType.ASSIGN_PART_TO_MACHINE
            parameters = self._generate_random_parameters(state, action_type)
            return SchedulingAction(action_type, parameters)
        
        # 选择Q值最大的动作
        best_action_idx = np.argmax(self.q_table[state_key])
        return self._index_to_action(best_action_idx, state)
    
    def _generate_random_parameters(self, state: JobShopState, action_type: ActionType) -> Dict[str, Any]:
        """为动作类型生成随机参数"""
        if action_type == ActionType.ASSIGN_PART_TO_MACHINE:
            return {
                'part_idx': np.random.randint(0, max(1, len(state.parts_queue))),
                'machine_idx': np.random.randint(0, 3)
            }
        elif action_type == ActionType.DISPATCH_AGV:
            return {
                'agv_idx': np.random.randint(0, max(1, len(state.agvs))),
                'destination': np.random.choice(['cnc_1', 'cnc_2', 'cnc_3'])
            }
        else:
            return {}
    
    def _index_to_action(self, action_idx: int, state: JobShopState) -> SchedulingAction:
        """将索引转换为动作"""
        # 简化版本
        action_type_idx = action_idx // 100
        action_type = list(ActionType)[action_type_idx % len(ActionType)]
        
        parameters = self._generate_random_parameters(state, action_type)
        return SchedulingAction(action_type, parameters)
    
    def update(self, 
               state: JobShopState, 
               action: SchedulingAction, 
               reward: float, 
               next_state: JobShopState, 
               done: bool):
        """Q-learning 更新"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # 初始化 Q 表
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        # 动作索引
        action_idx = action.to_index(self.action_dim)
        
        # Q-learning 更新公式
        current_q = self.q_table[state_key][action_idx]
        max_next_q = np.max(self.q_table[next_state_key])
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * max_next_q
        
        # 更新 Q 值
        self.q_table[state_key][action_idx] += self.learning_rate * (target_q - current_q)
        
        # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.total_steps += 1
    
    def save_model(self, path: str):
        """保存模型"""
        model_data = {
            'q_table': {k: v.tolist() for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✓ 模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        self.q_table = {k: np.array(v) for k, v in model_data['q_table'].items()}
        self.epsilon = model_data['epsilon']
        self.total_steps = model_data['total_steps']
        self.episode_rewards = model_data['episode_rewards']
        
        print(f"✓ 模型已加载: {path}")
        print(f"  训练步数: {self.total_steps}")
        print(f"  当前 epsilon: {self.epsilon:.4f}")


# ============================================================
# 调度环境（Environment）
# ============================================================

class JobShopEnvironment:
    """车间调度环境"""
    
    def __init__(self, initial_parts: List[Dict], machines: Dict, agvs: List, robots: List):
        self.initial_parts = initial_parts
        self.machines = machines
        self.agvs = agvs
        self.robots = robots
        
        self.current_state = None
        self.steps = 0
        self.max_steps = 1000
    
    def reset(self) -> JobShopState:
        """重置环境"""
        self.current_state = JobShopState(
            parts_queue=self.initial_parts.copy(),
            machines=self.machines.copy(),
            agvs=self.agvs.copy(),
            robots=self.robots.copy(),
            active_disturbances=[],
            current_time=0.0,
            metrics={
                'total_completion_time': 0.0,
                'average_utilization': 0.0,
                'tardiness': 0.0,
                'quality_score': 100.0,
                'energy_consumption': 0.0
            }
        )
        self.steps = 0
        return self.current_state
    
    def step(self, action: SchedulingAction) -> Tuple[JobShopState, float, bool, Dict]:
        """执行动作，返回 (next_state, reward, done, info)"""
        prev_state = self.current_state
        
        # 执行动作，更新状态
        next_state = self._execute_action(prev_state, action)
        
        # 计算奖励
        reward_calc = RewardCalculator()
        reward = reward_calc.calculate(prev_state, action, next_state)
        
        # 判断是否结束
        self.steps += 1
        done = (len(next_state.parts_queue) == 0) or (self.steps >= self.max_steps)
        
        # 额外信息
        info = {
            'steps': self.steps,
            'parts_remaining': len(next_state.parts_queue),
            'utilization': next_state.metrics.get('average_utilization', 0)
        }
        
        self.current_state = next_state
        return next_state, reward, done, info
    
    def _execute_action(self, state: JobShopState, action: SchedulingAction) -> JobShopState:
        """执行动作并返回新状态"""
        # 复制状态
        new_state = JobShopState(
            parts_queue=state.parts_queue.copy(),
            machines=state.machines.copy(),
            agvs=state.agvs.copy(),
            robots=state.robots.copy(),
            active_disturbances=state.active_disturbances.copy(),
            current_time=state.current_time + 1.0,
            metrics=state.metrics.copy()
        )
        
        # 根据动作类型更新状态
        if action.action_type == ActionType.ASSIGN_PART_TO_MACHINE:
            part_idx = action.parameters.get('part_idx', 0)
            if part_idx < len(new_state.parts_queue):
                # 分配零件到机床
                new_state.parts_queue.pop(part_idx)
                new_state.metrics['average_utilization'] = min(1.0, new_state.metrics.get('average_utilization', 0) + 0.05)
        
        elif action.action_type == ActionType.DISPATCH_AGV:
            # 调度 AGV
            new_state.metrics['average_utilization'] = min(1.0, new_state.metrics.get('average_utilization', 0) + 0.02)
        
        # 更新完工时间
        if len(new_state.parts_queue) < len(state.parts_queue):
            new_state.metrics['total_completion_time'] = new_state.current_time
        
        return new_state


# ============================================================
# 训练循环
# ============================================================

def train_rl_scheduler(
    scheduler: RLScheduler,
    env: JobShopEnvironment,
    num_episodes: int = 1000,
    save_interval: int = 100
):
    """训练强化学习调度器"""
    
    print("\n" + "=" * 60)
    print("开始训练强化学习调度器")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        done = False
        while not done:
            # 选择动作
            action = scheduler.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新模型
            scheduler.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # 记录统计
        scheduler.episode_rewards.append(episode_reward)
        scheduler.episode_lengths.append(episode_length)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(scheduler.episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {scheduler.epsilon:.4f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            scheduler.save_model(f'rl_scheduler_episode_{episode + 1}.json')
    
    print("\n✓ 训练完成！")
    print(f"  总步数: {scheduler.total_steps}")
    print(f"  平均回报: {np.mean(scheduler.episode_rewards[-100:]):.2f}")


# ============================================================
# 主程序示例
# ============================================================

if __name__ == '__main__':
    print("强化学习调度器 - 示例程序")
    print("=" * 60)
    
    # 初始化环境
    initial_parts = [
        {"id": "PART-001", "priority_score": 92, "process": ["铣削", "钻孔"]},
        {"id": "PART-002", "priority_score": 85, "process": ["粗铣", "精铣"]},
        {"id": "PART-003", "priority_score": 78, "process": ["钻孔", "攻丝"]},
    ]
    
    machines = {
        "cnc_1": {"status": "operational", "utilization": 0.5},
        "cnc_2": {"status": "operational", "utilization": 0.6},
        "cnc_3": {"status": "operational", "utilization": 0.4},
    }
    
    agvs = [
        {"id": "AGV-01", "status": "idle", "battery_level": 100.0},
        {"id": "AGV-02", "status": "idle", "battery_level": 90.0},
    ]
    
    robots = [
        {"id": "ROBOT-01"},
        {"id": "ROBOT-02"},
    ]
    
    # 创建环境
    env = JobShopEnvironment(initial_parts, machines, agvs, robots)
    
    # 创建调度器
    state_dim = env.reset().get_state_dim()
    action_dim = 600  # 动作空间大小
    
    scheduler = DQNScheduler(state_dim, action_dim, learning_rate=0.001, gamma=0.95)
    
    # 训练（演示）
    print("\n开始训练演示（10个回合）...")
    train_rl_scheduler(scheduler, env, num_episodes=10, save_interval=5)
    
    print("\n✓ 示例程序完成！")
