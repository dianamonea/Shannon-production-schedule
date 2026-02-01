# 强化学习集成指南

## 概述

本指南帮助你将自己的强化学习程序集成到 Shannon 生产调度系统中，用于求解动态车间调度问题（Dynamic Job Shop Scheduling Problem, DJSSP）。

## 目录

1. [快速开始](#快速开始)
2. [架构设计](#架构设计)
3. [集成步骤](#集成步骤)
4. [状态空间设计](#状态空间设计)
5. [动作空间设计](#动作空间设计)
6. [奖励函数设计](#奖励函数设计)
7. [使用自定义算法](#使用自定义算法)
8. [训练流程](#训练流程)
9. [推理部署](#推理部署)
10. [性能优化](#性能优化)

---

## 快速开始

### 1. 运行示例程序

```powershell
# 运行基础 DQN 示例
python rl_scheduler.py
```

### 2. 查看输出

```
强化学习调度器 - 示例程序
============================================================
✓ 初始化 DQN 调度器
  状态维度: 53
  动作维度: 600

开始训练强化学习调度器
============================================================
Episode 10/10 | Avg Reward: -12.45 | Epsilon: 0.9510
✓ 模型已保存到: rl_scheduler_episode_5.json
Episode 10/10 | Avg Reward: -8.32 | Epsilon: 0.9044
✓ 模型已保存到: rl_scheduler_episode_10.json

✓ 训练完成！
  总步数: 1543
  平均回报: -8.32
```

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Shannon 调度系统                      │
├─────────────────────────────────────────────────────────┤
│  ProductionSchedulingAgent                              │
│  ├── MachineToolAgent                                   │
│  ├── AGVCoordinator                                     │
│  └── RobotCellAgent                                     │
└───────────────┬─────────────────────────────────────────┘
                │
                │ 调用接口
                ▼
┌─────────────────────────────────────────────────────────┐
│              强化学习调度器（RL Scheduler）              │
├─────────────────────────────────────────────────────────┤
│  RLScheduler (基类)                                     │
│  ├── select_action()      # 策略选择                    │
│  ├── update()             # 模型更新                    │
│  ├── save_model()         # 保存模型                    │
│  └── load_model()         # 加载模型                    │
└───────────────┬─────────────────────────────────────────┘
                │
                │ 继承实现
                ▼
┌─────────────────────────────────────────────────────────┐
│                  具体算法实现                            │
├─────────────────────────────────────────────────────────┤
│  DQNScheduler          # Deep Q-Network                 │
│  PPOScheduler          # Proximal Policy Optimization   │
│  A3CScheduler          # A3C                            │
│  YourCustomScheduler   # 你的自定义算法                 │
└─────────────────────────────────────────────────────────┘
```

### 数据流

```
状态（State） → [RL模型] → 动作（Action） → [环境执行] → 奖励（Reward）
     ▲                                                        │
     └────────────────────────────────────────────────────────┘
                         反馈循环
```

---

## 集成步骤

### 方法 1: 扩展现有智能体

修改 `production_scheduler_demo.py` 中的智能体类：

```python
from rl_scheduler import RLScheduler, DQNScheduler, JobShopState

class MachineToolAgent(ProductionSchedulingAgent):
    def __init__(self, agent_id, name, machine_ids):
        super().__init__(agent_id, name)
        self.machine_ids = machine_ids
        
        # 集成 RL 调度器
        state_dim = 53  # 根据实际情况调整
        action_dim = 600
        self.rl_scheduler = DQNScheduler(state_dim, action_dim)
        
        # 加载预训练模型（可选）
        try:
            self.rl_scheduler.load_model('pretrained_scheduler.json')
        except:
            print(f"{self.name}: 使用随机初始化的模型")
    
    def schedule_parts(self, parts, current_time):
        """使用 RL 进行调度决策"""
        # 1. 构建当前状态
        current_state = self._build_state(parts, current_time)
        
        # 2. RL 选择动作（推理模式，epsilon=0）
        action = self.rl_scheduler.select_action(current_state, epsilon=0.0)
        
        # 3. 执行动作
        schedule = self._execute_action(action, parts)
        
        return schedule
    
    def _build_state(self, parts, current_time):
        """构建 JobShopState"""
        return JobShopState(
            parts_queue=[{
                'id': p['part_id'],
                'priority_score': p.get('priority', 50),
                'process': p.get('process', []),
                'urgent': p.get('urgent', False)
            } for p in parts],
            machines={
                mid: {
                    'status': 'operational',
                    'utilization': 0.7,
                    'queue_length': 2
                } for mid in self.machine_ids
            },
            current_time=current_time,
            metrics=self.state.get('metrics', {})
        )
    
    def _execute_action(self, action, parts):
        """将 RL 动作转换为实际调度计划"""
        schedule = []
        
        if action.action_type.value == "assign_part_to_machine":
            part_idx = action.parameters.get('part_idx', 0)
            machine_idx = action.parameters.get('machine_idx', 0)
            
            if part_idx < len(parts):
                selected_part = parts[part_idx]
                selected_machine = self.machine_ids[machine_idx % len(self.machine_ids)]
                
                schedule.append({
                    'part_id': selected_part['part_id'],
                    'machine': selected_machine,
                    'start_time': selected_part.get('arrival_time', 0),
                    'process': selected_part.get('process', [])
                })
        
        return schedule
```

### 方法 2: 创建独立的 RL 调度服务

创建新文件 `rl_scheduling_service.py`:

```python
from rl_scheduler import RLScheduler, DQNScheduler, JobShopEnvironment
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# 全局调度器
scheduler = None

@app.route('/init', methods=['POST'])
def init_scheduler():
    """初始化调度器"""
    global scheduler
    
    data = request.json
    state_dim = data.get('state_dim', 53)
    action_dim = data.get('action_dim', 600)
    
    scheduler = DQNScheduler(state_dim, action_dim)
    
    # 加载模型
    model_path = data.get('model_path')
    if model_path:
        scheduler.load_model(model_path)
    
    return jsonify({'status': 'success', 'message': '调度器已初始化'})

@app.route('/select_action', methods=['POST'])
def select_action():
    """选择动作"""
    global scheduler
    
    if scheduler is None:
        return jsonify({'error': '调度器未初始化'}), 400
    
    # 解析状态
    state_data = request.json
    state = JobShopState(**state_data)
    
    # 选择动作
    action = scheduler.select_action(state, epsilon=0.0)
    
    return jsonify({
        'action_type': action.action_type.value,
        'parameters': action.parameters
    })

@app.route('/update', methods=['POST'])
def update_model():
    """更新模型（在线学习）"""
    global scheduler
    
    data = request.json
    state = JobShopState(**data['state'])
    action = SchedulingAction(**data['action'])
    reward = data['reward']
    next_state = JobShopState(**data['next_state'])
    done = data['done']
    
    scheduler.update(state, action, reward, next_state, done)
    
    return jsonify({'status': 'updated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

然后在主程序中调用：

```python
import requests

# 初始化 RL 服务
requests.post('http://localhost:5001/init', json={
    'state_dim': 53,
    'action_dim': 600,
    'model_path': 'pretrained_scheduler.json'
})

# 调度时调用
def get_rl_action(state_dict):
    response = requests.post('http://localhost:5001/select_action', json=state_dict)
    return response.json()
```

---

## 状态空间设计

### 当前实现

`JobShopState` 包含以下信息：

```python
@dataclass
class JobShopState:
    parts_queue: List[Dict]           # 零件队列
    machines: Dict[str, Dict]          # 机床状态
    agvs: List[Dict]                   # AGV 状态
    robots: List[Dict]                 # 机器人状态
    active_disturbances: List[Dict]    # 活动扰动
    current_time: float                # 当前时间
    metrics: Dict[str, float]          # 性能指标
```

### 向量化表示

`to_vector()` 方法将状态转换为 53 维向量：

- **零件队列** (30维): 10个零件 × 3特征（优先级、工序数、紧急度）
- **机床状态** (9维): 3台机床 × 3特征（状态、利用率、队列长度）
- **AGV状态** (10维): 5辆AGV × 2特征（状态、电量）
- **扰动信息** (2维): 扰动数量、严重程度
- **时间信息** (1维): 归一化时间
- **其他** (1维): 保留

### 自定义状态空间

如果需要扩展状态空间：

```python
class CustomJobShopState(JobShopState):
    def to_vector(self) -> np.ndarray:
        """自定义状态向量"""
        features = []
        
        # 添加你自己的特征
        features.append(self.get_makespan())
        features.append(self.get_tardiness())
        features.append(self.get_flow_time())
        
        # 调用父类方法
        base_features = super().to_vector()
        features.extend(base_features)
        
        return np.array(features, dtype=np.float32)
    
    def get_makespan(self):
        """计算完工时间"""
        # 实现逻辑
        return 0.0
```

---

## 动作空间设计

### 当前动作类型

```python
class ActionType(Enum):
    ASSIGN_PART_TO_MACHINE = "assign_part_to_machine"      # 分配零件到机床
    DISPATCH_AGV = "dispatch_agv"                          # 调度AGV
    ASSIGN_ROBOT_TASK = "assign_robot_task"                # 分配机器人任务
    ADJUST_PRIORITY = "adjust_priority"                    # 调整优先级
    ACTIVATE_BACKUP_RESOURCE = "activate_backup_resource"  # 激活备用资源
    NO_ACTION = "no_action"                                # 无操作
```

### 离散动作空间

当前实现使用离散动作：

- 动作索引 = `action_type_idx * 100 + parameters_hash`
- 总动作数：600（可调整）

### 连续动作空间

如果使用 PPO/DDPG 等算法，可以定义连续动作：

```python
class ContinuousAction:
    def __init__(self, vector: np.ndarray):
        # vector shape: [10]
        self.part_selection_prob = vector[0:5]      # 零件选择概率
        self.machine_selection_prob = vector[5:8]   # 机床选择概率
        self.priority_adjustment = vector[8]        # 优先级调整 [-1, 1]
        self.urgency_threshold = vector[9]          # 紧急阈值 [0, 1]
    
    def to_discrete_action(self) -> SchedulingAction:
        """转换为离散动作"""
        part_idx = np.argmax(self.part_selection_prob)
        machine_idx = np.argmax(self.machine_selection_prob)
        
        return SchedulingAction(
            action_type=ActionType.ASSIGN_PART_TO_MACHINE,
            parameters={'part_idx': part_idx, 'machine_idx': machine_idx}
        )
```

---

## 奖励函数设计

### 当前实现

`RewardCalculator` 使用多目标加权：

```python
weights = {
    'completion_time': -1.0,       # 完工时间（越短越好）
    'utilization': 2.0,             # 设备利用率（越高越好）
    'tardiness': -5.0,              # 延期（越少越好）
    'disturbance_handling': 3.0,    # 扰动处理（越多越好）
    'energy_efficiency': 1.0,       # 能源效率（越高越好）
    'quality': 2.0                  # 质量（越高越好）
}
```

### 自定义奖励函数

```python
class CustomRewardCalculator(RewardCalculator):
    def calculate(self, prev_state, action, next_state):
        """自定义奖励计算"""
        reward = 0.0
        
        # 1. Makespan 奖励（完工时间）
        if self.is_all_jobs_completed(next_state):
            makespan = next_state.current_time
            reward += -makespan / 1000.0  # 归一化
        
        # 2. 流程时间奖励
        flow_time = self.calculate_flow_time(next_state)
        reward += -flow_time / 500.0
        
        # 3. 延期惩罚（加权）
        for job in next_state.completed_jobs:
            if job['completion_time'] > job['due_date']:
                tardiness = job['completion_time'] - job['due_date']
                reward += -tardiness * job['weight'] / 100.0
        
        # 4. 设备利用率奖励
        utilization = next_state.metrics['average_utilization']
        reward += utilization * 10.0
        
        # 5. 能耗惩罚
        energy = next_state.metrics.get('energy_consumption', 0)
        reward += -energy / 1000.0
        
        # 6. 扰动响应速度奖励
        if len(next_state.active_disturbances) < len(prev_state.active_disturbances):
            response_speed = (prev_state.current_time - next_state.current_time)
            reward += response_speed / 10.0
        
        return reward
```

### 奖励塑形（Reward Shaping）

为了加速学习，可以添加中间奖励：

```python
def shaped_reward(self, state, action, next_state):
    """奖励塑形"""
    # 基础奖励
    base_reward = self.calculate(state, action, next_state)
    
    # 潜力函数（Potential-based shaping）
    gamma = 0.99
    phi_s = self.potential(state)
    phi_s_prime = self.potential(next_state)
    
    shaping_reward = gamma * phi_s_prime - phi_s
    
    return base_reward + shaping_reward

def potential(self, state):
    """状态潜力函数"""
    # 剩余任务数（越少越好）
    remaining_jobs = len(state.parts_queue)
    
    # 平均紧急度
    avg_urgency = np.mean([p.get('priority_score', 0) for p in state.parts_queue])
    
    return -remaining_jobs * 10.0 - avg_urgency
```

---

## 使用自定义算法

### 1. 实现自己的 PPO 调度器

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    """PPO 策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class PPOScheduler(RLScheduler):
    """PPO 调度器"""
    
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        self.clip_epsilon = 0.2
        self.memory = []
    
    def select_action(self, state: JobShopState, epsilon: float = 0.0):
        """选择动作"""
        state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_vector)
        
        # 采样动作
        dist = Categorical(action_probs)
        action_idx = dist.sample().item()
        
        # 记录日志概率
        log_prob = dist.log_prob(torch.tensor(action_idx))
        
        # 转换为调度动作
        action = self._index_to_action(action_idx, state)
        
        # 存储经验
        self.memory.append({
            'state': state_vector,
            'action': action_idx,
            'log_prob': log_prob
        })
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """PPO 更新"""
        # 收集一批经验后再更新
        if done or len(self.memory) >= 128:
            self._ppo_update()
            self.memory = []
    
    def _ppo_update(self):
        """执行 PPO 更新"""
        if len(self.memory) == 0:
            return
        
        # 准备数据
        states = torch.cat([m['state'] for m in self.memory])
        actions = torch.tensor([m['action'] for m in self.memory])
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])
        
        # 计算优势函数（这里简化）
        advantages = torch.ones(len(self.memory))
        
        # PPO 更新
        for _ in range(10):  # K epochs
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO 裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # 损失函数
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), advantages)
            
            loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards
        }, path)
        print(f"✓ PPO 模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"✓ PPO 模型已加载: {path}")
```

### 2. 使用外部库（如 Stable-Baselines3）

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym import spaces

class JobShopGymEnv(gym.Env):
    """Gym 兼容的车间调度环境"""
    
    def __init__(self, initial_parts, machines, agvs, robots):
        super().__init__()
        
        self.base_env = JobShopEnvironment(initial_parts, machines, agvs, robots)
        
        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(600)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(53,), 
            dtype=np.float32
        )
    
    def reset(self):
        state = self.base_env.reset()
        return state.to_vector()
    
    def step(self, action_idx):
        # 将索引转换为 SchedulingAction
        action = self._index_to_action(action_idx)
        
        next_state, reward, done, info = self.base_env.step(action)
        
        return next_state.to_vector(), reward, done, info
    
    def render(self, mode='human'):
        pass

# 使用 Stable-Baselines3
env = JobShopGymEnv(initial_parts, machines, agvs, robots)
model = PPO("MlpPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=10000)

# 保存
model.save("ppo_job_shop")

# 加载和推理
model = PPO.load("ppo_job_shop")
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

---

## 训练流程

### 离线训练

```python
# 1. 准备训练数据
training_scenarios = [
    {
        'parts': generate_random_parts(10),
        'disturbances': generate_disturbances(5)
    }
    for _ in range(100)
]

# 2. 创建环境和调度器
env = JobShopEnvironment(...)
scheduler = DQNScheduler(state_dim=53, action_dim=600)

# 3. 训练
for scenario in training_scenarios:
    env.set_scenario(scenario)
    train_rl_scheduler(scheduler, env, num_episodes=100)

# 4. 保存模型
scheduler.save_model('trained_scheduler.json')
```

### 在线学习

```python
class OnlineLearningScheduler:
    def __init__(self, base_scheduler: RLScheduler):
        self.scheduler = base_scheduler
        self.experience_buffer = []
    
    def schedule_with_learning(self, state):
        """调度并在线学习"""
        # 1. 选择动作
        action = self.scheduler.select_action(state, epsilon=0.1)
        
        # 2. 执行并记录
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'timestamp': time.time()
        })
        
        return action
    
    def update_from_feedback(self, reward, next_state, done):
        """根据反馈更新"""
        if len(self.experience_buffer) > 0:
            exp = self.experience_buffer[-1]
            self.scheduler.update(
                exp['state'], 
                exp['action'], 
                reward, 
                next_state, 
                done
            )
```

### 迁移学习

```python
# 从预训练模型开始
scheduler = DQNScheduler(state_dim=53, action_dim=600)
scheduler.load_model('pretrained_on_small_dataset.json')

# 在新环境中微调
scheduler.epsilon = 0.3  # 增加探索
scheduler.learning_rate = 0.0001  # 降低学习率

new_env = JobShopEnvironment(new_parts, new_machines, ...)
train_rl_scheduler(scheduler, new_env, num_episodes=500)
```

---

## 推理部署

### 方法 1: 直接集成到智能体

```python
# 在 production_scheduler_demo.py 中
class MachineToolAgent(ProductionSchedulingAgent):
    def __init__(self, ...):
        # 加载训练好的模型
        self.rl_scheduler = DQNScheduler(53, 600)
        self.rl_scheduler.load_model('trained_scheduler.json')
        self.rl_scheduler.epsilon = 0.0  # 纯利用模式
```

### 方法 2: 独立服务

```bash
# 启动 RL 服务
python rl_scheduling_service.py
```

```python
# 在主程序中调用
import requests

def get_rl_decision(state_data):
    response = requests.post(
        'http://localhost:5001/select_action',
        json=state_data
    )
    return response.json()
```

### 方法 3: 模型导出（ONNX）

```python
import torch.onnx

# 导出 PyTorch 模型到 ONNX
dummy_input = torch.randn(1, 53)
torch.onnx.export(
    scheduler.policy,
    dummy_input,
    "scheduler.onnx",
    export_params=True
)

# 使用 ONNX Runtime 推理
import onnxruntime as ort

session = ort.InferenceSession("scheduler.onnx")
output = session.run(None, {'input': state_vector})
```

---

## 性能优化

### 1. 状态表示优化

```python
# 使用归一化
class NormalizedState(JobShopState):
    def to_vector(self):
        raw_vector = super().to_vector()
        
        # Z-score 归一化
        mean = np.mean(raw_vector)
        std = np.std(raw_vector) + 1e-8
        normalized = (raw_vector - mean) / std
        
        return normalized
```

### 2. 经验回放（Experience Replay）

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 在 DQNScheduler 中使用
class ImprovedDQNScheduler(DQNScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.batch_size = 64
    
    def update(self, state, action, reward, next_state, done):
        # 存储经验
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # 批量更新
        if len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            for s, a, r, ns, d in batch:
                super().update(s, a, r, ns, d)
```

### 3. 优先经验回放（PER）

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, transition):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
```

### 4. 多线程训练

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_training(num_workers=4):
    """并行训练多个智能体"""
    
    def train_worker(worker_id):
        # 创建独立的环境和调度器
        env = JobShopEnvironment(...)
        scheduler = DQNScheduler(53, 600)
        
        # 训练
        train_rl_scheduler(scheduler, env, num_episodes=250)
        
        # 保存模型
        scheduler.save_model(f'scheduler_worker_{worker_id}.json')
        
        return scheduler.episode_rewards
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(train_worker, range(num_workers)))
    
    print(f"✓ 并行训练完成，共 {num_workers} 个工作线程")
    return results
```

---

## 示例：完整集成流程

### 1. 训练阶段

```python
# train_rl_scheduler.py
from rl_scheduler import DQNScheduler, JobShopEnvironment, train_rl_scheduler

# 准备数据
training_data = {
    'parts': [...],
    'machines': {...},
    'agvs': [...],
    'robots': [...]
}

# 创建环境
env = JobShopEnvironment(**training_data)

# 创建调度器
scheduler = DQNScheduler(
    state_dim=env.reset().get_state_dim(),
    action_dim=600,
    learning_rate=0.001,
    gamma=0.95
)

# 训练
train_rl_scheduler(
    scheduler,
    env,
    num_episodes=10000,
    save_interval=500
)

# 保存最终模型
scheduler.save_model('final_scheduler.json')
```

### 2. 部署阶段

```python
# production_scheduler_demo.py（修改后）
from rl_scheduler import DQNScheduler, JobShopState

class MachineToolAgent(ProductionSchedulingAgent):
    def __init__(self, agent_id, name, machine_ids):
        super().__init__(agent_id, name)
        self.machine_ids = machine_ids
        
        # 加载训练好的 RL 模型
        self.rl_scheduler = DQNScheduler(53, 600)
        self.rl_scheduler.load_model('final_scheduler.json')
        
        print(f"{self.name}: ✓ 已加载 RL 调度模型")
    
    def schedule_parts(self, parts, current_time):
        """使用 RL 进行调度"""
        # 构建状态
        state = JobShopState(
            parts_queue=[{
                'id': p['part_id'],
                'priority_score': p.get('priority', 50),
                'process': p.get('process', [])
            } for p in parts],
            machines={mid: {'status': 'operational'} for mid in self.machine_ids},
            current_time=current_time,
            metrics=self.state.get('metrics', {})
        )
        
        # RL 决策
        action = self.rl_scheduler.select_action(state, epsilon=0.0)
        
        # 执行并返回调度计划
        return self._action_to_schedule(action, parts)
```

### 3. 运行完整演示

```powershell
# 训练模型
python train_rl_scheduler.py

# 运行生产调度演示（使用训练好的模型）
python production_scheduler_demo.py
```

---

## 常见问题

### Q1: 状态维度太大怎么办？

**A**: 使用特征工程或降维：

```python
from sklearn.decomposition import PCA

class CompressedState(JobShopState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca = PCA(n_components=20)
    
    def to_vector(self):
        raw = super().to_vector()
        compressed = self.pca.fit_transform(raw.reshape(1, -1))
        return compressed.flatten()
```

### Q2: 训练速度太慢怎么办？

**A**: 
1. 使用经验回放减少样本需求
2. 并行训练多个智能体
3. 使用 GPU 加速（PyTorch/TensorFlow）
4. 简化环境模拟

### Q3: 如何处理连续动作空间？

**A**: 使用 DDPG/SAC 等算法：

```python
from stable_baselines3 import SAC

env = JobShopGymEnv(...)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Q4: 如何评估模型性能？

**A**: 

```python
def evaluate_scheduler(scheduler, test_env, num_episodes=100):
    """评估调度器性能"""
    makespans = []
    utilizations = []
    
    for _ in range(num_episodes):
        state = test_env.reset()
        done = False
        
        while not done:
            action = scheduler.select_action(state, epsilon=0.0)
            state, _, done, info = test_env.step(action)
        
        makespans.append(info['makespan'])
        utilizations.append(info['utilization'])
    
    print(f"平均完工时间: {np.mean(makespans):.2f}")
    print(f"平均利用率: {np.mean(utilizations):.2%}")
```

---

## 下一步

1. ✅ 阅读本指南
2. ✅ 运行 `rl_scheduler.py` 示例
3. ⬜ 根据你的需求修改状态/动作/奖励函数
4. ⬜ 实现你自己的 RL 算法
5. ⬜ 训练模型
6. ⬜ 集成到 `production_scheduler_demo.py`
7. ⬜ 测试和优化

---

## 参考资料

- **论文**: "Deep Reinforcement Learning for Job Shop Scheduling" (2020)
- **书籍**: "Reinforcement Learning: An Introduction" by Sutton & Barto
- **代码库**: 
  - Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
  - OpenAI Gym: https://gym.openai.com/
  - Ray RLlib: https://docs.ray.io/en/latest/rllib/

---

**编写**: Shannon 团队  
**日期**: 2026-01-29  
**版本**: 1.0
