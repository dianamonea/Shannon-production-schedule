"""
扩散式多智能体强化学习策略 (Diffusion-based Multi-Agent Reinforcement Learning)
Diffusion Policy for MARL - Dynamic Job Shop Scheduling

核心思想：
1. 使用扩散模型生成多智能体的协调动作序列
2. 每个智能体的策略由扩散过程指导
3. 多智能体之间通过通信进行协调
4. 支持动态调整和实时学习

参考论文：
- "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al., 2023)
- "DiffMARL: Diffusion-based Multi-Agent Reinforcement Learning" (Proposed)

作者：Shannon 团队
日期：2026-01-29
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import time
from collections import defaultdict


# ============================================================
# 扩散模型核心
# ============================================================

class DiffusionScheduler(Enum):
    """扩散调度器类型"""
    DDPM = "ddpm"           # Denoising Diffusion Probabilistic Model
    DDIM = "ddim"           # Denoising Diffusion Implicit Model
    ODE = "ode"             # ODE-based Solver
    CONSISTENCY = "consistency"  # Consistency Model


@dataclass
class DiffusionConfig:
    """扩散模型配置"""
    scheduler: DiffusionScheduler = DiffusionScheduler.DDPM
    num_steps: int = 50           # 扩散步数
    beta_start: float = 0.0001    # 初始 beta
    beta_end: float = 0.02        # 最终 beta
    noise_scale: float = 1.0      # 噪声缩放
    guidance_scale: float = 7.5   # 无分类器引导强度
    
    # 多智能体特定参数
    communication_rounds: int = 3  # 智能体通信轮数
    consensus_threshold: float = 0.8  # 共识阈值


@dataclass
class NoiseSchedule:
    """噪声调度"""
    betas: np.ndarray
    alphas: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray


class DiffusionModel:
    """扩散模型基类"""
    
    def __init__(self, config: DiffusionConfig, action_dim: int):
        self.config = config
        self.action_dim = action_dim
        
        # 初始化噪声调度
        self.noise_schedule = self._create_noise_schedule()
        
        # 噪声预测网络参数（简化版，实际使用 PyTorch/TensorFlow）
        self.model_params = self._initialize_model_params()
        
        print(f"✓ 初始化扩散模型")
        print(f"  调度器: {config.scheduler.value}")
        print(f"  扩散步数: {config.num_steps}")
        print(f"  动作维度: {action_dim}")
    
    def _create_noise_schedule(self) -> NoiseSchedule:
        """创建噪声调度"""
        betas = np.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_steps,
            dtype=np.float32
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        
        return NoiseSchedule(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod
        )
    
    def _initialize_model_params(self):
        """初始化模型参数"""
        # 简化版：使用随机参数表示噪声预测网络
        # 实际应该使用深度神经网络
        return {
            'mean': np.random.randn(self.action_dim) * 0.01,
            'std': np.ones(self.action_dim) * 0.1
        }
    
    def add_noise(self, action: np.ndarray, t: int) -> np.ndarray:
        """在时间步 t 向动作添加噪声"""
        noise = np.random.randn(*action.shape).astype(np.float32)
        
        # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * epsilon
        sqrt_alpha = self.noise_schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.noise_schedule.sqrt_one_minus_alphas_cumprod[t]
        
        noisy_action = sqrt_alpha * action + sqrt_one_minus_alpha * noise
        return noisy_action, noise
    
    def denoise(self, noisy_action: np.ndarray, t: int, context: Optional[Dict] = None) -> np.ndarray:
        """去噪步骤"""
        # 简化实现：使用高斯过程进行去噪
        # 实际应该使用训练好的神经网络
        
        noise_pred = self._predict_noise(noisy_action, t, context)
        
        # 重构原始动作
        alpha_t = self.noise_schedule.alphas_cumprod[t]
        alpha_prev = self.noise_schedule.alphas_cumprod_prev[t]
        
        sqrt_recip_alpha = self.noise_schedule.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alpha = self.noise_schedule.sqrt_recipm1_alphas_cumprod[t]
        
        # 均值估计
        mean = sqrt_recip_alpha * (noisy_action - sqrt_recipm1_alpha * noise_pred)
        
        # 方差
        beta_t = self.noise_schedule.betas[t]
        posterior_var = beta_t * (1.0 - alpha_prev) / (1.0 - alpha_t)
        
        # 添加随机性
        std = np.sqrt(posterior_var)
        noise = np.random.randn(*mean.shape).astype(np.float32)
        
        denoised = mean + std * noise if t > 0 else mean
        return denoised
    
    def _predict_noise(self, noisy_action: np.ndarray, t: int, context: Optional[Dict] = None) -> np.ndarray:
        """预测噪声（使用模型）"""
        # 这是一个简化版本
        # 实际应该使用深度神经网络
        
        # 基于上下文的条件信息
        if context is not None:
            # 融合上下文信息
            context_vec = self._encode_context(context)
            noise = context_vec * 0.1 + np.random.randn(self.action_dim) * 0.05
        else:
            noise = np.random.randn(self.action_dim) * 0.05
        
        return noise
    
    def _encode_context(self, context: Dict) -> np.ndarray:
        """编码上下文信息"""
        features = []
        
        if 'state' in context:
            state_val = context['state']
            if isinstance(state_val, np.ndarray):
                features.extend(state_val.flatten()[:self.action_dim].tolist())
            else:
                features.extend([state_val] if isinstance(state_val, (int, float)) else [0.0])
        
        if 'constraints' in context:
            constraints = context['constraints']
            if isinstance(constraints, list):
                features.extend([len(constraints) / 10.0] if constraints else [0.0])
            else:
                features.append(0.0)
        
        if 'agent_states' in context:
            agent_states = context['agent_states']
            if isinstance(agent_states, dict):
                for state_dict in agent_states.values():
                    if isinstance(state_dict, dict):
                        features.extend(list(state_dict.values())[:5])
        
        # 确保有足够的特征
        if len(features) == 0:
            features = [0.0] * self.action_dim
        
        # 转换为浮点数并截取到所需维度
        features_float = []
        for f in features[:self.action_dim]:
            if isinstance(f, (int, float)):
                features_float.append(float(f))
            else:
                features_float.append(0.0)
        
        # 补充至所需维度
        while len(features_float) < self.action_dim:
            features_float.append(0.0)
        
        return np.array(features_float[:self.action_dim], dtype=np.float32)
    
    def sample_actions(self, context: Dict, num_samples: int = 1) -> np.ndarray:
        """从扩散模型采样动作"""
        actions = []
        
        for _ in range(num_samples):
            # 从标准高斯分布开始
            x_t = np.random.randn(self.action_dim).astype(np.float32)
            
            # 反向过程（去噪）
            if self.config.scheduler == DiffusionScheduler.DDPM:
                x_t = self._ddpm_sampling(x_t, context)
            elif self.config.scheduler == DiffusionScheduler.DDIM:
                x_t = self._ddim_sampling(x_t, context)
            else:
                x_t = self._ddpm_sampling(x_t, context)
            
            actions.append(x_t)
        
        return np.array(actions)
    
    def _ddpm_sampling(self, x_t: np.ndarray, context: Dict) -> np.ndarray:
        """DDPM 采样"""
        for t in reversed(range(self.config.num_steps)):
            x_t = self.denoise(x_t, t, context)
        
        return np.clip(x_t, -1.0, 1.0)
    
    def _ddim_sampling(self, x_t: np.ndarray, context: Dict, eta: float = 0.0) -> np.ndarray:
        """DDIM 采样（加速版本）"""
        # DDIM 允许跳过步骤以加速采样
        stride = max(1, self.config.num_steps // 10)
        
        for t in reversed(range(0, self.config.num_steps, stride)):
            x_t = self.denoise(x_t, t, context)
        
        return np.clip(x_t, -1.0, 1.0)
    
    def loss_diffusion(self, action_batch: np.ndarray) -> float:
        """扩散模型损失（用于训练）"""
        batch_size = action_batch.shape[0]
        
        total_loss = 0.0
        for _ in range(batch_size):
            # 随机选择时间步
            t = np.random.randint(0, self.config.num_steps)
            
            # 添加噪声
            action = action_batch[np.random.randint(batch_size)]
            noisy_action, noise = self.add_noise(action, t)
            
            # 噪声预测
            noise_pred = self._predict_noise(noisy_action, t)
            
            # MSE 损失
            loss = np.mean((noise - noise_pred) ** 2)
            total_loss += loss
        
        return total_loss / batch_size


# ============================================================
# 多智能体通信和协调
# ============================================================

@dataclass
class AgentMessage:
    """智能体通信消息"""
    sender_id: str
    receiver_id: str
    message_type: str  # "action_proposal", "constraint", "consensus", etc.
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0


class CommunicationGraph:
    """多智能体通信图"""
    
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.edges = {aid: set() for aid in agent_ids}
        self.messages = defaultdict(list)
        
        # 默认完全连接图
        for agent1 in agent_ids:
            for agent2 in agent_ids:
                if agent1 != agent2:
                    self.edges[agent1].add(agent2)
    
    def add_edge(self, from_agent: str, to_agent: str):
        """添加通信边"""
        if from_agent in self.edges and to_agent in self.agent_ids:
            self.edges[from_agent].add(to_agent)
    
    def remove_edge(self, from_agent: str, to_agent: str):
        """移除通信边"""
        if from_agent in self.edges:
            self.edges[from_agent].discard(to_agent)
    
    def send_message(self, message: AgentMessage):
        """发送消息"""
        self.messages[message.receiver_id].append(message)
    
    def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """接收消息"""
        messages = self.messages.get(agent_id, [])
        self.messages[agent_id] = []  # 清空消息
        return messages
    
    def get_neighbors(self, agent_id: str) -> List[str]:
        """获取邻域智能体"""
        return list(self.edges.get(agent_id, []))


class MultiAgentCoordinator:
    """多智能体协调器"""
    
    def __init__(self, agent_ids: List[str], config: DiffusionConfig):
        self.agent_ids = agent_ids
        self.config = config
        self.comm_graph = CommunicationGraph(agent_ids)
        
        # 每个智能体的扩散模型
        self.diffusion_models = {
            aid: DiffusionModel(config, action_dim=50)
            for aid in agent_ids
        }
        
        # 智能体状态
        self.agent_states = {aid: {} for aid in agent_ids}
        self.local_actions = {aid: None for aid in agent_ids}
        self.action_history = {aid: [] for aid in agent_ids}
    
    def coordinate_actions(self, 
                          global_state: Dict[str, Any],
                          agent_contexts: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """协调多智能体动作生成"""
        
        coordinated_actions = {}
        
        for communication_round in range(self.config.communication_rounds):
            actions = {}
            
            # 第1阶段：每个智能体独立生成初始动作
            for agent_id in self.agent_ids:
                context = agent_contexts.get(agent_id, {})
                
                # 添加全局状态信息
                context['global_state'] = global_state
                
                # 添加邻域信息
                neighbors = self.comm_graph.get_neighbors(agent_id)
                context['neighbors'] = {
                    nid: self.agent_states[nid]
                    for nid in neighbors
                }
                
                # 采样动作
                action = self.diffusion_models[agent_id].sample_actions(context, num_samples=1)[0]
                actions[agent_id] = action
                self.local_actions[agent_id] = action
            
            # 第2阶段：通信和约束检查
            if communication_round < self.config.communication_rounds - 1:
                actions = self._communicate_and_refine(actions, global_state)
        
        return actions
    
    def _communicate_and_refine(self, 
                                actions: Dict[str, np.ndarray],
                                global_state: Dict) -> Dict[str, np.ndarray]:
        """通信和动作优化"""
        
        refined_actions = {}
        
        for agent_id in self.agent_ids:
            # 收集邻域信息
            neighbors = self.comm_graph.get_neighbors(agent_id)
            neighbor_actions = {
                nid: actions.get(nid, np.zeros(50))
                for nid in neighbors
            }
            
            # 检查冲突
            conflicts = self._detect_conflicts(agent_id, actions[agent_id], neighbor_actions)
            
            if conflicts:
                # 解决冲突：调整动作
                refined_action = self._resolve_conflicts(
                    agent_id,
                    actions[agent_id],
                    neighbor_actions,
                    conflicts
                )
            else:
                refined_action = actions[agent_id]
            
            refined_actions[agent_id] = refined_action
        
        return refined_actions
    
    def _detect_conflicts(self, 
                         agent_id: str,
                         action: np.ndarray,
                         neighbor_actions: Dict[str, np.ndarray]) -> List[str]:
        """检测动作冲突"""
        conflicts = []
        
        for neighbor_id, neighbor_action in neighbor_actions.items():
            # 简单的冲突检测：动作相似度高
            similarity = np.dot(action, neighbor_action) / (np.linalg.norm(action) * np.linalg.norm(neighbor_action) + 1e-8)
            
            if similarity > 0.9:  # 动作过于相似
                conflicts.append(neighbor_id)
        
        return conflicts
    
    def _resolve_conflicts(self,
                          agent_id: str,
                          action: np.ndarray,
                          neighbor_actions: Dict[str, np.ndarray],
                          conflicts: List[str]) -> np.ndarray:
        """解决冲突"""
        
        # 方案1：添加噪声实现多样化
        perturbed_action = action + np.random.randn(len(action)) * 0.1
        
        # 方案2：朝着不同方向调整
        for conflict_agent in conflicts:
            direction = perturbed_action - neighbor_actions[conflict_agent]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            perturbed_action += direction * 0.05
        
        return np.clip(perturbed_action, -1.0, 1.0)
    
    def update_agent_state(self, agent_id: str, state: Dict):
        """更新智能体状态"""
        self.agent_states[agent_id] = state
        self.action_history[agent_id].append(self.local_actions[agent_id])


# ============================================================
# 扩散策略智能体
# ============================================================

class DiffusionPolicyAgent(ABC):
    """扩散策略智能体基类"""
    
    def __init__(self, agent_id: str, name: str, config: DiffusionConfig):
        self.agent_id = agent_id
        self.name = name
        self.config = config
        
        self.diffusion_model = DiffusionModel(config, action_dim=50)
        self.state = {}
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
    
    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        pass
    
    def generate_action(self) -> np.ndarray:
        """使用扩散模型生成动作"""
        context = self.get_context()
        action = self.diffusion_model.sample_actions(context, num_samples=1)[0]
        
        self.action_history.append(action)
        return action
    
    def update_observation(self, observation: Dict):
        """更新观察"""
        self.observation_history.append(observation)
        self.state = observation
    
    def receive_feedback(self, reward: float, info: Dict):
        """接收反馈"""
        self.reward_history.append(reward)


# ============================================================
# 扩散式车间调度集成
# ============================================================

@dataclass
class DiffusionSchedulingContext:
    """扩散式调度上下文"""
    state_vector: np.ndarray
    constraints: List[str]
    agent_states: Dict[str, Dict]
    global_metrics: Dict[str, float]
    active_disturbances: List[Dict]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'state': self.state_vector,
            'constraints': self.constraints,
            'agent_states': self.agent_states,
            'global_metrics': self.global_metrics,
            'active_disturbances': self.active_disturbances
        }


class DiffusionMachineToolAgent(DiffusionPolicyAgent):
    """扩散式机床智能体"""
    
    def __init__(self, agent_id: str, machine_ids: List[str], config: DiffusionConfig):
        super().__init__(agent_id, f"MachineToolAgent-{agent_id}", config)
        self.machine_ids = machine_ids
        self.parts_queue = []
        self.current_utilization = {}
    
    def get_context(self) -> Dict[str, Any]:
        """获取上下文"""
        return {
            'state': np.random.randn(50),  # 简化示例
            'constraints': [
                'max_queue_length:10',
                'min_utilization:0.6',
                'max_power:5000'
            ],
            'agent_states': {
                f'machine_{i}': {
                    'utilization': self.current_utilization.get(mid, 0.5),
                    'queue_length': len(self.parts_queue) if i == 0 else 0
                }
                for i, mid in enumerate(self.machine_ids)
            }
        }
    
    def schedule_parts(self, parts, current_time) -> List[Dict]:
        """使用扩散策略调度零件"""
        self.parts_queue = parts
        
        # 生成动作
        action = self.generate_action()
        
        # 将动作转换为调度计划
        schedule = self._action_to_schedule(action, parts)
        
        return schedule
    
    def _action_to_schedule(self, action: np.ndarray, parts: List[Dict]) -> List[Dict]:
        """将扩散模型的动作转换为具体调度计划"""
        
        schedule = []
        
        # 动作向量解释
        # [0:3] - 零件选择概率（softmax）
        # [3:6] - 机床选择概率（softmax）
        # [6:10] - 优先级调整
        # [10:] - 其他参数
        
        if len(parts) > 0:
            # 零件选择
            part_probs = self._softmax(action[0:min(3, len(parts))])
            selected_part_idx = np.argmax(part_probs) if len(parts) > 0 else 0
            selected_part = parts[selected_part_idx]
            
            # 机床选择
            machine_probs = self._softmax(action[3:6])
            selected_machine = self.machine_ids[np.argmax(machine_probs) % len(self.machine_ids)]
            
            # 优先级
            priority = np.tanh(action[6]) * 10  # 归一化到 [-10, 10]
            
            schedule.append({
                'part_id': selected_part['part_id'],
                'machine': selected_machine,
                'start_time': selected_part.get('arrival_time', 0),
                'process': selected_part.get('process', []),
                'priority': priority
            })
        
        return schedule
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class DiffusionAGVCoordinator(DiffusionPolicyAgent):
    """扩散式 AGV 协调器"""
    
    def __init__(self, agent_id: str, agv_ids: List[str], config: DiffusionConfig):
        super().__init__(agent_id, f"AGVCoordinator-{agent_id}", config)
        self.agv_ids = agv_ids
        self.agv_states = {}
    
    def get_context(self) -> Dict[str, Any]:
        """获取上下文"""
        return {
            'state': np.random.randn(50),
            'constraints': [
                'battery_threshold:20',
                'max_distance:500',
                'collision_avoidance:true'
            ],
            'agent_states': {
                aid: {
                    'battery': self.agv_states.get(aid, {}).get('battery', 100),
                    'location': self.agv_states.get(aid, {}).get('location', [0, 0])
                }
                for aid in self.agv_ids
            }
        }
    
    def dispatch_agvs(self, transport_requests: List[Dict], current_time: float) -> List[Dict]:
        """使用扩散策略调度 AGV"""
        
        # 生成动作
        action = self.generate_action()
        
        # 将动作转换为派遣计划
        dispatch_plan = self._action_to_dispatch(action, transport_requests)
        
        return dispatch_plan
    
    def _action_to_dispatch(self, action: np.ndarray, requests: List[Dict]) -> List[Dict]:
        """将动作转换为派遣计划"""
        
        dispatch_plan = []
        
        if len(requests) > 0:
            # AGV 选择
            agv_probs = self._softmax(action[0:min(len(self.agv_ids), 5)])
            selected_agv_idx = np.argmax(agv_probs)
            selected_agv = self.agv_ids[selected_agv_idx % len(self.agv_ids)]
            
            # 请求选择
            if len(requests) > 0:
                request_probs = self._softmax(action[5:min(5 + len(requests), 15)])
                selected_request_idx = np.argmax(request_probs) if len(request_probs) > 0 else 0
                selected_request = requests[selected_request_idx % len(requests)]
                
                dispatch_plan.append({
                    'agv_id': selected_agv,
                    'source': selected_request['source'],
                    'destination': selected_request['destination'],
                    'priority': np.tanh(action[15]) * 10
                })
        
        return dispatch_plan
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        if len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class DiffusionRobotCellAgent(DiffusionPolicyAgent):
    """扩散式机器人单元智能体"""
    
    def __init__(self, agent_id: str, robot_ids: List[str], config: DiffusionConfig):
        super().__init__(agent_id, f"RobotCellAgent-{agent_id}", config)
        self.robot_ids = robot_ids
        self.task_queue = []
    
    def get_context(self) -> Dict[str, Any]:
        """获取上下文"""
        return {
            'state': np.random.randn(50),
            'constraints': [
                'max_payload:100',
                'cycle_time:60',
                'safety_distance:0.5'
            ],
            'agent_states': {
                rid: {
                    'status': 'idle',
                    'payload': 0
                }
                for rid in self.robot_ids
            }
        }
    
    def assign_robot_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """使用扩散策略分配机器人任务"""
        
        # 生成动作
        action = self.generate_action()
        
        # 转换为任务分配计划
        assignment_plan = self._action_to_assignment(action, tasks)
        
        return assignment_plan
    
    def _action_to_assignment(self, action: np.ndarray, tasks: List[Dict]) -> List[Dict]:
        """将动作转换为任务分配"""
        
        assignment_plan = []
        
        if len(tasks) > 0:
            robot_probs = self._softmax(action[0:min(len(self.robot_ids), 5)])
            selected_robot_idx = np.argmax(robot_probs)
            selected_robot = self.robot_ids[selected_robot_idx % len(self.robot_ids)]
            
            task_probs = self._softmax(action[5:min(5 + len(tasks), 15)])
            selected_task_idx = np.argmax(task_probs) if len(task_probs) > 0 else 0
            selected_task = tasks[selected_task_idx % len(tasks)]
            
            assignment_plan.append({
                'robot_id': selected_robot,
                'task_id': selected_task['task_id'],
                'task_type': selected_task['type'],
                'sequence': np.argsort(-action[15:25])[:3]  # 前3个优先级最高的
            })
        
        return assignment_plan
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        if len(x) == 0:
            return np.array([])
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


# ============================================================
# 训练框架
# ============================================================

class DiffusionMARL:
    """扩散式多智能体强化学习框架"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.agents = {}
        self.coordinator = None
        self.training_history = {
            'epoch': [],
            'loss': [],
            'reward': [],
            'coordination_quality': []
        }
    
    def register_agent(self, agent_id: str, agent: DiffusionPolicyAgent):
        """注册智能体"""
        self.agents[agent_id] = agent
    
    def initialize_coordinator(self):
        """初始化协调器"""
        agent_ids = list(self.agents.keys())
        self.coordinator = MultiAgentCoordinator(agent_ids, self.config)
        print(f"✓ 初始化多智能体协调器")
        print(f"  智能体数量: {len(agent_ids)}")
        print(f"  通信轮数: {self.config.communication_rounds}")
    
    def train_episode(self, environment_state: Dict, episode_num: int = 0) -> Dict:
        """训练一个回合"""
        
        # 构建上下文
        agent_contexts = {
            agent_id: agent.get_context()
            for agent_id, agent in self.agents.items()
        }
        
        # 协调生成动作
        if self.coordinator is None:
            self.initialize_coordinator()
        
        actions = self.coordinator.coordinate_actions(environment_state, agent_contexts)
        
        # 计算奖励
        total_reward = 0.0
        coordination_quality = self._evaluate_coordination(actions)
        
        for agent_id, action in actions.items():
            # 计算损失
            loss = self.agents[agent_id].diffusion_model.loss_diffusion(action.reshape(1, -1))
            total_reward -= loss
        
        # 记录历史
        self.training_history['epoch'].append(episode_num)
        self.training_history['loss'].append(np.mean([
            self.agents[aid].diffusion_model.loss_diffusion(a.reshape(1, -1))
            for aid, a in actions.items()
        ]))
        self.training_history['reward'].append(total_reward / len(self.agents))
        self.training_history['coordination_quality'].append(coordination_quality)
        
        return {
            'actions': actions,
            'total_reward': total_reward,
            'coordination_quality': coordination_quality,
            'loss': self.training_history['loss'][-1]
        }
    
    def _evaluate_coordination(self, actions: Dict[str, np.ndarray]) -> float:
        """评估协调质量"""
        
        if len(actions) < 2:
            return 1.0
        
        # 计算动作多样性
        action_list = list(actions.values())
        diversities = []
        
        for i, action1 in enumerate(action_list):
            for action2 in action_list[i+1:]:
                # 计算欧氏距离
                dist = np.linalg.norm(action1 - action2)
                diversities.append(dist)
        
        # 归一化
        if diversities:
            avg_diversity = np.mean(diversities)
            coordination_quality = min(1.0, avg_diversity / 10.0)
        else:
            coordination_quality = 0.0
        
        return coordination_quality
    
    def save_checkpoint(self, path: str, episode: int):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'config': asdict(self.config),
            'agent_models': {
                agent_id: {
                    'model_params': agent.diffusion_model.model_params,
                    'state': agent.state
                }
                for agent_id, agent in self.agents.items()
            },
            'training_history': self.training_history
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        print(f"✓ 检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.training_history = checkpoint['training_history']
        
        print(f"✓ 检查点已加载: {path}")
        print(f"  最后训练回合: {checkpoint['episode']}")


# ============================================================
# 主程序示例
# ============================================================

if __name__ == '__main__':
    print("扩散式多智能体强化学习 - 示例程序")
    print("=" * 60)
    
    # 配置
    config = DiffusionConfig(
        scheduler=DiffusionScheduler.DDPM,
        num_steps=50,
        communication_rounds=3
    )
    
    # 创建智能体
    machine_agent = DiffusionMachineToolAgent(
        agent_id="machine_agent",
        machine_ids=["cnc_1", "cnc_2", "cnc_3"],
        config=config
    )
    
    agv_agent = DiffusionAGVCoordinator(
        agent_id="agv_agent",
        agv_ids=["AGV-01", "AGV-02"],
        config=config
    )
    
    robot_agent = DiffusionRobotCellAgent(
        agent_id="robot_agent",
        robot_ids=["ROBOT-01", "ROBOT-02"],
        config=config
    )
    
    # 初始化 MARL 框架
    marl = DiffusionMARL(config)
    marl.register_agent("machine_agent", machine_agent)
    marl.register_agent("agv_agent", agv_agent)
    marl.register_agent("robot_agent", robot_agent)
    marl.initialize_coordinator()
    
    # 训练
    print("\n开始训练 (5 个回合)...")
    print("=" * 60)
    
    environment_state = {
        'parts_queue_length': 5,
        'average_utilization': 0.7,
        'active_disturbances': 2
    }
    
    for epoch in range(5):
        result = marl.train_episode(environment_state, episode_num=epoch)
        
        print(f"\nEpoch {epoch + 1}/5")
        print(f"  奖励: {result['total_reward']:.4f}")
        print(f"  协调质量: {result['coordination_quality']:.4f}")
        print(f"  损失: {result['loss']:.4f}")
        
        if (epoch + 1) % 2 == 0:
            marl.save_checkpoint(f'diffusion_marl_epoch_{epoch + 1}.json', epoch + 1)
    
    print("\n✓ 训练完成！")
    print(f"  平均奖励: {np.mean(marl.training_history['reward']):.4f}")
    print(f"  平均协调质量: {np.mean(marl.training_history['coordination_quality']):.4f}")
