"""
Vectorizer: 将结构化状态转换为张量表示
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math
from common.schemas import (
    StepObservation, RobotAction, VectorizedState, 
    RobotState, JobSpec, StationState
)


class StateVectorizer:
    """状态向量化器"""
    
    def __init__(
        self,
        max_robots: int = 10,
        max_jobs: int = 50,
        max_stations: int = 20,
        embed_dim: int = 128,
    ):
        """
        Args:
            max_robots: 最大机器人数（用于 padding）
            max_jobs: 最大任务数
            max_stations: 最大工作站数
            embed_dim: 嵌入维度
        """
        self.max_robots = max_robots
        self.max_jobs = max_jobs
        self.max_stations = max_stations
        self.embed_dim = embed_dim
        
        self.robot_id_to_idx = {}
        self.job_id_to_idx = {}
        self.station_id_to_idx = {}
    
    def vectorize_step(self, obs: StepObservation) -> VectorizedState:
        """将单步观测向量化"""
        
        # 1. 机器人向量化
        robot_embeddings, robot_mask = self._vectorize_robots(obs.robots)
        
        # 2. 任务向量化
        job_embeddings, job_mask = self._vectorize_jobs(obs.jobs)
        
        # 3. 工作站向量化
        station_embeddings, station_mask = self._vectorize_stations(obs.stations)
        
        # 4. 时间嵌入
        time_embedding = self._time_embedding(obs.t, obs.global_time)
        
        return VectorizedState(
            robot_embeddings=robot_embeddings,
            robot_mask=robot_mask,
            job_embeddings=job_embeddings,
            job_mask=job_mask,
            station_embeddings=station_embeddings,
            station_mask=station_mask,
            time_embedding=time_embedding,
        )
    
    def _vectorize_robots(self, robots: List[RobotState]) -> Tuple[List[List[float]], List[int]]:
        """
        机器人向量化
        返回：[num_robots_padded, embed_dim] 和 mask
        """
        embeddings = []
        mask = []
        
        for robot in robots:
            embedding = self._robot_to_vector(robot)
            embeddings.append(embedding)
            mask.append(1)  # valid
        
        # Padding
        num_padding = self.max_robots - len(robots)
        for _ in range(num_padding):
            embeddings.append([0.0] * self.embed_dim)
            mask.append(0)  # invalid (padding)
        
        return embeddings, mask
    
    def _robot_to_vector(self, robot: RobotState) -> List[float]:
        """单个机器人向量化"""
        features = []
        
        # 位置（归一化，假设范围 [0, 100]）
        x = robot.position.get("x", 0.0) / 100.0
        y = robot.position.get("y", 0.0) / 100.0
        features.extend([x, y])
        
        # 状态（One-hot）
        status_map = {"idle": 0, "working": 1, "charging": 2, "error": 3}
        status_idx = status_map.get(robot.status, 0)
        status_one_hot = [0.0] * 4
        status_one_hot[status_idx] = 1.0
        features.extend(status_one_hot)
        
        # 电量
        battery = robot.battery_level / 100.0
        features.append(battery)
        
        # 负载
        load = robot.load_capacity
        features.append(load)
        
        # 是否有当前任务
        has_job = 1.0 if robot.current_job_id else 0.0
        features.append(has_job)
        
        # 补充到 embed_dim
        while len(features) < self.embed_dim:
            features.append(0.0)
        
        return features[:self.embed_dim]
    
    def _vectorize_jobs(self, jobs: List[JobSpec]) -> Tuple[List[List[float]], List[int]]:
        """任务向量化"""
        embeddings = []
        mask = []
        
        for job in jobs:
            embedding = self._job_to_vector(job)
            embeddings.append(embedding)
            mask.append(1)
        
        # Padding
        num_padding = self.max_jobs - len(jobs)
        for _ in range(num_padding):
            embeddings.append([0.0] * self.embed_dim)
            mask.append(0)
        
        return embeddings, mask
    
    def _job_to_vector(self, job: JobSpec) -> List[float]:
        """单个任务向量化"""
        features = []
        
        # 优先级（归一化）
        priority = job.priority / 100.0
        features.append(priority)
        
        # 截止时间（归一化，假设范围 [0, 1000]）
        deadline = min(job.deadline, 1000) / 1000.0
        features.append(deadline)
        
        # 所需容量
        capacity = job.required_capacity
        features.append(capacity)
        
        # 任务类型（Hash 或 One-hot）
        job_type_hash = hash(job.job_type) % 10 / 10.0
        features.append(job_type_hash)
        
        # 补充
        while len(features) < self.embed_dim:
            features.append(0.0)
        
        return features[:self.embed_dim]
    
    def _vectorize_stations(self, stations: List[StationState]) -> Tuple[List[List[float]], List[int]]:
        """工作站向量化"""
        embeddings = []
        mask = []
        
        for station in stations:
            embedding = self._station_to_vector(station)
            embeddings.append(embedding)
            mask.append(1)
        
        # Padding
        num_padding = self.max_stations - len(stations)
        for _ in range(num_padding):
            embeddings.append([0.0] * self.embed_dim)
            mask.append(0)
        
        return embeddings, mask
    
    def _station_to_vector(self, station: StationState) -> List[float]:
        """单个工作站向量化"""
        features = []
        
        # 位置
        x = station.position.get("x", 0.0) / 100.0
        y = station.position.get("y", 0.0) / 100.0
        features.extend([x, y])
        
        # 是否可用
        is_available = 1.0 if station.is_available else 0.0
        features.append(is_available)
        
        # 队列长度
        queue_length = min(len(station.queued_jobs), 10) / 10.0
        features.append(queue_length)
        
        # 补充
        while len(features) < self.embed_dim:
            features.append(0.0)
        
        return features[:self.embed_dim]
    
    def _time_embedding(self, t: int, global_time: float) -> List[float]:
        """时间嵌入（位置编码风格）"""
        embedding = []
        
        # 简单的周期嵌入
        for i in range(0, self.embed_dim, 2):
            freq = 10 ** (i / self.embed_dim)
            embedding.append(math.sin(t / freq))
            if i + 1 < self.embed_dim:
                embedding.append(math.cos(t / freq))
        
        return embedding[:self.embed_dim]
    
    def vectorize_trajectory(self, trajectory: List[StepObservation]) -> np.ndarray:
        """
        将轨迹（多步）向量化为 [seq_len, state_vec_dim]
        """
        vectors = []
        for obs in trajectory:
            vec_state = self.vectorize_step(obs)
            # 拼接所有部分
            full_vector = (
                vec_state.robot_embeddings +
                vec_state.job_embeddings +
                vec_state.station_embeddings +
                [vec_state.time_embedding]
            )
            # 展平
            flat_vector = []
            for part in full_vector:
                if isinstance(part, list):
                    flat_vector.extend(part)
                else:
                    flat_vector.extend(part)
            vectors.append(flat_vector)
        
        return np.array(vectors, dtype=np.float32)


class ActionVectorizer:
    """动作向量化/反向向量化"""
    
    def __init__(self):
        self.job_id_cache = {}  # job_id -> idx 缓存
        self.job_idx_cache = {}  # idx -> job_id 缓存
        self.next_job_idx = 0
    
    def actions_to_targets(self, actions: List[RobotAction], available_jobs: List[str]) -> np.ndarray:
        """
        将动作列表转换为目标索引数组
        用于监督学习：每个机器人一个目标（job index 或 idle=max_idx）
        
        Returns:
            [num_robots, num_actions] 的目标索引（用于 CrossEntropyLoss）
        """
        # 动作空间：可用的 job + idle
        action_space = available_jobs + ["idle"]
        action_to_idx = {action: i for i, action in enumerate(action_space)}
        
        targets = []
        for action in actions:
            if action.action_type == "assign_job":
                job_id = action.assign_job_id
                target_idx = action_to_idx.get(job_id, len(action_space) - 1)  # 默认 idle
            else:
                target_idx = action_to_idx.get("idle", len(action_space) - 1)
            targets.append(target_idx)
        
        return np.array(targets, dtype=np.int64)

    def build_job_id_order(self, jobs: List[JobSpec], max_jobs: int) -> List[Optional[str]]:
        """构建固定长度的 job_id 顺序（用于动作索引对齐）"""
        job_ids = [job.job_id for job in jobs][:max_jobs]
        if len(job_ids) < max_jobs:
            job_ids.extend([None] * (max_jobs - len(job_ids)))
        return job_ids

    def build_action_mask(self, job_id_order: List[Optional[str]]) -> np.ndarray:
        """构建动作 mask（有效 job + idle）"""
        valid_jobs = np.array([1.0 if jid is not None else 0.0 for jid in job_id_order], dtype=np.float32)
        idle = np.array([1.0], dtype=np.float32)
        return np.concatenate([valid_jobs, idle], axis=0)

    def actions_to_targets_fixed(
        self,
        actions: List[RobotAction],
        job_id_order: List[Optional[str]],
    ) -> np.ndarray:
        """将动作映射到固定 max_jobs+1 的索引空间"""
        action_to_idx = {jid: idx for idx, jid in enumerate(job_id_order) if jid is not None}
        idle_idx = len(job_id_order)  # 最后一个为 idle

        targets = []
        for action in actions:
            if action.action_type == "assign_job" and action.assign_job_id in action_to_idx:
                targets.append(action_to_idx[action.assign_job_id])
            else:
                targets.append(idle_idx)

        return np.array(targets, dtype=np.int64)
    
    def logits_to_actions(
        self,
        logits: np.ndarray,  # [num_robots, num_actions]
        robot_ids: List[str],
        available_jobs: List[str],
    ) -> List[RobotAction]:
        """
        从 logits 生成动作
        Args:
            logits: [num_robots, num_actions]
            robot_ids: 机器人 ID 列表
            available_jobs: 可用任务列表
        """
        action_space = available_jobs + ["idle"]
        actions = []
        
        # 贪心：选择 logit 最高的动作
        predicted_indices = np.argmax(logits, axis=1)
        
        for robot_id, action_idx in zip(robot_ids, predicted_indices):
            action = action_space[action_idx]
            
            if action == "idle":
                actions.append(RobotAction(
                    robot_id=robot_id,
                    action_type="idle"
                ))
            else:
                actions.append(RobotAction(
                    robot_id=robot_id,
                    action_type="assign_job",
                    assign_job_id=action
                ))
        
        return actions

    def logits_to_actions_fixed(
        self,
        logits: np.ndarray,  # [num_robots, max_jobs+1]
        robot_ids: List[str],
        job_id_order: List[Optional[str]],
        action_mask: Optional[np.ndarray] = None,
    ) -> List[RobotAction]:
        """从固定动作空间的 logits 生成动作（含 mask）"""
        masked_logits = logits
        if action_mask is not None:
            masked_logits = np.where(action_mask[None, :] > 0, logits, -1e8)

        predicted_indices = np.argmax(masked_logits, axis=1)
        actions = []

        for robot_id, action_idx in zip(robot_ids, predicted_indices):
            if action_idx >= len(job_id_order) or job_id_order[action_idx] is None:
                actions.append(RobotAction(
                    robot_id=robot_id,
                    action_type="idle",
                ))
            else:
                actions.append(RobotAction(
                    robot_id=robot_id,
                    action_type="assign_job",
                    assign_job_id=job_id_order[action_idx],
                ))

        return actions
