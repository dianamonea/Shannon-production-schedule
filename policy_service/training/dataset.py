"""
数据加载器：从 JSONL/Parquet 加载 Episode，构造训练数据集
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from common.schemas import (
    StepObservation, TrajectoryStep, RobotAction, JobSpec
)
from common.vectorizer import StateVectorizer, ActionVectorizer


class EpisodeDataset(Dataset):
    """从 Episode 构造训练数据集"""
    
    def __init__(
        self,
        episodes_dir: str,
        sequence_length: int = 4,
        vectorizer: Optional[StateVectorizer] = None,
        action_vectorizer: Optional[ActionVectorizer] = None,
        max_robots: int = 10,
        max_jobs: int = 50,
    ):
        """
        Args:
            episodes_dir: 存放 episode jsonl 的目录
            sequence_length: K 步窗口大小
            vectorizer: 状态向量化器
            action_vectorizer: 动作向量化器
        """
        self.episodes_dir = Path(episodes_dir)
        self.sequence_length = sequence_length
        self.max_robots = max_robots
        self.max_jobs = max_jobs
        
        self.vectorizer = vectorizer or StateVectorizer(
            max_robots=max_robots,
            max_jobs=max_jobs,
            embed_dim=128,
        )
        self.action_vectorizer = action_vectorizer or ActionVectorizer()
        
        # 加载所有 episode
        self.trajectories = self._load_episodes()
        
        # 构造训练样本（滑窗）
        self.samples = self._create_sliding_windows()
    
    def _load_episodes(self) -> List[List[TrajectoryStep]]:
        """从 JSONL 文件加载 episode"""
        trajectories = []
        
        if not self.episodes_dir.exists():
            print(f"Warning: episodes_dir {self.episodes_dir} does not exist")
            return trajectories
        
        for jsonl_file in self.episodes_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        episode_data = json.loads(line)
                        steps = [
                            TrajectoryStep(
                                obs=StepObservation(**step['obs']),
                                action=[RobotAction(**a) for a in step['action']],
                                reward=step.get('reward', 0.0),
                                done=step.get('done', False),
                                info=step.get('info', {}),
                            )
                            for step in episode_data.get('steps', [])
                        ]
                        if steps:
                            trajectories.append(steps)
                    except Exception as e:
                        print(f"Error loading episode from {jsonl_file}: {e}")
        
        print(f"Loaded {len(trajectories)} episodes")
        return trajectories
    
    def _create_sliding_windows(self) -> List[Dict]:
        """
        从轨迹构造滑窗样本
        每个样本：[K 步观测] + [对应的动作目标]
        """
        samples = []
        
        for traj in self.trajectories:
            for t in range(len(traj) - self.sequence_length + 1):
                # 采集 K 步
                window_steps = traj[t:t + self.sequence_length]
                observations = [step.obs for step in window_steps]
                
                # 目标动作（最后一步之后的动作）
                if t + self.sequence_length < len(traj):
                    target_step = traj[t + self.sequence_length]
                    target_actions = target_step.action
                    target_obs = target_step.obs
                else:
                    # 如果到达末尾，使用最后一步的动作
                    target_actions = window_steps[-1].action if window_steps else []
                    target_obs = window_steps[-1].obs if window_steps else None
                
                samples.append({
                    'observations': observations,
                    'target_actions': target_actions,
                    'target_obs': target_obs,
                })
        
        print(f"Created {len(samples)} training samples")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回一个训练样本
        
        Returns:
            {
                'state_vectors': np.ndarray [seq_len, state_vec_dim],
                'robot_ids': List[str],
                'action_targets': np.ndarray [num_robots],
                'robot_mask': np.ndarray [max_robots],
                'available_jobs': List[str],
                'action_mask': np.ndarray [max_actions],
            }
        """
        sample = self.samples[idx]
        observations = sample['observations']
        target_actions = sample['target_actions']
        target_obs = sample.get('target_obs')
        
        # 向量化观测序列
        state_vectors = []
        robot_ids = None
        
        for obs in observations:
            # 向量化单步
            vec_state = self.vectorizer.vectorize_step(obs)
            
            # 拼接为单个向量
            flat_vec = []
            flat_vec.extend(vec_state.robot_embeddings)
            flat_vec.extend(vec_state.job_embeddings)
            flat_vec.extend(vec_state.station_embeddings)
            flat_vec.append(vec_state.time_embedding)
            
            flat_vec_array = np.concatenate([
                np.array(v, dtype=np.float32) if isinstance(v, list) else v
                for v in flat_vec
            ])
            state_vectors.append(flat_vec_array)
            
            # 收集机器人 ID 和任务
            if robot_ids is None:
                robot_ids = [r.robot_id for r in obs.robots]
        
        state_vectors = np.stack(state_vectors, axis=0)  # [seq_len, state_vec_dim]
        
        # 动作目标向量化（固定 max_jobs+1 动作空间）
        if target_obs is not None:
            job_id_order = self.action_vectorizer.build_job_id_order(target_obs.jobs, self.max_jobs)
        else:
            job_id_order = self.action_vectorizer.build_job_id_order([], self.max_jobs)
        action_targets = self.action_vectorizer.actions_to_targets_fixed(target_actions, job_id_order)
        action_mask = self.action_vectorizer.build_action_mask(job_id_order)
        
        # Robot mask
        robot_mask = np.ones(self.max_robots, dtype=np.float32)
        if robot_ids:
            robot_mask[len(robot_ids):] = 0  # padding 部分标记为 0
        
        return {
            'state_vectors': state_vectors,
            'robot_ids': robot_ids or [],
            'action_targets': action_targets,
            'robot_mask': robot_mask,
            'available_jobs': [jid for jid in job_id_order if jid is not None],
            'action_mask': action_mask,
        }


class DataCollator:
    """
    Batch collator：处理可变长度的轨迹和 job 列表
    """
    
    def __init__(self, max_robots: int = 10, max_jobs: int = 50, device: str = 'cpu'):
        self.max_robots = max_robots
        self.max_jobs = max_jobs
        self.device = device
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Args:
            batch: List of dataset samples
        
        Returns:
            {
                'state_vectors': torch.Tensor [batch_size, seq_len, state_vec_dim],
                'seq_mask': torch.Tensor [batch_size, seq_len],
                'robot_ids': List of robot id lists,
                'action_targets': torch.Tensor [batch_size, max_robots],
                'robot_mask': torch.Tensor [batch_size, max_robots],
                'available_jobs': List of job lists,
                'action_mask': torch.Tensor [batch_size, max_actions],
            }
        """
        # 找最大 seq_len 和 state_vec_dim
        max_seq_len = max(item['state_vectors'].shape[0] for item in batch)
        state_vec_dim = batch[0]['state_vectors'].shape[1]
        
        state_vectors_padded = []
        action_targets_padded = []
        robot_masks = []
        robot_ids_batch = []
        available_jobs_batch = []
        action_masks = []
        seq_masks = []
        
        for item in batch:
            # 状态向量 Padding（在 seq_len 维度）
            state_vec = item['state_vectors']  # [seq_len, state_vec_dim]
            seq_len = state_vec.shape[0]
            
            if seq_len < max_seq_len:
                # Pad with zeros
                pad = np.zeros((max_seq_len - seq_len, state_vec_dim), dtype=np.float32)
                state_vec = np.vstack([state_vec, pad])
            
            state_vectors_padded.append(state_vec)
            seq_mask = np.zeros(max_seq_len, dtype=np.float32)
            seq_mask[:seq_len] = 1.0
            seq_masks.append(seq_mask)
            
            # 动作目标 Padding（在 num_robots 维度）
            targets = item['action_targets']  # [num_robots]
            num_robots = len(targets)
            
            if num_robots < self.max_robots:
                # Pad with -1（无效索引）
                targets = np.pad(targets, (0, self.max_robots - num_robots), constant_values=-1)
            else:
                targets = targets[:self.max_robots]
            
            action_targets_padded.append(targets)
            
            # Robot mask 和 ID
            robot_masks.append(item['robot_mask'])
            robot_ids_batch.append(item['robot_ids'])
            available_jobs_batch.append(item['available_jobs'])
            action_masks.append(item['action_mask'])
        
        # 转换为 torch 张量
        state_vectors_tensor = torch.from_numpy(
            np.stack(state_vectors_padded, axis=0)
        ).to(self.device)  # [batch_size, seq_len, state_vec_dim]
        
        action_targets_tensor = torch.from_numpy(
            np.stack(action_targets_padded, axis=0)
        ).to(self.device)  # [batch_size, max_robots]
        
        robot_mask_tensor = torch.from_numpy(
            np.stack(robot_masks, axis=0)
        ).to(self.device)  # [batch_size, max_robots]

        seq_mask_tensor = torch.from_numpy(
            np.stack(seq_masks, axis=0)
        ).to(self.device)  # [batch_size, seq_len]

        action_mask_tensor = torch.from_numpy(
            np.stack(action_masks, axis=0)
        ).to(self.device)  # [batch_size, max_actions]
        
        return {
            'state_vectors': state_vectors_tensor,
            'seq_mask': seq_mask_tensor,
            'robot_ids': robot_ids_batch,
            'action_targets': action_targets_tensor,
            'robot_mask': robot_mask_tensor,
            'available_jobs': available_jobs_batch,
            'action_mask': action_mask_tensor,
        }


def get_dataloaders(
    episodes_dir: str,
    batch_size: int = 32,
    sequence_length: int = 4,
    train_split: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = 'cpu',
) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器
    
    Args:
        episodes_dir: Episode 数据目录
        batch_size: 批大小
        sequence_length: K 步窗口
        train_split: 训练集比例
        shuffle: 是否打乱
        num_workers: DataLoader worker 数
        device: torch device
    
    Returns:
        (train_loader, val_loader)
    """
    dataset = EpisodeDataset(
        episodes_dir=episodes_dir,
        sequence_length=sequence_length,
    )
    
    # 分割训练/验证集
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    
    collator = DataCollator(device=device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader
