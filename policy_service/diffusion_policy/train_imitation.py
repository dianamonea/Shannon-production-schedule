"""
Diffusion Policy 训练脚本（离线模仿学习）
从 episode 日志中学习专家策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse

from .model import DiffusionPolicy, create_diffusion_policy
from ..common.vectorizer import StateVectorizer


class DiffusionDataset(Dataset):
    """
    扩散策略训练数据集
    从 episode 日志加载 (state, action) 对
    """
    
    def __init__(
        self,
        data_path: str,
        max_robots: int = 10,
        max_jobs: int = 50,
        max_stations: int = 20,
    ):
        self.data_path = Path(data_path)
        self.max_robots = max_robots
        self.max_jobs = max_jobs
        self.max_stations = max_stations
        
        self.vectorizer = StateVectorizer(
            max_robots=max_robots,
            max_jobs=max_jobs,
            max_stations=max_stations,
        )
        
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        """加载所有 episode 数据"""
        print(f"Loading data from {self.data_path}")
        
        if self.data_path.is_file():
            # 单个文件
            files = [self.data_path]
        elif self.data_path.is_dir():
            # 目录：加载所有 .jsonl 或 .json 文件
            files = list(self.data_path.glob("*.jsonl")) + list(self.data_path.glob("*.json"))
        else:
            raise ValueError(f"Data path {self.data_path} not found")
        
        for file_path in tqdm(files, desc="Loading episodes"):
            if file_path.suffix == '.jsonl':
                self._load_jsonl(file_path)
            elif file_path.suffix == '.json':
                self._load_json(file_path)
        
        print(f"Loaded {len(self.samples)} samples from {len(files)} files")
    
    def _load_jsonl(self, file_path: Path):
        """加载 JSONL 格式（每行一个 step）"""
        with open(file_path, 'r') as f:
            for line in f:
                step_data = json.loads(line)
                self._add_sample(step_data)
    
    def _load_json(self, file_path: Path):
        """加载 JSON 格式（完整 episode）"""
        with open(file_path, 'r') as f:
            episode_data = json.load(f)
        
        if 'steps' in episode_data:
            # Episode 格式
            for step in episode_data['steps']:
                self._add_sample(step)
        else:
            # 单个 step
            self._add_sample(episode_data)
    
    def _add_sample(self, step_data: Dict):
        """
        添加一个样本
        step_data 格式：
        {
            "obs": {"t": ..., "robots": [...], "jobs": [...], "stations": [...]},
            "action": [{"robot_id": ..., "assign_job_id": ...}, ...],
            "reward": ...,
            "done": ...
        }
        """
        obs = step_data.get('obs')
        action = step_data.get('action')
        
        if obs is None or action is None:
            return
        
        # Vectorize state
        try:
            robot_feats, job_feats, station_feats = self.vectorizer.vectorize_state(obs)
        except Exception as e:
            print(f"Warning: Failed to vectorize state: {e}")
            return
        
        # Parse action（联合动作 -> token 序列）
        action_tokens = self._parse_action(action, obs['robots'])
        
        if action_tokens is None:
            return
        
        self.samples.append({
            'robot_feats': robot_feats,
            'job_feats': job_feats,
            'station_feats': station_feats,
            'action_tokens': action_tokens,
        })
    
    def _parse_action(self, actions: List[Dict], robots: List[Dict]) -> Optional[np.ndarray]:
        """
        将动作列表解析为 token 序列
        
        Args:
            actions: [{"robot_id": "r0", "assign_job_id": "j5"}, ...]
            robots: [{"robot_id": "r0", ...}, ...]
        
        Returns:
            tokens: [num_robots]，每个值是 job_index 或 idle_token
        """
        num_robots = len(robots)
        tokens = np.zeros(num_robots, dtype=np.int64)
        
        # 创建 robot_id -> index 映射
        robot_id_to_idx = {r['robot_id']: i for i, r in enumerate(robots)}
        
        # 创建 job_id -> index 映射（从 obs 中获取）
        # 假设 idle = 0，job_0 = 1, job_1 = 2, ...
        action_dict = {a['robot_id']: a for a in actions}
        
        for robot_id, idx in robot_id_to_idx.items():
            if robot_id in action_dict:
                action_type = action_dict[robot_id].get('action_type', 'assign_job')
                assign_job_id = action_dict[robot_id].get('assign_job_id')
                
                if action_type == 'idle' or assign_job_id is None:
                    tokens[idx] = 0  # idle
                else:
                    # 提取 job index（假设 job_id 格式为 "jX"）
                    try:
                        if isinstance(assign_job_id, str) and assign_job_id.startswith('j'):
                            job_idx = int(assign_job_id[1:]) + 1  # +1 因为 0 是 idle
                        else:
                            job_idx = int(assign_job_id) + 1
                        
                        # 限制在有效范围内
                        if 0 < job_idx <= self.max_jobs:
                            tokens[idx] = job_idx
                        else:
                            tokens[idx] = 0  # 超范围视为 idle
                    except:
                        tokens[idx] = 0  # 解析失败视为 idle
            else:
                tokens[idx] = 0  # 未指定动作视为 idle
        
        return tokens
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'robot_feats': torch.tensor(sample['robot_feats'], dtype=torch.float32),
            'job_feats': torch.tensor(sample['job_feats'], dtype=torch.float32),
            'station_feats': torch.tensor(sample['station_feats'], dtype=torch.float32),
            'action_tokens': torch.tensor(sample['action_tokens'], dtype=torch.long),
        }


def train_diffusion_imitation(
    model: DiffusionPolicy,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict,
    device: torch.device,
) -> Dict:
    """
    训练扩散策略（模仿学习）
    """
    model = model.to(device)
    model.train()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 3e-4),
        weight_decay=config.get('weight_decay', 1e-5),
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('num_epochs', 100),
        eta_min=config.get('lr_min', 1e-6),
    )
    
    # 训练参数
    num_epochs = config.get('num_epochs', 100)
    num_diffusion_steps = model.num_diffusion_steps
    mask_token = model.denoiser.mask_token_id
    
    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            robot_feats = batch['robot_feats'].to(device)
            job_feats = batch['job_feats'].to(device)
            station_feats = batch['station_feats'].to(device)
            target_tokens = batch['action_tokens'].to(device)  # [batch, num_robots]
            
            batch_size = robot_feats.size(0)
            
            # 随机采样 timestep t
            t = torch.randint(0, num_diffusion_steps, (batch_size,), device=device)
            
            # 生成 noisy tokens（masking diffusion）
            # 策略：t 越大，mask 比例越高
            mask_ratio = t.float() / num_diffusion_steps
            noisy_tokens = target_tokens.clone()
            
            for b in range(batch_size):
                num_to_mask = int(model.max_robots * mask_ratio[b].item())
                if num_to_mask > 0:
                    mask_indices = torch.randperm(model.max_robots, device=device)[:num_to_mask]
                    noisy_tokens[b, mask_indices] = mask_token
            
            # 前向传播
            logits = model(robot_feats, job_feats, station_feats, noisy_tokens, t)
            
            # 只计算被 mask 位置的 loss
            mask_positions = (noisy_tokens == mask_token)
            loss = F.cross_entropy(
                logits[mask_positions],
                target_tokens[mask_positions],
                reduction='mean',
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # 验证
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    robot_feats = batch['robot_feats'].to(device)
                    job_feats = batch['job_feats'].to(device)
                    station_feats = batch['station_feats'].to(device)
                    target_tokens = batch['action_tokens'].to(device)
                    
                    batch_size = robot_feats.size(0)
                    t = torch.randint(0, num_diffusion_steps, (batch_size,), device=device)
                    
                    mask_ratio = t.float() / num_diffusion_steps
                    noisy_tokens = target_tokens.clone()
                    
                    for b in range(batch_size):
                        num_to_mask = int(model.max_robots * mask_ratio[b].item())
                        if num_to_mask > 0:
                            mask_indices = torch.randperm(model.max_robots, device=device)[:num_to_mask]
                            noisy_tokens[b, mask_indices] = mask_token
                    
                    logits = model(robot_feats, job_feats, station_feats, noisy_tokens, t)
                    mask_positions = (noisy_tokens == mask_token)
                    loss = F.cross_entropy(
                        logits[mask_positions],
                        target_tokens[mask_positions],
                        reduction='mean',
                    )
                    
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = Path(config.get('checkpoint_dir', './checkpoints')) / 'best_diffusion_model.pt'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                
                print(f"Saved best model to {checkpoint_path}")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        scheduler.step()
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Policy')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 创建数据集
    train_dataset = DiffusionDataset(
        data_path=config['data']['train_path'],
        max_robots=config['model']['max_robots'],
        max_jobs=config['model']['max_jobs'],
        max_stations=config['model']['max_stations'],
    )
    
    val_dataset = None
    if 'val_path' in config['data']:
        val_dataset = DiffusionDataset(
            data_path=config['data']['val_path'],
            max_robots=config['model']['max_robots'],
            max_jobs=config['model']['max_jobs'],
            max_stations=config['model']['max_stations'],
        )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=True if device.type == 'cuda' else False,
        )
    
    # 创建模型
    model = create_diffusion_policy(config['model'])
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 训练
    history = train_diffusion_imitation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
    )
    
    # 保存最终模型
    final_checkpoint_path = Path(config['training'].get('checkpoint_dir', './checkpoints')) / 'final_diffusion_model.pt'
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
    }, final_checkpoint_path)
    
    print(f"Training complete. Final model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    main()
