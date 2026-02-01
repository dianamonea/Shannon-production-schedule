"""
Diffusion Policy for Multi-Agent Scheduling
扩散式多智能体调度策略 - v1 实现

核心思路：
- 将联合动作表示为离散 token 序列（每个 robot 选择一个 job_id 或 idle）
- 使用 masking diffusion：从全 mask 开始，迭代去噪预测每个 robot 的任务分配
- 条件输入：全局状态（robots, jobs, stations）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class SinusoidalEmbedding(nn.Module):
    """时间步嵌入（用于扩散步数 t）"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] 或 [batch_size, 1]
        Returns:
            embeddings: [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
        
        embeddings = timesteps.float() * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        
        return embeddings


class StateEncoder(nn.Module):
    """
    全局状态编码器
    输入：robots, jobs, stations
    输出：condition embedding
    """
    
    def __init__(
        self,
        max_robots: int = 10,
        max_jobs: int = 50,
        max_stations: int = 20,
        robot_feat_dim: int = 8,
        job_feat_dim: int = 8,
        station_feat_dim: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_robots = max_robots
        self.max_jobs = max_jobs
        self.max_stations = max_stations
        
        # 分别编码每类实体
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        self.job_encoder = nn.Sequential(
            nn.Linear(job_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        self.station_encoder = nn.Sequential(
            nn.Linear(station_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        # 全局融合
        total_dim = hidden_dim * 3  # robot + job + station
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        robot_feats: torch.Tensor,  # [batch, max_robots, robot_feat_dim]
        job_feats: torch.Tensor,    # [batch, max_jobs, job_feat_dim]
        station_feats: torch.Tensor,  # [batch, max_stations, station_feat_dim]
    ) -> torch.Tensor:
        """
        Returns:
            condition: [batch, hidden_dim]
        """
        batch_size = robot_feats.size(0)
        
        # 编码并池化
        robot_embed = self.robot_encoder(robot_feats).mean(dim=1)  # [batch, hidden_dim]
        job_embed = self.job_encoder(job_feats).mean(dim=1)
        station_embed = self.station_encoder(station_feats).mean(dim=1)
        
        # 拼接融合
        combined = torch.cat([robot_embed, job_embed, station_embed], dim=-1)
        condition = self.fusion(combined)
        
        return condition


class MaskingDiffusionDenoiser(nn.Module):
    """
    Masking Diffusion 去噪网络
    输入：noisy_action_tokens (masked), condition, timestep
    输出：预测的 clean action logits
    """
    
    def __init__(
        self,
        num_robots: int,
        num_actions: int,  # max_jobs + 1（idle）
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_robots = num_robots
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Token embedding（每个 robot 的 action token）
        self.action_embed = nn.Embedding(num_actions + 1, hidden_dim)  # +1 for MASK token
        self.mask_token_id = num_actions  # 最后一个 token 作为 MASK
        
        # Timestep embedding
        self.time_embed = SinusoidalEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Condition embedding
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head：预测每个 robot 的 action logits
        self.output_head = nn.Linear(hidden_dim, num_actions)
    
    def forward(
        self,
        noisy_tokens: torch.Tensor,  # [batch, num_robots], values in [0, num_actions]
        condition: torch.Tensor,     # [batch, hidden_dim]
        timestep: torch.Tensor,      # [batch]
    ) -> torch.Tensor:
        """
        Returns:
            logits: [batch, num_robots, num_actions]
        """
        batch_size, num_robots = noisy_tokens.shape
        
        # Embed noisy action tokens
        token_embed = self.action_embed(noisy_tokens)  # [batch, num_robots, hidden_dim]
        
        # Embed timestep
        time_embed = self.time_mlp(self.time_embed(timestep))  # [batch, hidden_dim]
        time_embed = time_embed.unsqueeze(1).expand(-1, num_robots, -1)
        
        # Embed condition
        cond_embed = self.condition_proj(condition)  # [batch, hidden_dim]
        cond_embed = cond_embed.unsqueeze(1).expand(-1, num_robots, -1)
        
        # Combine
        x = token_embed + time_embed + cond_embed  # [batch, num_robots, hidden_dim]
        
        # Transformer
        x = self.transformer(x)  # [batch, num_robots, hidden_dim]
        
        # Output logits
        logits = self.output_head(x)  # [batch, num_robots, num_actions]
        
        return logits


class DiffusionPolicy(nn.Module):
    """
    完整的扩散策略模型
    """
    
    def __init__(
        self,
        max_robots: int = 10,
        max_jobs: int = 50,
        max_stations: int = 20,
        robot_feat_dim: int = 8,
        job_feat_dim: int = 8,
        station_feat_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_diffusion_steps: int = 10,
    ):
        super().__init__()
        self.max_robots = max_robots
        self.max_jobs = max_jobs
        self.num_actions = max_jobs + 1  # jobs + idle
        self.num_diffusion_steps = num_diffusion_steps
        
        # 状态编码器
        self.state_encoder = StateEncoder(
            max_robots=max_robots,
            max_jobs=max_jobs,
            max_stations=max_stations,
            robot_feat_dim=robot_feat_dim,
            job_feat_dim=job_feat_dim,
            station_feat_dim=station_feat_dim,
            hidden_dim=hidden_dim,
        )
        
        # 去噪网络
        self.denoiser = MaskingDiffusionDenoiser(
            num_robots=max_robots,
            num_actions=self.num_actions,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def encode_state(
        self,
        robot_feats: torch.Tensor,
        job_feats: torch.Tensor,
        station_feats: torch.Tensor,
    ) -> torch.Tensor:
        """编码状态为 condition"""
        return self.state_encoder(robot_feats, job_feats, station_feats)
    
    def forward(
        self,
        robot_feats: torch.Tensor,
        job_feats: torch.Tensor,
        station_feats: torch.Tensor,
        noisy_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练时的前向传播
        
        Returns:
            logits: [batch, num_robots, num_actions]
        """
        # Encode condition
        condition = self.encode_state(robot_feats, job_feats, station_feats)
        
        # Denoise
        logits = self.denoiser(noisy_tokens, condition, timestep)
        
        return logits
    
    @torch.no_grad()
    def sample(
        self,
        robot_feats: torch.Tensor,
        job_feats: torch.Tensor,
        station_feats: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理采样
        
        Args:
            robot_feats: [batch, max_robots, robot_feat_dim]
            job_feats: [batch, max_jobs, job_feat_dim]
            station_feats: [batch, max_stations, station_feat_dim]
            num_samples: 采样数量（每个 batch 生成多个候选）
            temperature: 采样温度
            seed: 随机种子
        
        Returns:
            actions: [batch, num_samples, num_robots]（每个值是 job_id 或 idle）
            scores: [batch, num_samples]（简单评分，越低越好）
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        batch_size = robot_feats.size(0)
        device = robot_feats.device
        
        # Encode condition
        condition = self.encode_state(robot_feats, job_feats, station_feats)
        
        # 扩展 batch 用于并行采样多个候选
        condition = condition.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            batch_size * num_samples, -1
        )
        
        # 初始化全 MASK
        mask_token = self.denoiser.mask_token_id
        tokens = torch.full(
            (batch_size * num_samples, self.max_robots),
            mask_token,
            dtype=torch.long,
            device=device,
        )
        
        # 迭代去噪
        for t in range(self.num_diffusion_steps - 1, -1, -1):
            timestep = torch.full((batch_size * num_samples,), t, device=device)
            
            # 预测 logits
            logits = self.denoiser(tokens, condition, timestep)  # [batch*samples, robots, actions]
            
            # 温度缩放
            logits = logits / temperature
            
            # 采样决定每个 robot 是否 unmask
            # 简单策略：每步 unmask 一部分（线性递减）
            unmask_ratio = 1.0 - (t / self.num_diffusion_steps)
            num_to_unmask = max(1, int(self.max_robots * unmask_ratio))
            
            # 对仍是 MASK 的位置进行采样
            mask_positions = (tokens == mask_token)
            
            for b in range(batch_size * num_samples):
                masked_indices = torch.where(mask_positions[b])[0]
                if len(masked_indices) == 0:
                    continue
                
                # 从 masked 位置中随机选择 unmask
                num_unmask = min(num_to_unmask, len(masked_indices))
                unmask_idx = masked_indices[torch.randperm(len(masked_indices), device=device)[:num_unmask]]
                
                # 采样新 token
                probs = F.softmax(logits[b, unmask_idx, :], dim=-1)
                new_tokens = torch.multinomial(probs, 1).squeeze(-1)
                tokens[b, unmask_idx] = new_tokens
        
        # Reshape
        actions = tokens.view(batch_size, num_samples, self.max_robots)
        
        # 简单评分（可替换为更复杂的启发式）
        # 这里用随机分数作为占位符，实际应用时可用预计完成时间等
        scores = torch.rand(batch_size, num_samples, device=device)
        
        return actions, scores


# ============================================================
# 辅助函数
# ============================================================

def create_diffusion_policy(config: Dict) -> DiffusionPolicy:
    """根据配置创建模型"""
    return DiffusionPolicy(
        max_robots=config.get('max_robots', 10),
        max_jobs=config.get('max_jobs', 50),
        max_stations=config.get('max_stations', 20),
        robot_feat_dim=config.get('robot_feat_dim', 8),
        job_feat_dim=config.get('job_feat_dim', 8),
        station_feat_dim=config.get('station_feat_dim', 6),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        num_diffusion_steps=config.get('num_diffusion_steps', 10),
    )
