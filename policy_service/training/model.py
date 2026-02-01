"""
Decision Transformer 模型实现
集中式多智能体 Transformer：将全局状态序列编码，输出每个机器人的动作分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """标准位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, 1, seq_len] 或 [batch_size, seq_len] 或 [batch_size, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]
        
        # 线性投影 + reshape
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用到 value
        out = torch.matmul(attention, V)
        
        # 拼接多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 最终线性投影
        out = self.fc_out(out)
        
        return out, attention


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len]
        """
        # 自注意力 + 残差连接 + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 前向网络 + 残差连接 + LayerNorm
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class DecisionTransformer(nn.Module):
    """
    决策 Transformer 模型
    
    输入：
        - state_seq: [batch_size, seq_len, state_vec_dim] 状态序列（K 步）
        - robot_ids: 机器人 ID（用于索引动作输出）
        - available_jobs: 可用任务列表
    
    输出：
        - action_logits: [batch_size, num_robots, num_actions] 每个机器人的动作 logits
    """
    
    def __init__(
        self,
        state_vec_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_dim: int = 1024,
        max_robots: int = 10,
        max_actions: int = 51,  # max_jobs (50) + idle (1)
    ):
        super().__init__()
        
        self.state_vec_dim = state_vec_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_robots = max_robots
        self.max_actions = max_actions
        
        # 状态投影
        self.state_embedding = nn.Linear(state_vec_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=1000)
        
        # Transformer 编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 动作头（多头分类）：每个机器人一个输出
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, max_actions),  # logits for all actions
            )
            for _ in range(max_robots)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        state_seq: torch.Tensor,  # [batch_size, seq_len, state_vec_dim]
        robot_mask: Optional[torch.Tensor] = None,  # [batch_size, max_robots]
        seq_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.Tensor:
        """
        Args:
            state_seq: [batch_size, seq_len, state_vec_dim]
            robot_mask: [batch_size, max_robots] (1=valid robot, 0=padding)
        
        Returns:
            action_logits: [batch_size, max_robots, max_actions]
        """
        batch_size, seq_len, _ = state_seq.shape
        
        # 1. 状态嵌入
        x = self.state_embedding(state_seq)  # [batch_size, seq_len, hidden_dim]
        
        # 2. 位置编码
        x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        # 3. Transformer 编码
        # 构造注意力 mask（padding + causal）
        attn_mask = None
        if seq_mask is not None:
            seq_mask = seq_mask.float()
            attn_mask = seq_mask[:, None, :]  # [batch, 1, seq_len]
            attn_mask = attn_mask * seq_mask[:, :, None]  # [batch, seq_len, seq_len]

        # causal mask: 只看当前及之前时间步
        causal = torch.tril(torch.ones(seq_len, seq_len, device=state_seq.device))
        if attn_mask is None:
            attn_mask = causal.unsqueeze(0)
        else:
            attn_mask = attn_mask * causal.unsqueeze(0)

        for layer in self.transformer_layers:
            x = layer(x, attn_mask)  # [batch_size, seq_len, hidden_dim]
        
        # 4. 使用最后一步的输出进行动作预测
        last_hidden = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # 5. 多头动作分类
        action_logits_list = []
        for i, action_head in enumerate(self.action_heads):
            logits = action_head(last_hidden)  # [batch_size, max_actions]
            action_logits_list.append(logits)
        
        action_logits = torch.stack(action_logits_list, dim=1)  # [batch_size, max_robots, max_actions]
        
        # 6. 应用 robot mask（可选）
        if robot_mask is not None:
            # robot_mask: [batch_size, max_robots] -> [batch_size, max_robots, 1]
            mask_expanded = robot_mask.unsqueeze(-1)
            # 对 padding robot 的 logits 应用大负数（使 softmax -> 0）
            action_logits = action_logits * mask_expanded + (1 - mask_expanded) * (-1e8)
        
        return action_logits
    
    def get_action_distribution(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 logits 获取动作分布（softmax）
        
        Args:
            logits: [batch_size, max_robots, max_actions]
        
        Returns:
            probs: [batch_size, max_robots, max_actions]
        """
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, logits: torch.Tensor) -> torch.Tensor:
        """
        从 logits 采样动作（用于推理）
        
        Args:
            logits: [batch_size, max_robots, max_actions]
        
        Returns:
            actions: [batch_size, max_robots] 每个机器人的动作索引
        """
        # 贪心：选择 logit 最高的
        return torch.argmax(logits, dim=-1)


class MADTLoss(nn.Module):
    """
    多智能体决策 Transformer 损失函数
    v1：行为克隆 (BC) 损失 = 交叉熵
    预留：v1.5 可加入 RTG 条件化
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, max_robots, max_actions]
        targets: torch.Tensor,  # [batch_size, max_robots]
        robot_mask: Optional[torch.Tensor] = None,  # [batch_size, max_robots]
        action_mask: Optional[torch.Tensor] = None,  # [batch_size, max_actions]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            logits: 模型输出的 logits
            targets: 目标动作索引
            robot_mask: 机器人有效掩码
        
        Returns:
            loss: 标量损失
            metrics: 字典，包含详细的损失指标
        """
        # 计算每个机器人的交叉熵损失
        batch_size, num_robots, num_actions = logits.shape
        
        # 可选：屏蔽无效动作
        if action_mask is not None:
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(1)  # [batch, 1, max_actions]
            logits = torch.where(action_mask > 0, logits, torch.tensor(-1e8, device=logits.device))

        # 处理 padding 目标
        targets_safe = targets.clone()
        if robot_mask is not None:
            targets_safe[robot_mask == 0] = -100
        targets_safe[targets_safe < 0] = -100

        logits_flat = logits.reshape(-1, num_actions)  # [batch_size * max_robots, max_actions]
        targets_flat = targets_safe.reshape(-1)  # [batch_size * max_robots]
        
        ce_losses = self.ce_loss(logits_flat, targets_flat)  # [batch_size * max_robots]
        ce_losses = ce_losses.reshape(batch_size, num_robots)
        
        # 应用 mask
        if robot_mask is not None:
            ce_losses = ce_losses * robot_mask
            loss = ce_losses.sum() / (robot_mask.sum() + 1e-8)
        else:
            loss = ce_losses.mean()
        
        # 计算准确率
        predictions = torch.argmax(logits, dim=-1)  # [batch_size, max_robots]
        accuracy = (predictions == targets_safe).float()
        
        if robot_mask is not None:
            accuracy = (accuracy * robot_mask).sum() / (robot_mask.sum() + 1e-8)
        else:
            valid_mask = (targets_safe != -100).float()
            accuracy = (accuracy * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
        }
        
        return loss, metrics
