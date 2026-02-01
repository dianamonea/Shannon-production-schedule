"""
训练脚本：行为克隆 (Behavior Cloning)
"""

import os
import sys
import json
import torch
import torch.optim as optim
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Any
import yaml

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from training.model import DecisionTransformer, MADTLoss
from training.dataset import get_dataloaders
from common.schemas import ModelConfig, TrainingConfig, DatasetConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(
    model: DecisionTransformer,
    train_loader,
    optimizer: optim.Optimizer,
    loss_fn: MADTLoss,
    device: str,
    log_interval: int = 10,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        state_vectors = batch['state_vectors']  # [batch_size, seq_len, state_vec_dim]
        action_targets = batch['action_targets']  # [batch_size, max_robots]
        robot_mask = batch['robot_mask']  # [batch_size, max_robots]
        action_mask = batch.get('action_mask')  # [batch_size, max_actions]
        seq_mask = batch.get('seq_mask')  # [batch_size, seq_len]
        
        # 前向传播
        logits = model(state_vectors, robot_mask, seq_mask)  # [batch_size, max_robots, max_actions]
        
        # 计算损失
        loss, metrics = loss_fn(logits, action_targets, robot_mask, action_mask)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
    }


def eval_epoch(
    model: DecisionTransformer,
    val_loader,
    loss_fn: MADTLoss,
    device: str,
) -> Dict[str, float]:
    """评估"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            state_vectors = batch['state_vectors']
            action_targets = batch['action_targets']
            robot_mask = batch['robot_mask']
            action_mask = batch.get('action_mask')
            seq_mask = batch.get('seq_mask')
            
            logits = model(state_vectors, robot_mask, seq_mask)
            loss, metrics = loss_fn(logits, action_targets, robot_mask, action_mask)
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/v1_bc.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    model_cfg = ModelConfig(**config['model'])
    training_cfg = TrainingConfig(**config['training'])
    dataset_cfg = DatasetConfig(**config['dataset'])
    
    # 设置设备
    device = torch.device(training_cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建检查点目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    # 计算 state_vec_dim：
    # = (max_robots * embed_dim) + (max_jobs * embed_dim) + (max_stations * embed_dim) + time_embed_dim
    state_vec_dim = (
        model_cfg.max_robots * model_cfg.hidden_dim +
        model_cfg.max_jobs * model_cfg.hidden_dim +
        model_cfg.max_stations * model_cfg.hidden_dim +
        model_cfg.hidden_dim  # time embedding
    )
    
    model = DecisionTransformer(
        state_vec_dim=state_vec_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        dropout=model_cfg.dropout,
        ff_dim=model_cfg.hidden_dim * 4,
        max_robots=model_cfg.max_robots,
        max_actions=model_cfg.max_jobs + 1,  # +1 for idle
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )
    
    loss_fn = MADTLoss()
    
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg.epochs,
    )
    
    # 数据加载器
    train_loader, val_loader = get_dataloaders(
        episodes_dir=dataset_cfg.episode_dir,
        batch_size=dataset_cfg.batch_size,
        sequence_length=dataset_cfg.sequence_length,
        train_split=dataset_cfg.train_split,
        shuffle=dataset_cfg.shuffle,
        device=str(device),
    )
    
    if len(train_loader) == 0:
        print(f"Warning: No training data in {dataset_cfg.episode_dir}")
        print("Creating synthetic data for demo...")
        train_loader, val_loader = _create_dummy_loaders(device, model_cfg)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"Starting training for {training_cfg.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(training_cfg.epochs):
        print(f"\n[Epoch {epoch + 1}/{training_cfg.epochs}]")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, str(device),
            log_interval=training_cfg.log_interval,
        )
        
        # 评估
        val_metrics = eval_epoch(model, val_loader, loss_fn, str(device))
        
        # 日志
        print(f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")
        
        writer.add_scalar('loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('accuracy/val', val_metrics['accuracy'], epoch)
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            checkpoint_path = checkpoint_dir / f"best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'model': model_cfg.dict(),
                    'training': training_cfg.dict(),
                    'dataset': dataset_cfg.dict(),
                },
                'metrics': {
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                },
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
        
        scheduler.step()
    
    writer.close()
    
    print(f"\n=== Training Complete ===")
    print(f"Best epoch: {best_epoch + 1}, Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def _create_dummy_loaders(device, model_cfg):
    """创建虚拟数据加载器用于演示"""
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = 2
    seq_len = 4
    
    # 虚拟数据
    state_vec_dim = (
        model_cfg.max_robots * model_cfg.hidden_dim +
        model_cfg.max_jobs * model_cfg.hidden_dim +
        model_cfg.max_stations * model_cfg.hidden_dim +
        model_cfg.hidden_dim
    )
    
    state_vectors = torch.randn(8, seq_len, state_vec_dim, device=device)
    action_targets = torch.randint(0, model_cfg.max_jobs + 1, (8, model_cfg.max_robots), device=device)
    robot_mask = torch.ones(8, model_cfg.max_robots, device=device)
    
    dataset = TensorDataset(state_vectors, action_targets, robot_mask)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [6, 2], generator=torch.Generator().manual_seed(42)
    )
    
    def collate_fn(batch):
        states, targets, masks = zip(*batch)
        return {
            'state_vectors': torch.stack(states),
            'action_targets': torch.stack(targets),
            'robot_mask': torch.stack(masks),
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, val_loader


if __name__ == '__main__':
    main()
