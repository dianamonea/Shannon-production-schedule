"""
学习引导MAPF的训练框架和评估系统
Training and Evaluation Framework for Learning-Guided MAPF

包含：
1. 数据集生成和管理
2. 模型训练流程
3. 评估指标计算
4. 超参数搜索
5. 结果分析和可视化

作者：Shannon Research Team
日期：2026-02-01
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


# ============================================================
# 数据集定义
# ============================================================

@dataclass
class MapfInstance:
    """MAPF问题实例"""
    agents: int
    grid_size: int
    obstacles: float  # 障碍物比例
    grid: np.ndarray
    start_positions: List[Tuple[int, int]]
    goal_positions: List[Tuple[int, int]]
    instance_id: str


@dataclass
class ConflictTrainingExample:
    """冲突训练样本"""
    # 图结构
    agent_features: np.ndarray  # [num_agents, 6]
    edge_indices: np.ndarray    # [2, num_edges]
    edge_features: np.ndarray   # [num_edges, 4]
    
    # GNN目标
    conflict_type_labels: np.ndarray  # [num_edges] -> 0/1/2
    
    # Transformer目标
    conflict_sequence: np.ndarray     # [num_conflicts, 8]
    priority_labels: np.ndarray       # [num_conflicts, 1]
    difficulty_labels: np.ndarray     # [num_conflicts, 1]
    scope_labels: np.ndarray          # [num_conflicts, 1]
    
    # 元信息
    instance_id: str
    solution_cost: float
    search_nodes_expanded: int


class MapfDataset(Dataset):
    """MAPF数据集"""
    
    def __init__(self, examples: List[ConflictTrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class DatasetGenerator:
    """MAPF数据集生成器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_instances(self, 
                          num_instances: int = 100,
                          agents_range: Tuple[int, int] = (10, 50),
                          grid_size_range: Tuple[int, int] = (32, 64),
                          obstacle_ratio_range: Tuple[float, float] = (0.1, 0.3)
                          ) -> List[MapfInstance]:
        """
        生成MAPF问题实例
        """
        instances = []
        
        for i in range(num_instances):
            num_agents = np.random.randint(agents_range[0], agents_range[1])
            grid_size = np.random.choice([32, 64, 128])  # 标准大小
            obstacle_ratio = np.random.uniform(obstacle_ratio_range[0], 
                                             obstacle_ratio_range[1])
            
            # 生成栅格
            grid = np.random.choice([0, 1], 
                                   size=(grid_size, grid_size),
                                   p=[1-obstacle_ratio, obstacle_ratio])
            
            # 生成起点和目标点
            starts = []
            goals = []
            positions = set()
            
            for j in range(num_agents):
                while True:
                    start = (np.random.randint(0, grid_size),
                            np.random.randint(0, grid_size))
                    if grid[start[1], start[0]] == 0 and start not in positions:
                        starts.append(start)
                        positions.add(start)
                        break
                
                while True:
                    goal = (np.random.randint(0, grid_size),
                           np.random.randint(0, grid_size))
                    if (grid[goal[1], goal[0]] == 0 and goal not in positions and
                        goal != start):
                        goals.append(goal)
                        positions.add(goal)
                        break
            
            instance = MapfInstance(
                agents=num_agents,
                grid_size=grid_size,
                obstacles=obstacle_ratio,
                grid=grid,
                start_positions=starts,
                goal_positions=goals,
                instance_id=f"instance_{i:04d}"
            )
            instances.append(instance)
        
        logger.info(f"✓ 生成 {len(instances)} 个MAPF实例")
        return instances
    
    def create_training_examples(self, 
                                instances: List[MapfInstance],
                                solver_func) -> List[ConflictTrainingExample]:
        """
        从实例生成训练样本
        
        solver_func: 求解器函数，返回 (paths, search_stats)
        """
        examples = []
        
        for instance in instances:
            logger.info(f"处理 {instance.instance_id}...")
            
            # 求解MAPF问题（使用某个求解器）
            try:
                paths, search_stats = solver_func(instance)
                
                # 构造训练样本
                example = self._extract_training_example(
                    instance, paths, search_stats
                )
                examples.append(example)
            except Exception as e:
                logger.warning(f"处理 {instance.instance_id} 失败: {e}")
                continue
        
        logger.info(f"✓ 生成 {len(examples)} 个训练样本")
        return examples
    
    def _extract_training_example(self, 
                                 instance: MapfInstance,
                                 paths: Dict,
                                 search_stats: Dict) -> ConflictTrainingExample:
        """从求解结果提取训练样本"""
        
        # 构造图结构
        num_agents = instance.agents
        
        # 节点特征：[priority, norm_x, norm_y, goal_x, goal_y, path_len]
        agent_features = []
        for i in range(num_agents):
            start_x, start_y = instance.start_positions[i]
            goal_x, goal_y = instance.goal_positions[i]
            path_len = len(paths.get(i, []))
            
            features = np.array([
                i % 10,  # priority (简化)
                start_x / instance.grid_size,
                start_y / instance.grid_size,
                goal_x / instance.grid_size,
                goal_y / instance.grid_size,
                min(path_len, 50) / 50  # 归一化路径长度
            ], dtype=np.float32)
            agent_features.append(features)
        
        agent_features = np.array(agent_features)
        
        # 边特征（潜在冲突）
        edge_list = []
        edge_features = []
        
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                # 计算两个智能体之间的距离
                start_i = np.array(instance.start_positions[i])
                start_j = np.array(instance.start_positions[j])
                dist = np.linalg.norm(start_i - start_j)
                
                path_i = paths.get(i, [])
                path_j = paths.get(j, [])
                
                # 检查路径是否相交
                crossing = 1.0 if set(path_i) & set(path_j) else 0.0
                
                # 冲突时刻预测
                future_conflict = min(len(path_i), len(path_j))
                
                # 方向冲突
                direction_conflict = 0.5
                
                edge_list.append([i, j])
                edge_features.append([
                    dist / (instance.grid_size * np.sqrt(2)),  # 归一化距离
                    crossing,
                    min(future_conflict, 50) / 50,  # 归一化时间
                    direction_conflict
                ])
        
        edge_indices = np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 4), dtype=np.float32)
        
        # 标签（简化版，实际需要从搜索过程中提取）
        num_edges = edge_indices.shape[1]
        conflict_type_labels = np.random.randint(0, 3, num_edges, dtype=np.int64)
        
        # Transformer标签
        conflict_sequence = np.random.randn(num_edges, 8).astype(np.float32)
        priority_labels = np.random.rand(num_edges, 1).astype(np.float32)
        difficulty_labels = np.random.rand(num_edges, 1).astype(np.float32)
        scope_labels = np.random.rand(num_edges, 1).astype(np.float32)
        
        return ConflictTrainingExample(
            agent_features=agent_features,
            edge_indices=edge_indices,
            edge_features=edge_features,
            conflict_type_labels=conflict_type_labels,
            conflict_sequence=conflict_sequence,
            priority_labels=priority_labels,
            difficulty_labels=difficulty_labels,
            scope_labels=scope_labels,
            instance_id=instance.instance_id,
            solution_cost=search_stats.get('total_cost', 0),
            search_nodes_expanded=search_stats.get('expanded_nodes', 0)
        )


# ============================================================
# 模型训练框架
# ============================================================

class TrainingConfig:
    """训练配置"""
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.patience = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path('./checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)


class TrainingMetrics:
    """训练指标追踪"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.train_losses) - 1
    
    def plot(self, save_path='training_metrics.png'):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 精度曲线
        axes[1].plot(self.train_accuracies, label='Train Accuracy')
        axes[1].plot(self.val_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        logger.info(f"✓ 保存训练曲线到 {save_path}")


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, gnn_model: nn.Module, transformer_model: nn.Module, 
                 config: TrainingConfig):
        self.gnn_model = gnn_model.to(config.device)
        self.transformer_model = transformer_model.to(config.device)
        self.config = config
        
        # 优化器
        self.gnn_optimizer = torch.optim.Adam(
            self.gnn_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.transformer_optimizer = torch.optim.Adam(
            self.transformer_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 损失函数
        self.gnn_loss_fn = nn.CrossEntropyLoss()
        self.priority_loss_fn = nn.MSELoss()
        self.difficulty_loss_fn = nn.MSELoss()
        self.scope_loss_fn = nn.MSELoss()
        
        # 指标跟踪
        self.metrics = TrainingMetrics()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        训练GNN和Transformer模型
        """
        logger.info("\n" + "="*60)
        logger.info("开始模型训练")
        logger.info("="*60)
        
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n[Epoch {epoch+1}/{self.config.num_epochs}]")
            
            # 训练
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self._validate(val_loader)
            
            # 更新指标
            self.metrics.update(train_loss, val_loss, train_acc, val_acc)
            
            logger.info(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # 早停
            if val_loss < self.metrics.best_val_loss:
                patience_counter = 0
                self._save_checkpoint(epoch)
                logger.info(f"  ✓ 模型已保存 (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"  ⚠ 早停触发 (patience: {self.config.patience})")
                    break
        
        logger.info(f"\n✓ 训练完成，最佳epoch: {self.metrics.best_epoch + 1}")
        self.metrics.plot()
        
        return self.metrics
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.gnn_model.train()
        self.transformer_model.train()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for batch in train_loader:
            # GNN训练
            agent_feats = torch.tensor(batch.agent_features, device=self.config.device)
            edge_idx = torch.tensor(batch.edge_indices, device=self.config.device, dtype=torch.long)
            edge_feats = torch.tensor(batch.edge_features, device=self.config.device)
            gnn_labels = torch.tensor(batch.conflict_type_labels, device=self.config.device)
            
            # GNN前向
            gnn_logits = self.gnn_model(agent_feats, edge_idx, edge_feats)
            gnn_loss = self.gnn_loss_fn(gnn_logits, gnn_labels)
            
            # Transformer训练
            conflict_feats = torch.tensor(batch.conflict_sequence, device=self.config.device)
            priority_labels = torch.tensor(batch.priority_labels, device=self.config.device)
            difficulty_labels = torch.tensor(batch.difficulty_labels, device=self.config.device)
            scope_labels = torch.tensor(batch.scope_labels, device=self.config.device)
            
            # Transformer前向
            priorities, difficulties, scopes = self.transformer_model(conflict_feats)
            
            transformer_loss = (
                self.priority_loss_fn(priorities, priority_labels) +
                self.difficulty_loss_fn(difficulties, difficulty_labels) +
                self.scope_loss_fn(scopes, scope_labels)
            )
            
            # 总损失
            total = gnn_loss + transformer_loss
            
            # 反向传播
            self.gnn_optimizer.zero_grad()
            self.transformer_optimizer.zero_grad()
            total.backward()
            self.gnn_optimizer.step()
            self.transformer_optimizer.step()
            
            # 累计指标
            total_loss += total.item()
            
            # 计算精度
            gnn_acc = (gnn_logits.argmax(dim=1) == gnn_labels).float().mean().item()
            total_acc += gnn_acc
            num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.gnn_model.eval()
        self.transformer_model.eval()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # GNN验证
                agent_feats = torch.tensor(batch.agent_features, device=self.config.device)
                edge_idx = torch.tensor(batch.edge_indices, device=self.config.device, dtype=torch.long)
                edge_feats = torch.tensor(batch.edge_features, device=self.config.device)
                gnn_labels = torch.tensor(batch.conflict_type_labels, device=self.config.device)
                
                gnn_logits = self.gnn_model(agent_feats, edge_idx, edge_feats)
                gnn_loss = self.gnn_loss_fn(gnn_logits, gnn_labels)
                
                # Transformer验证
                conflict_feats = torch.tensor(batch.conflict_sequence, device=self.config.device)
                priority_labels = torch.tensor(batch.priority_labels, device=self.config.device)
                difficulty_labels = torch.tensor(batch.difficulty_labels, device=self.config.device)
                scope_labels = torch.tensor(batch.scope_labels, device=self.config.device)
                
                priorities, difficulties, scopes = self.transformer_model(conflict_feats)
                
                transformer_loss = (
                    self.priority_loss_fn(priorities, priority_labels) +
                    self.difficulty_loss_fn(difficulties, difficulty_labels) +
                    self.scope_loss_fn(scopes, scope_labels)
                )
                
                total = gnn_loss + transformer_loss
                total_loss += total.item()
                
                gnn_acc = (gnn_logits.argmax(dim=1) == gnn_labels).float().mean().item()
                total_acc += gnn_acc
                num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    def _save_checkpoint(self, epoch: int):
        """保存模型检查点"""
        checkpoint_path = self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'gnn_state': self.gnn_model.state_dict(),
            'transformer_state': self.transformer_model.state_dict(),
        }, checkpoint_path)


# ============================================================
# 评估框架
# ============================================================

class EvaluationMetrics:
    """评估指标"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, 
                  instance_id: str,
                  num_agents: int,
                  success: bool,
                  solution_cost: float,
                  makespan: float,
                  expanded_nodes: int,
                  computation_time: float,
                  method: str):
        """添加结果"""
        self.results.append({
            'instance_id': instance_id,
            'num_agents': num_agents,
            'success': success,
            'solution_cost': solution_cost,
            'makespan': makespan,
            'expanded_nodes': expanded_nodes,
            'computation_time': computation_time,
            'method': method
        })
    
    def compute_summary(self, method: str) -> Dict:
        """计算摘要统计"""
        method_results = [r for r in self.results if r['method'] == method]
        
        if not method_results:
            return {}
        
        successes = [r for r in method_results if r['success']]
        
        summary = {
            'method': method,
            'total_instances': len(method_results),
            'success_rate': len(successes) / len(method_results),
            'avg_cost': np.mean([r['solution_cost'] for r in successes]) if successes else float('inf'),
            'avg_makespan': np.mean([r['makespan'] for r in successes]) if successes else float('inf'),
            'avg_expanded': np.mean([r['expanded_nodes'] for r in method_results]),
            'avg_time': np.mean([r['computation_time'] for r in method_results]),
        }
        
        return summary
    
    def compare_methods(self, methods: List[str]) -> Dict:
        """比较多个方法"""
        comparison = {}
        for method in methods:
            comparison[method] = self.compute_summary(method)
        return comparison
    
    def save_results(self, path: str):
        """保存结果到文件"""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"✓ 结果已保存到 {path}")


# ============================================================
# 超参数搜索
# ============================================================

class HyperparameterTuner:
    """超参数搜索器"""
    
    def __init__(self, config_space: Dict):
        self.config_space = config_space
        self.results = []
    
    def grid_search(self, train_loader, val_loader, num_trials: int = 10):
        """网格搜索"""
        logger.info(f"\n开始网格搜索 ({num_trials} 次试验)")
        
        for trial in range(num_trials):
            # 从配置空间随机采样
            config = self._sample_config()
            
            logger.info(f"\n[Trial {trial+1}/{num_trials}]")
            logger.info(f"  超参数: {config}")
            
            # 训练模型
            # ... (训练逻辑)
            
            # 记录结果
            self.results.append({
                'trial': trial,
                'config': config,
                'val_loss': 0.0  # 实际评估结果
            })
        
        # 找到最佳配置
        best_result = min(self.results, key=lambda x: x['val_loss'])
        logger.info(f"\n✓ 最佳超参数: {best_result['config']}")
        
        return best_result['config']
    
    def _sample_config(self) -> Dict:
        """从配置空间采样"""
        config = {}
        for param, values in self.config_space.items():
            config[param] = np.random.choice(values)
        return config


if __name__ == '__main__':
    # 初始化日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*70)
    logger.info("学习引导MAPF - 训练和评估框架")
    logger.info("="*70)
    
    # 1. 生成数据集
    generator = DatasetGenerator(seed=42)
    instances = generator.generate_instances(
        num_instances=100,
        agents_range=(10, 50),
        grid_size_range=(32, 64)
    )
    
    logger.info(f"✓ 生成了 {len(instances)} 个MAPF实例")
    
    # 2. 创建训练样本
    # examples = generator.create_training_examples(instances, solver_func)
    
    # 3. 创建数据加载器
    # train_dataset = MapfDataset(examples[:80])
    # val_dataset = MapfDataset(examples[80:])
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 4. 训练模型
    # config = TrainingConfig()
    # trainer = ModelTrainer(gnn_model, transformer_model, config)
    # metrics = trainer.train(train_loader, val_loader)
    
    logger.info("\n✓ 框架初始化完成，准备就绪")
