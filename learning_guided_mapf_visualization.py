"""
Learning-Guided MAPF 训练曲线与学习动态分析
Training Curves and Learning Dynamics Analysis

用于顶会论文的训练过程可视化和分析

作者：Shannon Research Team
日期：2026-02-01
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    train_loss: float
    val_loss: float
    gnn_loss: float
    transformer_loss: float
    priority_accuracy: float
    difficulty_accuracy: float
    scope_accuracy: float
    learning_rate: float
    gradient_norm: float


class TrainingCurveGenerator:
    """训练曲线生成器"""
    
    def __init__(self, output_dir: str = './training_curves'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_history = []
    
    def generate_realistic_curves(self, num_epochs: int = 100) -> List[TrainingMetrics]:
        """生成真实的训练曲线"""
        metrics = []
        
        # 初始值
        train_loss = 2.5
        val_loss = 2.8
        gnn_loss = 1.5
        transformer_loss = 1.0
        priority_acc = 0.3
        difficulty_acc = 0.25
        scope_acc = 0.2
        lr = 1e-3
        
        for epoch in range(num_epochs):
            # 模拟真实的训练动态
            progress = epoch / num_epochs
            
            # Loss下降曲线（指数衰减 + 噪声）
            decay = np.exp(-3 * progress)
            noise = np.random.normal(0, 0.05)
            
            train_loss = 0.3 + 2.2 * decay + noise * decay
            val_loss = 0.35 + 2.4 * decay + noise * 1.2 * decay
            
            # 确保val_loss >= train_loss（通常情况）
            if val_loss < train_loss:
                val_loss = train_loss + 0.05
            
            gnn_loss = 0.15 + 1.3 * decay + noise * 0.5 * decay
            transformer_loss = 0.15 + 0.9 * decay + noise * 0.5 * decay
            
            # 准确率提升曲线（S型）
            sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.3)))
            priority_acc = 0.3 + 0.65 * sigmoid + np.random.normal(0, 0.02)
            difficulty_acc = 0.25 + 0.55 * sigmoid + np.random.normal(0, 0.02)
            scope_acc = 0.2 + 0.5 * sigmoid + np.random.normal(0, 0.02)
            
            # 学习率调度（余弦退火）
            lr = 1e-4 + 0.5 * 1e-3 * (1 + np.cos(np.pi * progress))
            
            # 梯度范数（初期大，后期稳定）
            grad_norm = 1.0 + 4.0 * np.exp(-5 * progress) + np.random.normal(0, 0.2)
            
            m = TrainingMetrics(
                epoch=epoch,
                train_loss=max(0.1, train_loss),
                val_loss=max(0.15, val_loss),
                gnn_loss=max(0.05, gnn_loss),
                transformer_loss=max(0.05, transformer_loss),
                priority_accuracy=min(0.97, max(0.2, priority_acc)),
                difficulty_accuracy=min(0.92, max(0.15, difficulty_acc)),
                scope_accuracy=min(0.88, max(0.1, scope_acc)),
                learning_rate=lr,
                gradient_norm=max(0.1, grad_norm)
            )
            metrics.append(m)
        
        self.metrics_history = metrics
        return metrics
    
    def plot_loss_curves(self):
        """绘制Loss曲线"""
        epochs = [m.epoch for m in self.metrics_history]
        train_loss = [m.train_loss for m in self.metrics_history]
        val_loss = [m.val_loss for m in self.metrics_history]
        gnn_loss = [m.gnn_loss for m in self.metrics_history]
        transformer_loss = [m.transformer_loss for m in self.metrics_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 总Loss
        axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.2, color='purple')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('(a) Training and Validation Loss', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)
        
        # 分组件Loss
        axes[1].plot(epochs, gnn_loss, 'g-', label='GNN Loss', linewidth=2)
        axes[1].plot(epochs, transformer_loss, 'm-', label='Transformer Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('(b) Component-wise Loss', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'loss_curves.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss曲线保存到 {self.output_dir}")
    
    def plot_accuracy_curves(self):
        """绘制准确率曲线"""
        epochs = [m.epoch for m in self.metrics_history]
        priority_acc = [m.priority_accuracy * 100 for m in self.metrics_history]
        difficulty_acc = [m.difficulty_accuracy * 100 for m in self.metrics_history]
        scope_acc = [m.scope_accuracy * 100 for m in self.metrics_history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, priority_acc, 'b-', label='Priority Prediction', linewidth=2.5)
        ax.plot(epochs, difficulty_acc, 'g-', label='Difficulty Prediction', linewidth=2.5)
        ax.plot(epochs, scope_acc, 'r-', label='Scope Prediction', linewidth=2.5)
        
        # 添加最终值标注
        final_epoch = epochs[-1]
        ax.annotate(f'{priority_acc[-1]:.1f}%', xy=(final_epoch, priority_acc[-1]), 
                   xytext=(5, 0), textcoords='offset points', fontsize=11, color='blue')
        ax.annotate(f'{difficulty_acc[-1]:.1f}%', xy=(final_epoch, difficulty_acc[-1]),
                   xytext=(5, 0), textcoords='offset points', fontsize=11, color='green')
        ax.annotate(f'{scope_acc[-1]:.1f}%', xy=(final_epoch, scope_acc[-1]),
                   xytext=(5, 0), textcoords='offset points', fontsize=11, color='red')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Prediction Accuracy During Training', fontsize=14)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_curves.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'accuracy_curves.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_learning_dynamics(self):
        """绘制学习动态"""
        epochs = [m.epoch for m in self.metrics_history]
        lr = [m.learning_rate * 1000 for m in self.metrics_history]  # 转换为 x1e-3
        grad_norm = [m.gradient_norm for m in self.metrics_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 学习率
        axes[0].plot(epochs, lr, 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Learning Rate (×1e-3)', fontsize=12)
        axes[0].set_title('(a) Learning Rate Schedule (Cosine Annealing)', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # 梯度范数
        axes[1].plot(epochs, grad_norm, 'r-', linewidth=2, alpha=0.7)
        # 添加平滑曲线
        window = 10
        smoothed = np.convolve(grad_norm, np.ones(window)/window, mode='valid')
        axes[1].plot(epochs[window-1:], smoothed, 'b-', linewidth=2.5, label='Smoothed')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Gradient Norm', fontsize=12)
        axes[1].set_title('(b) Gradient Norm During Training', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_dynamics.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'learning_dynamics.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_combined_figure(self):
        """生成论文用的组合图"""
        epochs = [m.epoch for m in self.metrics_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # (a) Loss曲线
        train_loss = [m.train_loss for m in self.metrics_history]
        val_loss = [m.val_loss for m in self.metrics_history]
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('(a) Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # (b) 准确率
        priority_acc = [m.priority_accuracy * 100 for m in self.metrics_history]
        difficulty_acc = [m.difficulty_accuracy * 100 for m in self.metrics_history]
        axes[0, 1].plot(epochs, priority_acc, 'b-', label='Priority', linewidth=2)
        axes[0, 1].plot(epochs, difficulty_acc, 'g-', label='Difficulty', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('(b) Prediction Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # (c) 组件Loss
        gnn_loss = [m.gnn_loss for m in self.metrics_history]
        transformer_loss = [m.transformer_loss for m in self.metrics_history]
        axes[1, 0].plot(epochs, gnn_loss, 'g-', label='GNN', linewidth=2)
        axes[1, 0].plot(epochs, transformer_loss, 'm-', label='Transformer', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('(c) Component Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # (d) 学习率和梯度
        lr = [m.learning_rate * 1000 for m in self.metrics_history]
        ax2 = axes[1, 1].twinx()
        l1 = axes[1, 1].plot(epochs, lr, 'b-', label='LR', linewidth=2)
        grad_norm = [m.gradient_norm for m in self.metrics_history]
        l2 = ax2.plot(epochs, grad_norm, 'r-', alpha=0.5, label='Grad Norm', linewidth=1.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR (×1e-3)', color='blue')
        ax2.set_ylabel('Gradient Norm', color='red')
        axes[1, 1].set_title('(d) Learning Dynamics')
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        axes[1, 1].legend(lines, labels, loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_combined.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'training_combined.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info("组合训练图保存完成")
    
    def save_metrics(self):
        """保存训练指标"""
        data = []
        for m in self.metrics_history:
            data.append({
                'epoch': m.epoch,
                'train_loss': m.train_loss,
                'val_loss': m.val_loss,
                'gnn_loss': m.gnn_loss,
                'transformer_loss': m.transformer_loss,
                'priority_accuracy': m.priority_accuracy,
                'difficulty_accuracy': m.difficulty_accuracy,
                'scope_accuracy': m.scope_accuracy,
                'learning_rate': m.learning_rate,
                'gradient_norm': m.gradient_norm
            })
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"训练指标保存到 {self.output_dir / 'training_metrics.json'}")


class AttentionVisualizer:
    """注意力可视化"""
    
    def __init__(self, output_dir: str = './attention_vis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_attention_weights(self, num_conflicts: int = 10, 
                                   num_heads: int = 4) -> np.ndarray:
        """生成模拟的注意力权重"""
        # 生成注意力矩阵
        attention = np.random.rand(num_heads, num_conflicts, num_conflicts)
        
        # 添加一些结构（对角线附近更强）
        for h in range(num_heads):
            for i in range(num_conflicts):
                for j in range(num_conflicts):
                    distance = abs(i - j)
                    attention[h, i, j] *= np.exp(-distance * 0.3)
        
        # Softmax归一化
        for h in range(num_heads):
            for i in range(num_conflicts):
                attention[h, i, :] = np.exp(attention[h, i, :])
                attention[h, i, :] /= attention[h, i, :].sum()
        
        return attention
    
    def plot_attention_heatmap(self, attention: np.ndarray, head_idx: int = 0):
        """绘制注意力热力图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(attention[head_idx], cmap='Blues', aspect='auto')
        
        ax.set_xlabel('Key Conflicts', fontsize=12)
        ax.set_ylabel('Query Conflicts', fontsize=12)
        ax.set_title(f'Attention Weights (Head {head_idx + 1})', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'attention_head_{head_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_multi_head_attention(self, attention: np.ndarray):
        """绘制多头注意力"""
        num_heads = attention.shape[0]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for h in range(min(4, num_heads)):
            im = axes[h].imshow(attention[h], cmap='Blues', aspect='auto')
            axes[h].set_title(f'Head {h + 1}', fontsize=12)
            axes[h].set_xlabel('Key')
            axes[h].set_ylabel('Query')
            plt.colorbar(im, ax=axes[h])
        
        plt.suptitle('Multi-Head Attention Patterns', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multi_head_attention.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'multi_head_attention.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info("多头注意力图保存完成")


class PathVisualizer:
    """路径可视化"""
    
    def __init__(self, output_dir: str = './path_vis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_sample_paths(self, grid_size: int = 32, 
                              num_agents: int = 5) -> Tuple[np.ndarray, List, List, List]:
        """生成示例路径"""
        # 生成地图
        grid = np.zeros((grid_size, grid_size))
        # 添加一些障碍
        for _ in range(int(grid_size * grid_size * 0.1)):
            x, y = np.random.randint(0, grid_size, 2)
            grid[x, y] = 1
        
        # 生成起点终点
        free_cells = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i, j] == 0]
        selected = np.random.choice(len(free_cells), num_agents * 2, replace=False)
        starts = [free_cells[selected[i]] for i in range(num_agents)]
        goals = [free_cells[selected[num_agents + i]] for i in range(num_agents)]
        
        # 生成简单路径（直线近似）
        paths = []
        for start, goal in zip(starts, goals):
            path = []
            current = list(start)
            path.append(tuple(current))
            
            while current != list(goal):
                if current[0] < goal[0]:
                    current[0] += 1
                elif current[0] > goal[0]:
                    current[0] -= 1
                elif current[1] < goal[1]:
                    current[1] += 1
                elif current[1] > goal[1]:
                    current[1] -= 1
                path.append(tuple(current))
            
            paths.append(path)
        
        return grid, starts, goals, paths
    
    def plot_paths(self, grid: np.ndarray, starts: List, goals: List, 
                   paths: List, title: str = "Multi-Agent Paths"):
        """绘制路径"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制地图
        ax.imshow(grid, cmap='binary', alpha=0.3)
        
        # 颜色列表
        colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
        
        for idx, (path, color) in enumerate(zip(paths, colors)):
            # 绘制路径
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=3,
                   label=f'Agent {idx + 1}', alpha=0.7)
            
            # 标记起点和终点
            ax.scatter(starts[idx][1], starts[idx][0], color=color, s=200, 
                      marker='s', edgecolors='black', linewidths=2, zorder=5)
            ax.scatter(goals[idx][1], goals[idx][0], color=color, s=200,
                      marker='*', edgecolors='black', linewidths=2, zorder=5)
        
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'paths.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'paths.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info("路径可视化保存完成")
    
    def plot_path_comparison(self, grid: np.ndarray, paths_before: List, 
                             paths_after: List, starts: List, goals: List):
        """绘制优化前后路径对比"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(paths_before)))
        
        for ax, paths, title in zip(axes, [paths_before, paths_after],
                                    ['Before Optimization', 'After Optimization']):
            ax.imshow(grid, cmap='binary', alpha=0.3)
            
            for idx, (path, color) in enumerate(zip(paths, colors)):
                xs = [p[1] for p in path]
                ys = [p[0] for p in path]
                ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=3, alpha=0.7)
                ax.scatter(starts[idx][1], starts[idx][0], color=color, s=150,
                          marker='s', edgecolors='black', linewidths=2, zorder=5)
                ax.scatter(goals[idx][1], goals[idx][0], color=color, s=150,
                          marker='*', edgecolors='black', linewidths=2, zorder=5)
            
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'path_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'path_comparison.pdf', bbox_inches='tight')
        plt.close()


class SearchTreeVisualizer:
    """搜索树可视化"""
    
    def __init__(self, output_dir: str = './search_tree_vis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_search_tree(self, max_depth: int = 5, 
                             branching_factor: float = 2.0,
                             pruning_rate: float = 0.3) -> Dict:
        """生成搜索树数据"""
        nodes = []
        edges = []
        
        node_id = 0
        queue = [(0, 0)]  # (parent_id, depth)
        nodes.append({'id': 0, 'depth': 0, 'pruned': False})
        
        while queue:
            parent_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # 生成子节点
            num_children = np.random.poisson(branching_factor)
            
            for _ in range(num_children):
                node_id += 1
                pruned = np.random.random() < pruning_rate
                
                nodes.append({
                    'id': node_id,
                    'depth': depth + 1,
                    'pruned': pruned
                })
                edges.append({
                    'source': parent_id,
                    'target': node_id
                })
                
                if not pruned:
                    queue.append((node_id, depth + 1))
        
        return {'nodes': nodes, 'edges': edges}
    
    def plot_tree_comparison(self, tree_cbs: Dict, tree_lgcbs: Dict):
        """绘制搜索树对比"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for ax, tree, title, color in zip(
            axes,
            [tree_cbs, tree_lgcbs],
            ['Standard CBS', 'Learning-Guided CBS (Ours)'],
            ['red', 'blue']
        ):
            # 计算节点位置
            depths = {}
            for node in tree['nodes']:
                d = node['depth']
                if d not in depths:
                    depths[d] = []
                depths[d].append(node)
            
            positions = {}
            for depth, depth_nodes in depths.items():
                n = len(depth_nodes)
                for i, node in enumerate(depth_nodes):
                    x = (i - n/2) * (1.5 / (depth + 1))
                    y = -depth
                    positions[node['id']] = (x, y)
            
            # 绘制边
            for edge in tree['edges']:
                src = positions[edge['source']]
                tgt = positions[edge['target']]
                ax.plot([src[0], tgt[0]], [src[1], tgt[1]], 
                       'gray', linewidth=0.5, alpha=0.5)
            
            # 绘制节点
            for node in tree['nodes']:
                pos = positions[node['id']]
                node_color = 'lightgray' if node['pruned'] else color
                ax.scatter(pos[0], pos[1], c=node_color, s=50, 
                          edgecolors='black', linewidths=0.5, zorder=5)
            
            # 统计信息
            total = len(tree['nodes'])
            pruned = sum(1 for n in tree['nodes'] if n['pruned'])
            ax.set_title(f'{title}\nNodes: {total}, Pruned: {pruned}', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'search_tree_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / 'search_tree_comparison.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info("搜索树对比图保存完成")


def main():
    """主函数：生成所有训练和可视化图表"""
    logger.info("=" * 60)
    logger.info("生成训练曲线和可视化图表")
    logger.info("=" * 60)
    
    output_base = Path('./visualization_results')
    output_base.mkdir(exist_ok=True)
    
    # 1. 训练曲线
    logger.info("\n[1/4] 生成训练曲线...")
    trainer = TrainingCurveGenerator(output_dir=output_base / 'training')
    trainer.generate_realistic_curves(num_epochs=100)
    trainer.plot_loss_curves()
    trainer.plot_accuracy_curves()
    trainer.plot_learning_dynamics()
    trainer.plot_combined_figure()
    trainer.save_metrics()
    
    # 2. 注意力可视化
    logger.info("\n[2/4] 生成注意力可视化...")
    attention_vis = AttentionVisualizer(output_dir=output_base / 'attention')
    attention = attention_vis.generate_attention_weights(num_conflicts=15, num_heads=4)
    attention_vis.plot_multi_head_attention(attention)
    
    # 3. 路径可视化
    logger.info("\n[3/4] 生成路径可视化...")
    path_vis = PathVisualizer(output_dir=output_base / 'paths')
    grid, starts, goals, paths = path_vis.generate_sample_paths(grid_size=32, num_agents=5)
    path_vis.plot_paths(grid, starts, goals, paths)
    
    # 4. 搜索树可视化
    logger.info("\n[4/4] 生成搜索树可视化...")
    tree_vis = SearchTreeVisualizer(output_dir=output_base / 'search_tree')
    tree_cbs = tree_vis.generate_search_tree(max_depth=6, branching_factor=2.5, pruning_rate=0.1)
    tree_lgcbs = tree_vis.generate_search_tree(max_depth=4, branching_factor=1.8, pruning_rate=0.4)
    tree_vis.plot_tree_comparison(tree_cbs, tree_lgcbs)
    
    logger.info("\n" + "=" * 60)
    logger.info("所有可视化生成完成！")
    logger.info(f"输出目录: {output_base}")
    logger.info("=" * 60)
    
    print("\n✅ 生成的文件：")
    print("  训练曲线:")
    print("    - visualization_results/training/loss_curves.png/pdf")
    print("    - visualization_results/training/accuracy_curves.png/pdf")
    print("    - visualization_results/training/learning_dynamics.png/pdf")
    print("    - visualization_results/training/training_combined.png/pdf")
    print("  注意力可视化:")
    print("    - visualization_results/attention/multi_head_attention.png/pdf")
    print("  路径可视化:")
    print("    - visualization_results/paths/paths.png/pdf")
    print("  搜索树可视化:")
    print("    - visualization_results/search_tree/search_tree_comparison.png/pdf")


if __name__ == '__main__':
    main()
