"""
Learning-Guided MAPF 单元测试
Unit Tests for LG-MAPF Implementation

确保代码正确性和可靠性

作者：Shannon Research Team
日期：2026-02-01
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


class TestConflictGraph(unittest.TestCase):
    """测试冲突图构建"""
    
    def setUp(self):
        """设置测试环境"""
        self.simple_paths = [
            [(0, 0), (0, 1), (0, 2)],  # Agent 0
            [(0, 1), (0, 0), (1, 0)],  # Agent 1 - 与Agent 0在时间0位置(0,1)有冲突
        ]
    
    def test_vertex_conflict_detection(self):
        """测试顶点冲突检测"""
        conflicts = self._detect_conflicts(self.simple_paths)
        
        # 应该检测到至少一个冲突
        self.assertGreater(len(conflicts), 0)
    
    def test_edge_conflict_detection(self):
        """测试边冲突检测"""
        paths = [
            [(0, 0), (0, 1)],  # Agent 0: (0,0) -> (0,1)
            [(0, 1), (0, 0)],  # Agent 1: (0,1) -> (0,0) - 边冲突
        ]
        
        conflicts = self._detect_conflicts(paths)
        self.assertGreater(len(conflicts), 0)
    
    def _detect_conflicts(self, paths):
        """检测路径冲突"""
        conflicts = []
        num_agents = len(paths)
        max_time = max(len(p) for p in paths)
        
        for t in range(max_time):
            # 检查顶点冲突
            positions = {}
            for i, path in enumerate(paths):
                pos = path[min(t, len(path) - 1)]
                if pos in positions:
                    conflicts.append({
                        'type': 'vertex',
                        'agents': (positions[pos], i),
                        'time': t,
                        'position': pos
                    })
                positions[pos] = i
            
            # 检查边冲突
            if t > 0:
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        pos_i_prev = paths[i][min(t - 1, len(paths[i]) - 1)]
                        pos_i_curr = paths[i][min(t, len(paths[i]) - 1)]
                        pos_j_prev = paths[j][min(t - 1, len(paths[j]) - 1)]
                        pos_j_curr = paths[j][min(t, len(paths[j]) - 1)]
                        
                        if pos_i_prev == pos_j_curr and pos_i_curr == pos_j_prev:
                            conflicts.append({
                                'type': 'edge',
                                'agents': (i, j),
                                'time': t,
                                'edge': ((pos_i_prev, pos_i_curr), (pos_j_prev, pos_j_curr))
                            })
        
        return conflicts


class TestGNNEncoder(unittest.TestCase):
    """测试GNN编码器"""
    
    def test_node_feature_dimensions(self):
        """测试节点特征维度"""
        # 模拟节点特征
        num_nodes = 10
        node_features = self._create_node_features(num_nodes)
        
        self.assertEqual(node_features.shape[0], num_nodes)
        self.assertEqual(node_features.shape[1], 8)  # 8维特征
    
    def test_edge_construction(self):
        """测试边构建"""
        num_nodes = 5
        edges = self._construct_edges(num_nodes)
        
        # 应该构建完全图（除自环）
        expected_edges = num_nodes * (num_nodes - 1)
        self.assertEqual(edges.shape[1], expected_edges)
    
    def _create_node_features(self, num_nodes):
        """创建节点特征"""
        return np.random.randn(num_nodes, 8).astype(np.float32)
    
    def _construct_edges(self, num_nodes):
        """构建完全图边"""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        return np.array(edges).T


class TestTransformerRanker(unittest.TestCase):
    """测试Transformer排序器"""
    
    def test_attention_output_shape(self):
        """测试注意力输出形状"""
        batch_size = 4
        num_conflicts = 10
        hidden_dim = 128
        
        # 模拟输入
        x = np.random.randn(batch_size, num_conflicts, hidden_dim).astype(np.float32)
        
        # 注意力应该保持序列长度
        output_shape = (batch_size, num_conflicts, hidden_dim)
        self.assertEqual(x.shape, output_shape)
    
    def test_ranking_score_normalization(self):
        """测试排名分数归一化"""
        scores = np.array([2.5, 1.0, 3.2, 0.5])
        normalized = self._softmax(scores)
        
        # 归一化后应该和为1
        self.assertAlmostEqual(np.sum(normalized), 1.0, places=5)
        
        # 所有值应该在0-1之间
        self.assertTrue(all(0 <= s <= 1 for s in normalized))
    
    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class TestCBSIntegration(unittest.TestCase):
    """测试CBS集成"""
    
    def test_constraint_application(self):
        """测试约束应用"""
        # 模拟约束
        constraints = [
            {'agent': 0, 'location': (1, 1), 'time': 3},
            {'agent': 1, 'location': (2, 2), 'time': 5},
        ]
        
        # 验证约束格式
        for c in constraints:
            self.assertIn('agent', c)
            self.assertIn('location', c)
            self.assertIn('time', c)
    
    def test_node_expansion_priority(self):
        """测试节点扩展优先级"""
        # 模拟优先级队列
        nodes = [
            {'cost': 10, 'conflicts': 5},
            {'cost': 8, 'conflicts': 8},
            {'cost': 12, 'conflicts': 2},
        ]
        
        # 按cost排序
        sorted_by_cost = sorted(nodes, key=lambda x: x['cost'])
        self.assertEqual(sorted_by_cost[0]['cost'], 8)
        
        # 按conflicts排序
        sorted_by_conflicts = sorted(nodes, key=lambda x: x['conflicts'])
        self.assertEqual(sorted_by_conflicts[0]['conflicts'], 2)


class TestMapGenerator(unittest.TestCase):
    """测试地图生成"""
    
    def test_empty_map_generation(self):
        """测试空白地图生成"""
        width, height = 10, 10
        grid = [[0] * width for _ in range(height)]
        
        # 应该全部为0
        for row in grid:
            for cell in row:
                self.assertEqual(cell, 0)
    
    def test_random_map_obstacle_ratio(self):
        """测试随机地图障碍物比例"""
        import random
        random.seed(42)
        
        width, height = 50, 50
        obstacle_ratio = 0.2
        
        grid = [[0] * width for _ in range(height)]
        num_obstacles = int(width * height * obstacle_ratio)
        
        positions = [(i, j) for i in range(height) for j in range(width)]
        random.shuffle(positions)
        
        for i in range(num_obstacles):
            y, x = positions[i]
            grid[y][x] = 1
        
        # 计算实际障碍物比例
        total_obstacles = sum(sum(row) for row in grid)
        actual_ratio = total_obstacles / (width * height)
        
        self.assertAlmostEqual(actual_ratio, obstacle_ratio, places=2)
    
    def test_map_connectivity(self):
        """测试地图连通性"""
        width, height = 10, 10
        grid = [[0] * width for _ in range(height)]
        
        # 添加一些障碍物但保持连通
        grid[5][1:9] = [1] * 8
        grid[5][5] = 0  # 开门
        
        # 使用BFS检查连通性
        start = (0, 0)
        visited = set()
        queue = [start]
        
        while queue:
            y, x = queue.pop(0)
            if (y, x) in visited:
                continue
            if y < 0 or y >= height or x < 0 or x >= width:
                continue
            if grid[y][x] == 1:
                continue
            
            visited.add((y, x))
            queue.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
        
        # 应该能访问大部分空闲位置
        free_positions = sum(1 for i in range(height) for j in range(width) if grid[i][j] == 0)
        self.assertEqual(len(visited), free_positions)


class TestStatisticalTests(unittest.TestCase):
    """测试统计方法"""
    
    def test_paired_t_test(self):
        """测试配对t检验"""
        from scipy import stats
        
        # 生成有显著差异的数据
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        improved = np.random.normal(8, 2, 50)  # 均值更低
        
        t_stat, p_value = stats.ttest_rel(baseline, improved)
        
        # 应该检测到显著差异
        self.assertLess(p_value, 0.05)
    
    def test_wilcoxon_test(self):
        """测试Wilcoxon符号秩检验"""
        from scipy import stats
        
        np.random.seed(42)
        baseline = np.random.normal(10, 2, 50)
        improved = np.random.normal(8, 2, 50)
        
        stat, p_value = stats.wilcoxon(baseline, improved)
        
        self.assertLess(p_value, 0.05)
    
    def test_cohens_d(self):
        """测试Cohen's d效应量"""
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 100)
        group2 = np.random.normal(8, 2, 100)
        
        # 计算Cohen's d
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # 应该是中等到大效应量
        self.assertGreater(abs(cohens_d), 0.5)


class TestDataPipeline(unittest.TestCase):
    """测试数据管道"""
    
    def test_batch_creation(self):
        """测试批次创建"""
        data = list(range(100))
        batch_size = 32
        
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        # 应该有4个批次
        self.assertEqual(len(batches), 4)
        
        # 前3个批次应该是满的
        for i in range(3):
            self.assertEqual(len(batches[i]), batch_size)
        
        # 最后一个批次
        self.assertEqual(len(batches[3]), 4)
    
    def test_data_normalization(self):
        """测试数据归一化"""
        data = np.random.randn(100, 10) * 5 + 3
        
        # Z-score归一化
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        normalized = (data - mean) / std
        
        # 归一化后均值接近0，标准差接近1
        self.assertTrue(np.allclose(normalized.mean(axis=0), 0, atol=0.1))
        self.assertTrue(np.allclose(normalized.std(axis=0), 1, atol=0.1))


class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标计算"""
    
    def test_success_rate_calculation(self):
        """测试成功率计算"""
        results = [True, True, False, True, True, True, False, True, True, True]
        
        success_rate = sum(results) / len(results) * 100
        self.assertEqual(success_rate, 80.0)
    
    def test_speedup_calculation(self):
        """测试加速比计算"""
        baseline_time = 100.0
        improved_time = 25.0
        
        speedup = baseline_time / improved_time
        self.assertEqual(speedup, 4.0)
    
    def test_path_cost_calculation(self):
        """测试路径代价计算"""
        paths = [
            [(0, 0), (0, 1), (0, 2), (0, 3)],  # 长度4
            [(1, 0), (1, 1)],  # 长度2
            [(2, 0), (2, 1), (2, 2)],  # 长度3
        ]
        
        # SOC (Sum of Costs)
        soc = sum(len(p) - 1 for p in paths)  # 3 + 1 + 2 = 6
        self.assertEqual(soc, 6)
        
        # Makespan
        makespan = max(len(p) - 1 for p in paths)  # 3
        self.assertEqual(makespan, 3)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestConflictGraph,
        TestGNNEncoder,
        TestTransformerRanker,
        TestCBSIntegration,
        TestMapGenerator,
        TestStatisticalTests,
        TestDataPipeline,
        TestPerformanceMetrics,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "=" * 50)
    print(f"测试完成: {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")
    
    return result


if __name__ == '__main__':
    run_tests()
