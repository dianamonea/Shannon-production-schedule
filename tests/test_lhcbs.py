"""
Unit Tests for L-HCBS Algorithm

Tests cover:
1. Heterogeneous agent creation
2. H-CBS correctness
3. L-HCBS correctness
4. Online replanning
5. GNN conflict predictor
"""

import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.shannon.mapf import (
    # Agents
    AgentType, KinematicConstraints, HeterogeneousAgent,
    HeterogeneousAgentFactory, AgentState,
    # CBS
    GridMap, HeterogeneousCBS, HeterogeneousLowLevelPlanner,
    Conflict, ConflictType, Constraint,
    # L-HCBS
    LearningGuidedHCBS, SearchStatistics,
    # GNN
    ConflictPredictorNetwork, ConflictFeatureExtractor,
    AgentFeatures, EdgeFeatures,
    # Online
    Disruption, DisruptionType, OnlineExecutionManager, PartialReplanner,
    # Experiments
    BenchmarkGenerator, MetricsCalculator
)


class TestHeterogeneousAgent(unittest.TestCase):
    """Tests for heterogeneous agent module."""
    
    def setUp(self):
        self.factory = HeterogeneousAgentFactory()
    
    def test_cnc_creation(self):
        """CNC machines should be static."""
        cnc = self.factory.create_agent('cnc_1', AgentType.CNC, (5.0, 5.0))
        
        self.assertEqual(cnc.agent_type, AgentType.CNC)
        self.assertEqual(cnc.position, (5.0, 5.0))
        self.assertEqual(cnc.kinematics.max_velocity, 0.0)
        self.assertFalse(cnc.can_move())
    
    def test_agv_creation(self):
        """AGVs should be mobile."""
        agv = self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        
        self.assertEqual(agv.agent_type, AgentType.AGV)
        self.assertTrue(agv.can_move())
        self.assertGreater(agv.kinematics.max_velocity, 0)
    
    def test_robot_creation(self):
        """Robots should have limited workspace."""
        robot = self.factory.create_agent('robot_1', AgentType.ROBOT, (10.0, 10.0))
        
        self.assertEqual(robot.agent_type, AgentType.ROBOT)
        self.assertTrue(robot.can_move())
        self.assertGreater(robot.kinematics.workspace_radius, 0)
    
    def test_movement_cost_static(self):
        """Static agents should have infinite cost for movement."""
        cnc = self.factory.create_agent('cnc_1', AgentType.CNC, (5.0, 5.0))
        
        cost = cnc.movement_cost((5.0, 5.0), (6.0, 5.0))
        self.assertEqual(cost, float('inf'))
        
        cost = cnc.movement_cost((5.0, 5.0), (5.0, 5.0))
        self.assertEqual(cost, 0.0)
    
    def test_movement_cost_mobile(self):
        """Mobile agents should have finite movement cost."""
        agv = self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        
        cost = agv.movement_cost((0.0, 0.0), (1.0, 0.0))
        self.assertGreater(cost, 0)
        self.assertLess(cost, float('inf'))


class TestGridMap(unittest.TestCase):
    """Tests for grid map."""
    
    def test_empty_map(self):
        """Empty map should be passable everywhere."""
        grid = GridMap(10, 10)
        
        self.assertTrue(grid.in_bounds(0, 0))
        self.assertTrue(grid.in_bounds(9, 9))
        self.assertFalse(grid.in_bounds(-1, 0))
        self.assertFalse(grid.in_bounds(10, 10))
        
        self.assertTrue(grid.passable(5, 5))
    
    def test_obstacles(self):
        """Obstacles should block positions."""
        obstacles = {(5, 5), (5, 6), (6, 5)}
        grid = GridMap(10, 10, obstacles)
        
        self.assertFalse(grid.passable(5, 5))
        self.assertTrue(grid.passable(0, 0))
    
    def test_neighbors(self):
        """Should return valid neighbors."""
        grid = GridMap(10, 10)
        
        neighbors = grid.neighbors(5, 5)
        self.assertIn((5, 5), neighbors)  # Wait
        self.assertIn((5, 6), neighbors)
        self.assertIn((5, 4), neighbors)
        self.assertIn((4, 5), neighbors)
        self.assertIn((6, 5), neighbors)


class TestHeterogeneousCBS(unittest.TestCase):
    """Tests for H-CBS algorithm."""
    
    def setUp(self):
        self.factory = HeterogeneousAgentFactory()
    
    def test_single_agent(self):
        """Single agent should find straight path."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        }
        
        solver = HeterogeneousCBS(grid, agents)
        paths = solver.solve(
            starts={'agv_1': (0, 0)},
            goals={'agv_1': (5, 0)}
        )
        
        self.assertIsNotNone(paths)
        self.assertEqual(paths['agv_1'][0], (0, 0))
        self.assertEqual(paths['agv_1'][-1], (5, 0))
    
    def test_two_agents_no_conflict(self):
        """Two agents with no conflict should find paths."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0)),
            'agv_2': self.factory.create_agent('agv_2', AgentType.AGV, (0.0, 9.0))
        }
        
        solver = HeterogeneousCBS(grid, agents)
        paths = solver.solve(
            starts={'agv_1': (0, 0), 'agv_2': (0, 9)},
            goals={'agv_1': (9, 0), 'agv_2': (9, 9)}
        )
        
        self.assertIsNotNone(paths)
        self.assertEqual(len(paths), 2)
    
    def test_two_agents_head_on(self):
        """Two agents head-on should resolve conflict."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 5.0)),
            'agv_2': self.factory.create_agent('agv_2', AgentType.AGV, (9.0, 5.0))
        }
        
        solver = HeterogeneousCBS(grid, agents)
        paths = solver.solve(
            starts={'agv_1': (0, 5), 'agv_2': (9, 5)},
            goals={'agv_1': (9, 5), 'agv_2': (0, 5)}
        )
        
        self.assertIsNotNone(paths)
        
        # Verify no collision
        max_len = max(len(p) for p in paths.values())
        for t in range(max_len):
            pos1 = paths['agv_1'][min(t, len(paths['agv_1'])-1)]
            pos2 = paths['agv_2'][min(t, len(paths['agv_2'])-1)]
            self.assertNotEqual(pos1, pos2, f"Collision at time {t}")
    
    def test_conflict_detection(self):
        """Should detect vertex conflicts."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0)),
            'agv_2': self.factory.create_agent('agv_2', AgentType.AGV, (0.0, 0.0))
        }
        
        solver = HeterogeneousCBS(grid, agents)
        
        # Both at same position
        paths = {
            'agv_1': [(0, 0), (1, 0), (2, 0)],
            'agv_2': [(0, 0), (1, 0), (2, 0)]
        }
        
        conflicts = solver.detect_conflicts(paths)
        self.assertGreater(len(conflicts), 0)


class TestLearningGuidedCBS(unittest.TestCase):
    """Tests for L-HCBS algorithm."""
    
    def setUp(self):
        self.factory = HeterogeneousAgentFactory()
    
    def test_basic_solve(self):
        """L-HCBS should solve basic problems."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0)),
            'agv_2': self.factory.create_agent('agv_2', AgentType.AGV, (9.0, 9.0))
        }
        
        solver = LearningGuidedHCBS(grid, agents, use_learning=True)
        paths = solver.solve(
            starts={'agv_1': (0, 0), 'agv_2': (9, 9)},
            goals={'agv_1': (9, 9), 'agv_2': (0, 0)}
        )
        
        self.assertIsNotNone(paths)
    
    def test_learning_disabled(self):
        """Should work with learning disabled."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        }
        
        solver = LearningGuidedHCBS(grid, agents, use_learning=False)
        paths = solver.solve(
            starts={'agv_1': (0, 0)},
            goals={'agv_1': (5, 5)}
        )
        
        self.assertIsNotNone(paths)
    
    def test_statistics_collected(self):
        """Should collect search statistics."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        }
        
        solver = LearningGuidedHCBS(grid, agents, use_learning=True)
        solver.solve(
            starts={'agv_1': (0, 0)},
            goals={'agv_1': (5, 5)}
        )
        
        self.assertGreaterEqual(solver.stats.iterations, 1)
        self.assertGreaterEqual(solver.stats.time_elapsed, 0)


class TestGNNConflictPredictor(unittest.TestCase):
    """Tests for GNN conflict prediction."""
    
    def test_network_forward(self):
        """GNN should produce predictions."""
        network = ConflictPredictorNetwork(hidden_dim=32)
        
        # Create simple features
        node_features = [
            AgentFeatures(
                position=(0.0, 0.0), goal=(5.0, 5.0),
                velocity=1.0, agent_type_onehot=[0, 1, 0],
                path_length=10, remaining_distance=7.07,
                congestion_level=0.5
            ),
            AgentFeatures(
                position=(5.0, 5.0), goal=(0.0, 0.0),
                velocity=1.0, agent_type_onehot=[0, 1, 0],
                path_length=10, remaining_distance=7.07,
                congestion_level=0.5
            )
        ]
        
        edge_features = {
            (0, 1): EdgeFeatures(
                distance=7.07, relative_velocity=0.0,
                path_intersection_count=2,
                time_to_closest_approach=5.0,
                collision_risk=0.5
            )
        }
        
        edges = [(0, 1)]
        
        predictions = network.forward(node_features, edge_features, edges)
        
        self.assertIn((0, 1), predictions)
        self.assertGreaterEqual(predictions[(0, 1)], 0.0)
        self.assertLessEqual(predictions[(0, 1)], 1.0)
    
    def test_feature_extraction(self):
        """Feature extractor should produce valid features."""
        extractor = ConflictFeatureExtractor(20, 20)
        
        agents = {
            'agv_1': {'type': 'AGV', 'velocity': 1.0},
            'agv_2': {'type': 'AGV', 'velocity': 1.0}
        }
        positions = {'agv_1': (0, 0), 'agv_2': (10, 10)}
        goals = {'agv_1': (10, 10), 'agv_2': (0, 0)}
        
        node_feats, edge_feats, edges, id_map = extractor.extract_features(
            agents, positions, goals
        )
        
        self.assertEqual(len(node_feats), 2)
        self.assertEqual(len(edges), 1)


class TestOnlineReplanning(unittest.TestCase):
    """Tests for online replanning."""
    
    def setUp(self):
        self.factory = HeterogeneousAgentFactory()
    
    def test_execution_without_disruption(self):
        """Should complete execution without disruptions."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        }
        
        solver = LearningGuidedHCBS(grid, agents, use_learning=False)
        executor = OnlineExecutionManager(solver, agents)
        
        starts = {'agv_1': (0, 0)}
        goals = {'agv_1': (5, 5)}
        
        self.assertTrue(executor.initialize(starts, goals))
        success = executor.execute_full(max_steps=100)
        
        self.assertTrue(success)
    
    def test_disruption_handling(self):
        """Should handle path blocked disruption."""
        grid = GridMap(10, 10)
        agents = {
            'agv_1': self.factory.create_agent('agv_1', AgentType.AGV, (0.0, 0.0))
        }
        
        solver = LearningGuidedHCBS(grid, agents, use_learning=False)
        executor = OnlineExecutionManager(solver, agents)
        
        starts = {'agv_1': (0, 0)}
        goals = {'agv_1': (9, 0)}
        
        executor.initialize(starts, goals)
        
        # Add disruption at step 3
        disruption = Disruption(
            disruption_type=DisruptionType.PATH_BLOCKED,
            timestamp=3.0,
            affected_agents=[],
            affected_positions=[(5, 0)],
            duration=100.0  # Long duration
        )
        executor.add_disruption(disruption, trigger_time=3)
        
        success = executor.execute_full(max_steps=100)
        
        stats = executor.get_statistics()
        self.assertGreaterEqual(stats['replanning_count'], 1)


class TestBenchmarkGenerator(unittest.TestCase):
    """Tests for benchmark generation."""
    
    def test_manufacturing_floor(self):
        """Should generate manufacturing floor benchmark."""
        generator = BenchmarkGenerator(seed=42)
        instance = generator.generate_manufacturing_floor(
            width=20, height=20,
            num_cnc=2, num_agv=3, num_robot=1
        )
        
        self.assertEqual(instance.grid_map.width, 20)
        self.assertEqual(instance.grid_map.height, 20)
        self.assertGreater(len(instance.agents), 0)
        self.assertEqual(len(instance.starts), len(instance.agents))
        self.assertEqual(len(instance.goals), len(instance.agents))
    
    def test_warehouse(self):
        """Should generate warehouse benchmark."""
        generator = BenchmarkGenerator(seed=42)
        instance = generator.generate_warehouse(
            width=30, height=30,
            num_agents=5
        )
        
        self.assertEqual(len(instance.agents), 5)
    
    def test_bottleneck(self):
        """Should generate bottleneck benchmark."""
        generator = BenchmarkGenerator(seed=42)
        instance = generator.generate_bottleneck(
            width=20, height=20,
            num_agents=4,
            bottleneck_width=1
        )
        
        self.assertGreater(len(instance.grid_map.obstacles), 0)


class TestMetricsCalculator(unittest.TestCase):
    """Tests for metrics calculation."""
    
    def test_makespan(self):
        """Should compute correct makespan."""
        paths = {
            'a1': [(0, 0), (1, 0), (2, 0)],
            'a2': [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
        }
        
        makespan = MetricsCalculator.compute_makespan(paths)
        self.assertEqual(makespan, 4)  # Longest path - 1
    
    def test_sum_of_costs(self):
        """Should compute correct sum of costs."""
        paths = {
            'a1': [(0, 0), (1, 0), (2, 0)],  # Cost: 2
            'a2': [(0, 1), (1, 1)]  # Cost: 1
        }
        
        soc = MetricsCalculator.compute_sum_of_costs(paths)
        self.assertEqual(soc, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
