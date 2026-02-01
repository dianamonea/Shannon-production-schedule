"""
Experiment Framework for L-HCBS

Provides:
1. Benchmark instance generators
2. Baseline algorithm implementations
3. Evaluation metrics
4. Statistical analysis

For paper experiments comparing L-HCBS against baselines.
"""

import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import copy
import json

from .heterogeneous_agent import (
    HeterogeneousAgent, AgentType, KinematicConstraints,
    HeterogeneousAgentFactory
)
from .heterogeneous_cbs import GridMap, HeterogeneousCBS
from .lhcbs import LearningGuidedHCBS, SearchStatistics


# ============================================================
# Benchmark Instance Generators
# ============================================================

@dataclass
class BenchmarkInstance:
    """A single benchmark instance."""
    name: str
    grid_map: GridMap
    agents: Dict[str, HeterogeneousAgent]
    starts: Dict[str, Tuple[int, int]]
    goals: Dict[str, Tuple[int, int]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkType(Enum):
    """Types of benchmark scenarios."""
    MANUFACTURING_FLOOR = "manufacturing"
    WAREHOUSE = "warehouse"
    RANDOM_GRID = "random"
    BOTTLENECK = "bottleneck"
    INTERSECTION = "intersection"


class BenchmarkGenerator:
    """Generates benchmark instances for experiments."""
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.factory = HeterogeneousAgentFactory()
    
    def generate_manufacturing_floor(self, 
                                    width: int = 30, 
                                    height: int = 30,
                                    num_cnc: int = 4,
                                    num_agv: int = 5,
                                    num_robot: int = 2) -> BenchmarkInstance:
        """
        Generate manufacturing floor layout.
        
        Layout:
        - CNC machines along walls (static)
        - AGVs transport materials
        - Robots at workstations
        - Aisles between machines
        """
        # Create grid with obstacles
        obstacles = set()
        
        # Machine positions (obstacles for movement)
        machine_positions = []
        for i in range(4):
            for j in range(4):
                mx = 5 + i * 6
                my = 5 + j * 6
                obstacles.add((mx, my))
                obstacles.add((mx+1, my))
                obstacles.add((mx, my+1))
                obstacles.add((mx+1, my+1))
                machine_positions.append((mx, my))
        
        grid_map = GridMap(width, height, obstacles)
        
        # Create agents
        agents = {}
        starts = {}
        goals = {}
        
        # CNC machines (static at machine positions)
        for i in range(min(num_cnc, len(machine_positions))):
            agent_id = f"cnc_{i}"
            pos = machine_positions[i]
            # Place adjacent to machine
            agent_pos = (pos[0] - 1, pos[1])
            
            agents[agent_id] = self.factory.create_agent(
                agent_id, AgentType.CNC,
                position=(float(agent_pos[0]), float(agent_pos[1]))
            )
            starts[agent_id] = agent_pos
            goals[agent_id] = agent_pos  # CNC stays in place
        
        # AGVs
        for i in range(num_agv):
            agent_id = f"agv_{i}"
            
            # Random start in aisles
            while True:
                sx = self.random.randint(0, width-1)
                sy = self.random.randint(0, height-1)
                if (sx, sy) not in obstacles:
                    break
            
            # Random goal
            while True:
                gx = self.random.randint(0, width-1)
                gy = self.random.randint(0, height-1)
                if (gx, gy) not in obstacles and (gx, gy) != (sx, sy):
                    break
            
            agents[agent_id] = self.factory.create_agent(
                agent_id, AgentType.AGV,
                position=(float(sx), float(sy))
            )
            starts[agent_id] = (sx, sy)
            goals[agent_id] = (gx, gy)
        
        # Robot arms
        for i in range(num_robot):
            agent_id = f"robot_{i}"
            
            # Place near machines
            machine_idx = i % len(machine_positions)
            pos = machine_positions[machine_idx]
            agent_pos = (pos[0] + 2, pos[1])
            
            if agent_pos not in obstacles:
                agents[agent_id] = self.factory.create_agent(
                    agent_id, AgentType.ROBOT,
                    position=(float(agent_pos[0]), float(agent_pos[1]))
                )
                starts[agent_id] = agent_pos
                goals[agent_id] = agent_pos  # Robots have limited workspace
        
        return BenchmarkInstance(
            name=f"manufacturing_{num_cnc}c_{num_agv}a_{num_robot}r",
            grid_map=grid_map,
            agents=agents,
            starts=starts,
            goals=goals,
            metadata={
                'type': BenchmarkType.MANUFACTURING_FLOOR.value,
                'width': width, 'height': height,
                'num_cnc': num_cnc, 'num_agv': num_agv, 'num_robot': num_robot,
                'obstacle_density': len(obstacles) / (width * height)
            }
        )
    
    def generate_warehouse(self,
                          width: int = 40,
                          height: int = 40,
                          num_agents: int = 10,
                          shelf_rows: int = 5) -> BenchmarkInstance:
        """Generate warehouse-style benchmark."""
        obstacles = set()
        
        # Shelf rows
        for row in range(shelf_rows):
            y = 5 + row * 7
            for x in range(5, width - 5, 3):
                obstacles.add((x, y))
                obstacles.add((x, y+1))
                obstacles.add((x, y+2))
        
        grid_map = GridMap(width, height, obstacles)
        
        agents = {}
        starts = {}
        goals = {}
        
        # Mostly AGVs for warehouse
        for i in range(num_agents):
            agent_id = f"agv_{i}"
            
            while True:
                sx = self.random.randint(0, width-1)
                sy = self.random.randint(0, height-1)
                if (sx, sy) not in obstacles:
                    break
            
            while True:
                gx = self.random.randint(0, width-1)
                gy = self.random.randint(0, height-1)
                if (gx, gy) not in obstacles and (gx, gy) != (sx, sy):
                    break
            
            agents[agent_id] = self.factory.create_agent(
                agent_id, AgentType.AGV,
                position=(float(sx), float(sy))
            )
            starts[agent_id] = (sx, sy)
            goals[agent_id] = (gx, gy)
        
        return BenchmarkInstance(
            name=f"warehouse_{num_agents}",
            grid_map=grid_map,
            agents=agents,
            starts=starts,
            goals=goals,
            metadata={
                'type': BenchmarkType.WAREHOUSE.value,
                'width': width, 'height': height,
                'num_agents': num_agents
            }
        )
    
    def generate_bottleneck(self,
                           width: int = 30,
                           height: int = 30,
                           num_agents: int = 8,
                           bottleneck_width: int = 2) -> BenchmarkInstance:
        """Generate scenario with bottleneck that forces conflicts."""
        obstacles = set()
        
        # Create wall with narrow passage
        wall_y = height // 2
        for x in range(width):
            if not (width // 2 - bottleneck_width <= x <= width // 2 + bottleneck_width):
                obstacles.add((x, wall_y))
        
        grid_map = GridMap(width, height, obstacles)
        
        agents = {}
        starts = {}
        goals = {}
        
        # Place agents on both sides needing to cross
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            
            # Alternate sides
            if i % 2 == 0:
                sx = self.random.randint(0, width-1)
                sy = self.random.randint(0, wall_y - 2)
                gy = self.random.randint(wall_y + 2, height - 1)
            else:
                sx = self.random.randint(0, width-1)
                sy = self.random.randint(wall_y + 2, height - 1)
                gy = self.random.randint(0, wall_y - 2)
            
            gx = self.random.randint(0, width-1)
            
            if (sx, sy) not in obstacles and (gx, gy) not in obstacles:
                agents[agent_id] = self.factory.create_agent(
                    agent_id, AgentType.AGV,
                    position=(float(sx), float(sy))
                )
                starts[agent_id] = (sx, sy)
                goals[agent_id] = (gx, gy)
        
        return BenchmarkInstance(
            name=f"bottleneck_{num_agents}",
            grid_map=grid_map,
            agents=agents,
            starts=starts,
            goals=goals,
            metadata={
                'type': BenchmarkType.BOTTLENECK.value,
                'bottleneck_width': bottleneck_width
            }
        )
    
    def generate_random_instance(self,
                                width: int = 20,
                                height: int = 20,
                                num_agents: int = 5,
                                obstacle_ratio: float = 0.1,
                                agent_type_mix: Dict[AgentType, float] = None
                                ) -> BenchmarkInstance:
        """Generate random benchmark instance."""
        # Random obstacles
        obstacles = set()
        num_obstacles = int(width * height * obstacle_ratio)
        
        while len(obstacles) < num_obstacles:
            x = self.random.randint(0, width-1)
            y = self.random.randint(0, height-1)
            obstacles.add((x, y))
        
        grid_map = GridMap(width, height, obstacles)
        
        # Default agent mix
        if agent_type_mix is None:
            agent_type_mix = {
                AgentType.AGV: 0.6,
                AgentType.ROBOT: 0.2,
                AgentType.CNC: 0.2
            }
        
        agents = {}
        starts = {}
        goals = {}
        
        for i in range(num_agents):
            # Select agent type
            r = self.random.random()
            cumsum = 0
            agent_type = AgentType.AGV
            for at, prob in agent_type_mix.items():
                cumsum += prob
                if r < cumsum:
                    agent_type = at
                    break
            
            agent_id = f"{agent_type.value}_{i}"
            
            while True:
                sx = self.random.randint(0, width-1)
                sy = self.random.randint(0, height-1)
                if (sx, sy) not in obstacles and (sx, sy) not in starts.values():
                    break
            
            while True:
                gx = self.random.randint(0, width-1)
                gy = self.random.randint(0, height-1)
                if ((gx, gy) not in obstacles and (gx, gy) != (sx, sy) 
                    and (gx, gy) not in goals.values()):
                    break
            
            agents[agent_id] = self.factory.create_agent(
                agent_id, agent_type,
                position=(float(sx), float(sy))
            )
            starts[agent_id] = (sx, sy)
            goals[agent_id] = (gx, gy)
        
        return BenchmarkInstance(
            name=f"random_{width}x{height}_{num_agents}",
            grid_map=grid_map,
            agents=agents,
            starts=starts,
            goals=goals,
            metadata={
                'type': BenchmarkType.RANDOM_GRID.value,
                'obstacle_ratio': obstacle_ratio
            }
        )


# ============================================================
# Evaluation Metrics
# ============================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    instance_name: str
    algorithm: str
    success: bool
    runtime: float
    solution_cost: float
    makespan: int  # Length of longest path
    iterations: int
    nodes_expanded: int
    nodes_generated: int
    path_lengths: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """Calculates evaluation metrics."""
    
    @staticmethod
    def compute_makespan(paths: Dict[str, List[Tuple[int, int]]]) -> int:
        """Maximum path length (time to complete all tasks)."""
        if not paths:
            return 0
        return max(len(p) - 1 for p in paths.values())
    
    @staticmethod
    def compute_sum_of_costs(paths: Dict[str, List[Tuple[int, int]]]) -> int:
        """Sum of all path lengths."""
        return sum(len(p) - 1 for p in paths.values())
    
    @staticmethod
    def compute_average_path_length(paths: Dict[str, List[Tuple[int, int]]]) -> float:
        """Average path length."""
        if not paths:
            return 0.0
        return sum(len(p) - 1 for p in paths.values()) / len(paths)
    
    @staticmethod
    def compute_path_efficiency(paths: Dict[str, List[Tuple[int, int]]],
                               starts: Dict[str, Tuple[int, int]],
                               goals: Dict[str, Tuple[int, int]]) -> float:
        """Ratio of optimal (straight-line) to actual path length."""
        total_optimal = 0
        total_actual = 0
        
        for aid, path in paths.items():
            if aid in starts and aid in goals:
                start = starts[aid]
                goal = goals[aid]
                optimal = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                actual = len(path) - 1
                
                total_optimal += optimal
                total_actual += actual
        
        if total_actual == 0:
            return 1.0
        return total_optimal / total_actual


# ============================================================
# Baseline Algorithms
# ============================================================

class PrioritizedPlanning:
    """
    Baseline: Simple prioritized planning.
    Plans agents one by one in priority order.
    """
    
    def __init__(self, grid_map: GridMap, agents: Dict[str, HeterogeneousAgent]):
        self.grid = grid_map
        self.agents = agents
        from .heterogeneous_cbs import HeterogeneousLowLevelPlanner
        self.LowLevelPlanner = HeterogeneousLowLevelPlanner
    
    def solve(self, starts: Dict[str, Tuple[int, int]],
              goals: Dict[str, Tuple[int, int]],
              priority_order: Optional[List[str]] = None
              ) -> Tuple[Optional[Dict[str, List[Tuple[int, int]]]], SearchStatistics]:
        """
        Solve using prioritized planning.
        
        Returns:
            (paths, statistics)
        """
        stats = SearchStatistics()
        start_time = time.time()
        
        if priority_order is None:
            priority_order = list(starts.keys())
        
        paths = {}
        reserved_positions: Dict[Tuple[int, int, int], str] = {}  # (x, y, t) -> agent_id
        
        for agent_id in priority_order:
            if agent_id not in starts or agent_id not in goals:
                continue
            
            agent = self.agents[agent_id]
            planner = self.LowLevelPlanner(self.grid, agent)
            
            # Create constraints from reserved positions
            from .heterogeneous_cbs import Constraint
            constraints = []
            for (x, y, t), _ in reserved_positions.items():
                constraints.append(Constraint(
                    agent_id=agent_id,
                    time=t,
                    position=(x, y),
                    is_vertex=True
                ))
            
            path = planner.plan(starts[agent_id], goals[agent_id], constraints)
            
            if path is None:
                stats.time_elapsed = time.time() - start_time
                return None, stats
            
            paths[agent_id] = path
            stats.iterations += 1
            
            # Reserve positions
            for t, pos in enumerate(path):
                reserved_positions[(pos[0], pos[1], t)] = agent_id
        
        stats.time_elapsed = time.time() - start_time
        stats.solution_cost = sum(len(p) - 1 for p in paths.values())
        
        return paths, stats


class StandardCBS:
    """Baseline: Standard CBS (not heterogeneous)."""
    
    def __init__(self, grid_map: GridMap, agents: Dict[str, HeterogeneousAgent]):
        # Use H-CBS but ignore heterogeneity
        self.solver = HeterogeneousCBS(grid_map, agents)
        
        # Set all collision radii to 1.0
        self.solver.collision_radii = {
            (AgentType.AGV, AgentType.AGV): 1.0,
            (AgentType.AGV, AgentType.ROBOT): 1.0,
            (AgentType.ROBOT, AgentType.ROBOT): 1.0,
            (AgentType.CNC, AgentType.AGV): 1.0,
            (AgentType.CNC, AgentType.ROBOT): 1.0,
        }
    
    def solve(self, starts: Dict[str, Tuple[int, int]],
              goals: Dict[str, Tuple[int, int]]
              ) -> Tuple[Optional[Dict[str, List[Tuple[int, int]]]], SearchStatistics]:
        """Solve using standard CBS."""
        stats = SearchStatistics()
        start_time = time.time()
        
        paths = self.solver.solve(starts, goals)
        
        stats.time_elapsed = time.time() - start_time
        if paths:
            stats.solution_cost = sum(len(p) - 1 for p in paths.values())
        
        return paths, stats


# ============================================================
# Experiment Runner
# ============================================================

class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
    
    def run_experiment(self, instance: BenchmarkInstance,
                      algorithm: str,
                      solver: Callable,
                      **kwargs) -> ExperimentResult:
        """Run single experiment."""
        start_time = time.time()
        
        try:
            paths, stats = solver(instance.starts, instance.goals, **kwargs)
            success = paths is not None
            
            if success:
                result = ExperimentResult(
                    instance_name=instance.name,
                    algorithm=algorithm,
                    success=True,
                    runtime=time.time() - start_time,
                    solution_cost=stats.solution_cost if hasattr(stats, 'solution_cost') else 0,
                    makespan=MetricsCalculator.compute_makespan(paths),
                    iterations=stats.iterations if hasattr(stats, 'iterations') else 0,
                    nodes_expanded=stats.nodes_expanded if hasattr(stats, 'nodes_expanded') else 0,
                    nodes_generated=stats.nodes_generated if hasattr(stats, 'nodes_generated') else 0,
                    path_lengths={aid: len(p)-1 for aid, p in paths.items()}
                )
            else:
                result = ExperimentResult(
                    instance_name=instance.name,
                    algorithm=algorithm,
                    success=False,
                    runtime=time.time() - start_time,
                    solution_cost=0,
                    makespan=0,
                    iterations=stats.iterations if hasattr(stats, 'iterations') else 0,
                    nodes_expanded=0,
                    nodes_generated=0
                )
        except Exception as e:
            result = ExperimentResult(
                instance_name=instance.name,
                algorithm=algorithm,
                success=False,
                runtime=time.time() - start_time,
                solution_cost=0,
                makespan=0,
                iterations=0,
                nodes_expanded=0,
                nodes_generated=0,
                metadata={'error': str(e)}
            )
        
        self.results.append(result)
        return result
    
    def run_comparison(self, instance: BenchmarkInstance,
                      timeout: float = 60.0) -> Dict[str, ExperimentResult]:
        """Compare all algorithms on instance."""
        results = {}
        
        # 1. Prioritized Planning
        pp = PrioritizedPlanning(instance.grid_map, instance.agents)
        results['prioritized'] = self.run_experiment(
            instance, 'Prioritized Planning', pp.solve
        )
        
        # 2. Standard CBS
        std_cbs = StandardCBS(instance.grid_map, instance.agents)
        results['standard_cbs'] = self.run_experiment(
            instance, 'Standard CBS', std_cbs.solve
        )
        
        # 3. Heterogeneous CBS
        hcbs = HeterogeneousCBS(instance.grid_map, instance.agents)
        def hcbs_solve(starts, goals):
            paths = hcbs.solve(starts, goals)
            stats = SearchStatistics()
            if paths:
                stats.solution_cost = sum(len(p)-1 for p in paths.values())
            return paths, stats
        results['h_cbs'] = self.run_experiment(
            instance, 'H-CBS', hcbs_solve
        )
        
        # 4. L-HCBS (our method)
        lhcbs = LearningGuidedHCBS(instance.grid_map, instance.agents,
                                   use_learning=True)
        def lhcbs_solve(starts, goals):
            paths = lhcbs.solve(starts, goals, timeout=timeout)
            return paths, lhcbs.stats
        results['l_hcbs'] = self.run_experiment(
            instance, 'L-HCBS (Ours)', lhcbs_solve
        )
        
        # 5. L-HCBS without learning (ablation)
        lhcbs_no_learn = LearningGuidedHCBS(instance.grid_map, instance.agents,
                                            use_learning=False)
        def lhcbs_no_learn_solve(starts, goals):
            paths = lhcbs_no_learn.solve(starts, goals, timeout=timeout)
            return paths, lhcbs_no_learn.stats
        results['l_hcbs_no_learning'] = self.run_experiment(
            instance, 'L-HCBS (No Learning)', lhcbs_no_learn_solve
        )
        
        return results
    
    def generate_report(self) -> str:
        """Generate experiment report."""
        if not self.results:
            return "No results to report."
        
        report_lines = [
            "=" * 80,
            "L-HCBS EXPERIMENT REPORT",
            "=" * 80,
            "",
            f"Total experiments: {len(self.results)}",
            f"Successful: {sum(1 for r in self.results if r.success)}",
            "",
            "-" * 80,
            "Results by Algorithm:",
            "-" * 80,
        ]
        
        # Group by algorithm
        by_algo = {}
        for r in self.results:
            if r.algorithm not in by_algo:
                by_algo[r.algorithm] = []
            by_algo[r.algorithm].append(r)
        
        for algo, results in by_algo.items():
            success_rate = sum(1 for r in results if r.success) / len(results)
            avg_runtime = sum(r.runtime for r in results) / len(results)
            avg_cost = sum(r.solution_cost for r in results if r.success) / max(1, sum(1 for r in results if r.success))
            
            report_lines.extend([
                f"\n{algo}:",
                f"  Success Rate: {success_rate:.1%}",
                f"  Avg Runtime: {avg_runtime:.3f}s",
                f"  Avg Solution Cost: {avg_cost:.1f}",
            ])
        
        return "\n".join(report_lines)
    
    def save_results(self, filename: str):
        """Save results to JSON."""
        data = []
        for r in self.results:
            data.append({
                'instance': r.instance_name,
                'algorithm': r.algorithm,
                'success': r.success,
                'runtime': r.runtime,
                'solution_cost': r.solution_cost,
                'makespan': r.makespan,
                'iterations': r.iterations,
                'nodes_expanded': r.nodes_expanded,
                'path_lengths': r.path_lengths,
                'metadata': r.metadata
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
