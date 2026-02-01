"""
L-HCBS Demo: Learning-guided Heterogeneous CBS for Manufacturing MAPF

This script demonstrates the key features of L-HCBS:
1. Heterogeneous agents (CNC, AGV, Robot) with different kinematics
2. Learning-guided conflict resolution
3. Online replanning with disruptions
4. Comparison with baseline algorithms

For paper submission to IROS 2026 / ICRA 2027.
"""

import sys
import time
import os

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.shannon.mapf import (
    # Agents
    AgentType, HeterogeneousAgentFactory, AgentDict,
    # Environment
    GridMap,
    # Algorithms
    HeterogeneousCBS, LearningGuidedHCBS,
    # Online
    Disruption, DisruptionType, OnlineExecutionManager,
    # Experiments
    BenchmarkGenerator, ExperimentRunner, MetricsCalculator,
    PrioritizedPlanning, StandardCBS
)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, value, unit: str = ""):
    """Print formatted result."""
    if isinstance(value, float):
        print(f"  {name}: {value:.3f} {unit}")
    else:
        print(f"  {name}: {value} {unit}")


def demo_heterogeneous_agents():
    """Demonstrate heterogeneous agent creation."""
    print_section("1. HETEROGENEOUS AGENT TYPES")
    
    factory = HeterogeneousAgentFactory()
    
    # Create different agent types
    agents = {
        'cnc_1': factory.create_agent('cnc_1', AgentType.CNC, position=(5.0, 5.0)),
        'cnc_2': factory.create_agent('cnc_2', AgentType.CNC, position=(15.0, 5.0)),
        'agv_1': factory.create_agent('agv_1', AgentType.AGV, position=(0.0, 0.0)),
        'agv_2': factory.create_agent('agv_2', AgentType.AGV, position=(10.0, 10.0)),
        'robot_1': factory.create_agent('robot_1', AgentType.ROBOT, position=(8.0, 8.0)),
    }
    
    print("\n  Agent Configurations:")
    print("  " + "-" * 60)
    
    for aid, agent in agents.items():
        print(f"\n  {aid} ({agent.agent_type.value}):")
        print(f"    Position: {agent.position}")
        print(f"    Max Velocity: {agent.kinematics.max_velocity}")
        print(f"    Max Acceleration: {agent.kinematics.max_acceleration}")
        print(f"    Holonomic: {agent.kinematics.is_holonomic}")
        print(f"    Can Move: {agent.can_move()}")
    
    return agents


def demo_basic_mapf(agents: AgentDict):
    """Demonstrate basic MAPF solving."""
    print_section("2. HETEROGENEOUS CBS (H-CBS)")
    
    # Create simple grid
    grid = GridMap(20, 20, obstacles={(5, 5), (5, 6), (6, 5), (6, 6)})
    
    # Define start and goal positions
    starts = {
        'agv_1': (0, 0),
        'agv_2': (19, 19),
    }
    goals = {
        'agv_1': (19, 19),
        'agv_2': (0, 0),
    }
    
    # Filter to AGV agents only
    agv_agents = {k: v for k, v in agents.items() if k.startswith('agv')}
    
    print("\n  Problem Setup:")
    for aid in starts:
        print(f"    {aid}: {starts[aid]} -> {goals[aid]}")
    
    # Solve with H-CBS
    solver = HeterogeneousCBS(grid, agv_agents)
    
    start_time = time.time()
    paths = solver.solve(starts, goals)
    elapsed = time.time() - start_time
    
    if paths:
        print("\n  H-CBS Solution Found:")
        print_result("Solve Time", elapsed, "s")
        print_result("Sum of Costs", MetricsCalculator.compute_sum_of_costs(paths))
        print_result("Makespan", MetricsCalculator.compute_makespan(paths))
        
        print("\n  Paths:")
        for aid, path in paths.items():
            print(f"    {aid}: {path[:5]}... (length={len(path)})")
    else:
        print("\n  No solution found!")
    
    return paths


def demo_learning_guided_cbs(agents: AgentDict):
    """Demonstrate L-HCBS with learning."""
    print_section("3. LEARNING-GUIDED H-CBS (L-HCBS)")
    
    # Create larger grid
    obstacles = set()
    for i in range(5, 15):
        obstacles.add((i, 10))  # Horizontal wall
    obstacles.discard((10, 10))  # Gap in wall
    
    grid = GridMap(25, 25, obstacles)
    
    # More challenging problem
    agv_agents = {k: v for k, v in agents.items() if k.startswith('agv')}
    
    starts = {'agv_1': (0, 5), 'agv_2': (0, 15)}
    goals = {'agv_1': (24, 15), 'agv_2': (24, 5)}
    
    print("\n  Problem: Agents must cross through narrow passage")
    print(f"  Grid: 25x25 with wall at y=10 (gap at x=10)")
    
    # Compare with and without learning
    results = {}
    
    for use_learning in [False, True]:
        solver = LearningGuidedHCBS(grid, agv_agents, use_learning=use_learning)
        
        start_time = time.time()
        paths = solver.solve(starts, goals, max_iterations=5000)
        elapsed = time.time() - start_time
        
        label = "With Learning" if use_learning else "Without Learning"
        results[label] = {
            'success': paths is not None,
            'time': elapsed,
            'iterations': solver.stats.iterations,
            'nodes_expanded': solver.stats.nodes_expanded,
            'cost': solver.stats.solution_cost if paths else 0
        }
    
    print("\n  Comparison Results:")
    print("  " + "-" * 50)
    print(f"  {'Metric':<25} {'No Learning':>12} {'Learning':>12}")
    print("  " + "-" * 50)
    
    no_learn = results["Without Learning"]
    with_learn = results["With Learning"]
    
    print(f"  {'Success':<25} {str(no_learn['success']):>12} {str(with_learn['success']):>12}")
    print(f"  {'Solve Time (s)':<25} {no_learn['time']:>12.4f} {with_learn['time']:>12.4f}")
    print(f"  {'Iterations':<25} {no_learn['iterations']:>12} {with_learn['iterations']:>12}")
    print(f"  {'Nodes Expanded':<25} {no_learn['nodes_expanded']:>12} {with_learn['nodes_expanded']:>12}")
    print(f"  {'Solution Cost':<25} {no_learn['cost']:>12.0f} {with_learn['cost']:>12.0f}")


def demo_online_replanning(agents: AgentDict):
    """Demonstrate online replanning with disruptions."""
    print_section("4. ONLINE REPLANNING WITH DISRUPTIONS")
    
    grid = GridMap(20, 20)
    agv_agents = {k: v for k, v in agents.items() if k.startswith('agv')}
    
    starts = {'agv_1': (0, 0), 'agv_2': (19, 0)}
    goals = {'agv_1': (19, 19), 'agv_2': (0, 19)}
    
    # Create solver and execution manager
    solver = LearningGuidedHCBS(grid, agv_agents, use_learning=True)
    executor = OnlineExecutionManager(solver, agv_agents)
    
    # Initialize
    print("\n  Initializing execution...")
    if not executor.initialize(starts, goals):
        print("  Failed to find initial plan!")
        return
    
    print("  Initial plan computed.")
    
    # Add disruption
    disruption = Disruption(
        disruption_type=DisruptionType.PATH_BLOCKED,
        timestamp=5.0,
        affected_agents=[],
        affected_positions=[(10, 10)],
        duration=10.0
    )
    executor.add_disruption(disruption, trigger_time=5)
    
    print(f"\n  Disruption scheduled at t=5:")
    print(f"    Type: {disruption.disruption_type.value}")
    print(f"    Blocked: {disruption.affected_positions}")
    print(f"    Duration: {disruption.duration}")
    
    # Execute
    print("\n  Executing with disruption handling...")
    success = executor.execute_full(max_steps=200)
    
    stats = executor.get_statistics()
    print("\n  Execution Results:")
    print_result("Success", success)
    print_result("Total Time Steps", stats['total_time_steps'])
    print_result("Replanning Count", stats['replanning_count'])
    print_result("Avg Replan Time", stats['avg_replanning_time'], "s")
    print_result("Disruptions Handled", stats['disruptions_handled'])


def demo_benchmark_experiments():
    """Run benchmark experiments."""
    print_section("5. BENCHMARK EXPERIMENTS")
    
    generator = BenchmarkGenerator(seed=42)
    runner = ExperimentRunner()
    
    # Generate instances
    print("\n  Generating benchmark instances...")
    
    instances = [
        generator.generate_manufacturing_floor(
            width=20, height=20,
            num_cnc=2, num_agv=4, num_robot=1
        ),
        generator.generate_warehouse(
            width=25, height=25,
            num_agents=6
        ),
        generator.generate_bottleneck(
            width=20, height=20,
            num_agents=4,
            bottleneck_width=1
        ),
    ]
    
    print(f"  Generated {len(instances)} instances:")
    for inst in instances:
        print(f"    - {inst.name}: {len(inst.agents)} agents")
    
    # Run experiments
    print("\n  Running algorithm comparison...")
    print("  " + "-" * 60)
    
    all_results = {}
    for inst in instances:
        print(f"\n  Instance: {inst.name}")
        results = runner.run_comparison(inst, timeout=30.0)
        all_results[inst.name] = results
        
        # Print results for this instance
        for algo, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"    {status} {result.algorithm}: {result.runtime:.3f}s, cost={result.solution_cost:.0f}")
    
    # Summary
    print("\n" + "-" * 60)
    print("  SUMMARY BY ALGORITHM:")
    print("-" * 60)
    
    algo_stats = {}
    for inst_name, results in all_results.items():
        for algo, result in results.items():
            if algo not in algo_stats:
                algo_stats[algo] = {'success': 0, 'total': 0, 'runtime': 0, 'cost': 0}
            algo_stats[algo]['total'] += 1
            if result.success:
                algo_stats[algo]['success'] += 1
                algo_stats[algo]['runtime'] += result.runtime
                algo_stats[algo]['cost'] += result.solution_cost
    
    print(f"\n  {'Algorithm':<25} {'Success':>10} {'Avg Time':>12} {'Avg Cost':>12}")
    print("  " + "-" * 60)
    
    for algo, stats in algo_stats.items():
        success_rate = f"{stats['success']}/{stats['total']}"
        avg_time = stats['runtime'] / max(1, stats['success'])
        avg_cost = stats['cost'] / max(1, stats['success'])
        print(f"  {algo:<25} {success_rate:>10} {avg_time:>12.4f} {avg_cost:>12.1f}")


def demo_paper_contribution_summary():
    """Summarize paper contributions."""
    print_section("6. PAPER CONTRIBUTION SUMMARY")
    
    print("""
  Title: L-HCBS: Learning-guided Heterogeneous Conflict-Based Search
         for Multi-Agent Path Finding in Manufacturing Systems
  
  Target: IROS 2026 / ICRA 2027
  
  Key Contributions:
  
  1. HETEROGENEOUS MAPF FORMULATION
     - First to address MAPF with CNC + AGV + Robot arms
     - Different kinematic constraints per agent type
     - Type-aware collision detection
  
  2. LEARNING-GUIDED CBS
     - GNN-based conflict prediction
     - Prioritized conflict resolution (fail-fast)
     - Online learning from search experience
  
  3. ONLINE REPLANNING
     - Handle runtime disruptions (breakdowns, urgent orders)
     - Partial replanning preserves valid paths
     - Bounded replanning time guarantees
  
  4. THEORETICAL ANALYSIS
     - Completeness: Same as CBS (guaranteed if solution exists)
     - Bounded Suboptimality: Solution cost ≤ w × optimal
     - Convergence: GNN improves with more experience
  
  Experimental Setup:
  - Baselines: Prioritized Planning, Standard CBS, H-CBS, ECBS
  - Benchmarks: Manufacturing floor, Warehouse, Bottleneck
  - Metrics: Success rate, Runtime, Solution cost, Iterations
  
  Expected Results:
  - 2-5x speedup over H-CBS on large instances
  - Maintain solution quality (≤5% suboptimality)
  - Handle 10+ agents with disruptions
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  L-HCBS: Learning-guided Heterogeneous CBS Demo")
    print("  Multi-Agent Path Finding for Manufacturing Systems")
    print("=" * 70)
    
    # 1. Create heterogeneous agents
    agents = demo_heterogeneous_agents()
    
    # 2. Basic MAPF
    demo_basic_mapf(agents)
    
    # 3. Learning-guided CBS
    demo_learning_guided_cbs(agents)
    
    # 4. Online replanning
    demo_online_replanning(agents)
    
    # 5. Benchmark experiments
    demo_benchmark_experiments()
    
    # 6. Paper contribution summary
    demo_paper_contribution_summary()
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("  See python/shannon/mapf/ for full implementation")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
