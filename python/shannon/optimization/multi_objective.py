"""
Multi-Objective Optimization for Production Scheduling

Implements Pareto-optimal scheduling considering multiple competing objectives:
1. Makespan (minimize total completion time)
2. Cost (minimize production cost)
3. Quality (maximize quality metrics)
4. Utilization (maximize resource utilization)
5. On-Time Delivery (maximize order fulfillment rate)

Uses NSGA-III algorithm for many-objective optimization.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Objective(Enum):
    """Optimization objectives"""
    MAKESPAN = "MAKESPAN"
    COST = "COST"
    QUALITY = "QUALITY"
    UTILIZATION = "UTILIZATION"
    ON_TIME_DELIVERY = "ON_TIME_DELIVERY"


@dataclass
class ScheduleSolution:
    """Represents a production schedule solution"""
    solution_id: str
    job_assignments: Dict[str, str]  # job_id -> agent_id
    job_start_times: Dict[str, float]  # job_id -> start_time
    
    # Objective values
    makespan: float = 0.0
    total_cost: float = 0.0
    avg_quality: float = 0.0
    avg_utilization: float = 0.0
    on_time_rate: float = 0.0
    
    # Pareto ranking
    rank: int = 0
    crowding_distance: float = 0.0
    
    def __repr__(self):
        return (f"Solution(id={self.solution_id}, makespan={self.makespan:.1f}, "
                f"cost={self.total_cost:.1f}, quality={self.avg_quality:.2f}, "
                f"rank={self.rank})")


@dataclass
class Job:
    """Production job"""
    job_id: str
    duration: float
    deadline: float
    cost_coefficient: float = 1.0
    quality_requirement: float = 0.9


@dataclass
class Agent:
    """Production agent"""
    agent_id: str
    capability: float  # 0.0-1.0 quality capability
    cost_per_hour: float
    available_time: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for multi-objective optimization"""
    objectives: List[Objective] = field(default_factory=lambda: [
        Objective.MAKESPAN,
        Objective.COST,
        Objective.QUALITY,
        Objective.ON_TIME_DELIVERY
    ])
    
    # Objective weights (for weighted sum fallback)
    weights: Dict[Objective, float] = field(default_factory=lambda: {
        Objective.MAKESPAN: 0.3,
        Objective.COST: 0.25,
        Objective.QUALITY: 0.25,
        Objective.UTILIZATION: 0.1,
        Objective.ON_TIME_DELIVERY: 0.1
    })
    
    # NSGA-III parameters
    population_size: int = 100
    max_generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    
    # User preferences
    prefer_quality: bool = False
    prefer_cost: bool = False
    prefer_speed: bool = False


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer using NSGA-III algorithm
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.population: List[ScheduleSolution] = []
        self.pareto_front: List[ScheduleSolution] = []
        
    def optimize(self, jobs: List[Job], agents: List[Agent]) -> List[ScheduleSolution]:
        """
        Run multi-objective optimization
        
        Returns:
            Pareto-optimal solutions
        """
        logger.info(f"Starting optimization with {len(jobs)} jobs and {len(agents)} agents")
        
        # Initialize population
        self.population = self._initialize_population(jobs, agents)
        
        # Evaluate initial population
        for solution in self.population:
            self._evaluate_solution(solution, jobs, agents)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(self.population)
            
            # Select parents
            parents = self._selection(self.population)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i + 1], jobs, agents)
                child1 = self._mutate(child1, jobs, agents)
                child2 = self._mutate(child2, jobs, agents)
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            for solution in offspring:
                self._evaluate_solution(solution, jobs, agents)
            
            # Combine populations
            combined = self.population + offspring
            
            # Select next generation
            fronts = self._fast_non_dominated_sort(combined)
            self.population = []
            
            for front in fronts:
                if len(self.population) + len(front) <= self.config.population_size:
                    self.population.extend(front)
                else:
                    # Calculate crowding distance
                    self._calculate_crowding_distance(front)
                    # Sort by crowding distance
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    # Take remaining solutions
                    remaining = self.config.population_size - len(self.population)
                    self.population.extend(front[:remaining])
                    break
            
            if (generation + 1) % 10 == 0:
                logger.info(f"Generation {generation + 1}/{self.config.max_generations}: "
                          f"Pareto front size = {len(fronts[0])}")
        
        # Extract final Pareto front
        fronts = self._fast_non_dominated_sort(self.population)
        self.pareto_front = fronts[0]
        
        logger.info(f"Optimization complete. Pareto front has {len(self.pareto_front)} solutions")
        
        return self.pareto_front
    
    def select_solution_by_preference(self) -> Optional[ScheduleSolution]:
        """
        Select best solution from Pareto front based on user preferences
        """
        if not self.pareto_front:
            return None
        
        if self.config.prefer_quality:
            # Prioritize quality
            return max(self.pareto_front, key=lambda s: s.avg_quality)
        elif self.config.prefer_cost:
            # Minimize cost
            return min(self.pareto_front, key=lambda s: s.total_cost)
        elif self.config.prefer_speed:
            # Minimize makespan
            return min(self.pareto_front, key=lambda s: s.makespan)
        else:
            # Use weighted sum
            return self._select_by_weighted_sum()
    
    def _select_by_weighted_sum(self) -> ScheduleSolution:
        """Select solution using weighted sum of normalized objectives"""
        # Normalize objectives
        min_makespan = min(s.makespan for s in self.pareto_front)
        max_makespan = max(s.makespan for s in self.pareto_front)
        min_cost = min(s.total_cost for s in self.pareto_front)
        max_cost = max(s.total_cost for s in self.pareto_front)
        
        best_solution = None
        best_score = float('inf')
        
        for solution in self.pareto_front:
            # Normalize (lower is better for makespan/cost, higher for quality)
            norm_makespan = ((solution.makespan - min_makespan) / 
                           (max_makespan - min_makespan + 1e-6))
            norm_cost = ((solution.total_cost - min_cost) / 
                        (max_cost - min_cost + 1e-6))
            norm_quality = 1.0 - solution.avg_quality  # Invert (lower is better)
            norm_otd = 1.0 - solution.on_time_rate  # Invert
            
            # Weighted sum
            score = (self.config.weights[Objective.MAKESPAN] * norm_makespan +
                    self.config.weights[Objective.COST] * norm_cost +
                    self.config.weights[Objective.QUALITY] * norm_quality +
                    self.config.weights[Objective.ON_TIME_DELIVERY] * norm_otd)
            
            if score < best_score:
                best_score = score
                best_solution = solution
        
        return best_solution
    
    def _initialize_population(self, jobs: List[Job], agents: List[Agent]) -> List[ScheduleSolution]:
        """Initialize random population"""
        population = []
        
        for i in range(self.config.population_size):
            solution = ScheduleSolution(
                solution_id=f"SOL_{i}",
                job_assignments={},
                job_start_times={}
            )
            
            # Random assignment
            for job in jobs:
                agent = random.choice(agents)
                solution.job_assignments[job.job_id] = agent.agent_id
                solution.job_start_times[job.job_id] = 0.0
            
            # Calculate start times using greedy scheduling
            self._calculate_start_times(solution, jobs, agents)
            
            population.append(solution)
        
        return population
    
    def _calculate_start_times(self, solution: ScheduleSolution, jobs: List[Job], agents: List[Agent]):
        """Calculate job start times based on assignments"""
        agent_available = {agent.agent_id: 0.0 for agent in agents}
        
        # Sort jobs by ID for deterministic ordering
        sorted_jobs = sorted(jobs, key=lambda j: j.job_id)
        
        for job in sorted_jobs:
            agent_id = solution.job_assignments[job.job_id]
            start_time = agent_available[agent_id]
            solution.job_start_times[job.job_id] = start_time
            agent_available[agent_id] = start_time + job.duration
    
    def _evaluate_solution(self, solution: ScheduleSolution, jobs: List[Job], agents: List[Agent]):
        """Evaluate all objectives for a solution"""
        agent_map = {agent.agent_id: agent for agent in agents}
        job_map = {job.job_id: job for job in jobs}
        
        # 1. Makespan (completion time of last job)
        completion_times = [solution.job_start_times[job.job_id] + job.duration 
                          for job in jobs]
        solution.makespan = max(completion_times) if completion_times else 0.0
        
        # 2. Total cost
        total_cost = 0.0
        for job in jobs:
            agent = agent_map[solution.job_assignments[job.job_id]]
            job_cost = job.duration * agent.cost_per_hour * job.cost_coefficient
            total_cost += job_cost
        solution.total_cost = total_cost
        
        # 3. Average quality
        qualities = []
        for job in jobs:
            agent = agent_map[solution.job_assignments[job.job_id]]
            # Quality degrades if agent capability < requirement
            if agent.capability >= job.quality_requirement:
                quality = agent.capability
            else:
                quality = agent.capability * 0.8  # Penalty
            qualities.append(quality)
        solution.avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        
        # 4. Average utilization
        agent_work_time = {agent.agent_id: 0.0 for agent in agents}
        for job in jobs:
            agent_id = solution.job_assignments[job.job_id]
            agent_work_time[agent_id] += job.duration
        
        utilizations = [agent_work_time[agent.agent_id] / (solution.makespan + 1e-6) 
                       for agent in agents]
        solution.avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0
        
        # 5. On-time delivery rate
        on_time_count = 0
        for job in jobs:
            completion = solution.job_start_times[job.job_id] + job.duration
            if completion <= job.deadline:
                on_time_count += 1
        solution.on_time_rate = on_time_count / len(jobs) if jobs else 0.0
    
    def _fast_non_dominated_sort(self, population: List[ScheduleSolution]) -> List[List[ScheduleSolution]]:
        """Fast non-dominated sorting (NSGA-II)"""
        fronts = [[]]
        domination_count = {sol.solution_id: 0 for sol in population}
        dominated_solutions = {sol.solution_id: [] for sol in population}
        
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                
                if self._dominates(p, q):
                    dominated_solutions[p.solution_id].append(q)
                elif self._dominates(q, p):
                    domination_count[p.solution_id] += 1
            
            if domination_count[p.solution_id] == 0:
                p.rank = 0
                fronts[0].append(p)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for p in fronts[current_front]:
                for q in dominated_solutions[p.solution_id]:
                    domination_count[q.solution_id] -= 1
                    if domination_count[q.solution_id] == 0:
                        q.rank = current_front + 1
                        next_front.append(q)
            current_front += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, sol1: ScheduleSolution, sol2: ScheduleSolution) -> bool:
        """Check if sol1 dominates sol2"""
        # For minimization objectives: lower is better
        # For maximization objectives: higher is better
        
        better_in_any = False
        
        # Makespan (minimize)
        if sol1.makespan > sol2.makespan:
            return False
        elif sol1.makespan < sol2.makespan:
            better_in_any = True
        
        # Cost (minimize)
        if sol1.total_cost > sol2.total_cost:
            return False
        elif sol1.total_cost < sol2.total_cost:
            better_in_any = True
        
        # Quality (maximize)
        if sol1.avg_quality < sol2.avg_quality:
            return False
        elif sol1.avg_quality > sol2.avg_quality:
            better_in_any = True
        
        # On-time delivery (maximize)
        if sol1.on_time_rate < sol2.on_time_rate:
            return False
        elif sol1.on_time_rate > sol2.on_time_rate:
            better_in_any = True
        
        return better_in_any
    
    def _calculate_crowding_distance(self, front: List[ScheduleSolution]):
        """Calculate crowding distance for diversity preservation"""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        # Initialize
        for solution in front:
            solution.crowding_distance = 0.0
        
        # For each objective
        for obj_func in [lambda s: s.makespan, 
                        lambda s: s.total_cost,
                        lambda s: s.avg_quality,
                        lambda s: s.on_time_rate]:
            
            # Sort by objective
            front.sort(key=obj_func)
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = obj_func(front[-1]) - obj_func(front[0])
            if obj_range == 0:
                continue
            
            # Calculate distance for middle solutions
            for i in range(1, len(front) - 1):
                distance = (obj_func(front[i + 1]) - obj_func(front[i - 1])) / obj_range
                front[i].crowding_distance += distance
    
    def _selection(self, population: List[ScheduleSolution]) -> List[ScheduleSolution]:
        """Tournament selection"""
        parents = []
        
        for _ in range(len(population)):
            # Binary tournament
            candidate1 = random.choice(population)
            candidate2 = random.choice(population)
            
            if candidate1.rank < candidate2.rank:
                parents.append(candidate1)
            elif candidate1.rank > candidate2.rank:
                parents.append(candidate2)
            else:
                # Same rank - use crowding distance
                if candidate1.crowding_distance > candidate2.crowding_distance:
                    parents.append(candidate1)
                else:
                    parents.append(candidate2)
        
        return parents
    
    def _crossover(self, parent1: ScheduleSolution, parent2: ScheduleSolution,
                   jobs: List[Job], agents: List[Agent]) -> Tuple[ScheduleSolution, ScheduleSolution]:
        """Single-point crossover"""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        job_ids = list(parent1.job_assignments.keys())
        crossover_point = random.randint(1, len(job_ids) - 1)
        
        child1 = ScheduleSolution(
            solution_id=f"CHILD_{random.randint(0, 1000000)}",
            job_assignments={},
            job_start_times={}
        )
        
        child2 = ScheduleSolution(
            solution_id=f"CHILD_{random.randint(0, 1000000)}",
            job_assignments={},
            job_start_times={}
        )
        
        # First part from parent1, second from parent2
        for i, job_id in enumerate(job_ids):
            if i < crossover_point:
                child1.job_assignments[job_id] = parent1.job_assignments[job_id]
                child2.job_assignments[job_id] = parent2.job_assignments[job_id]
            else:
                child1.job_assignments[job_id] = parent2.job_assignments[job_id]
                child2.job_assignments[job_id] = parent1.job_assignments[job_id]
        
        self._calculate_start_times(child1, jobs, agents)
        self._calculate_start_times(child2, jobs, agents)
        
        return child1, child2
    
    def _mutate(self, solution: ScheduleSolution, jobs: List[Job], 
                agents: List[Agent]) -> ScheduleSolution:
        """Mutation by random reassignment"""
        if random.random() > self.config.mutation_rate:
            return solution
        
        # Randomly reassign one job
        job_id = random.choice(list(solution.job_assignments.keys()))
        new_agent = random.choice(agents)
        solution.job_assignments[job_id] = new_agent.agent_id
        
        # Recalculate start times
        self._calculate_start_times(solution, jobs, agents)
        
        return solution


# Example usage
def main():
    # Create jobs
    jobs = [
        Job(job_id=f"JOB_{i}", duration=random.uniform(10, 60), 
            deadline=random.uniform(100, 300), quality_requirement=random.uniform(0.7, 0.95))
        for i in range(20)
    ]
    
    # Create agents
    agents = [
        Agent(agent_id=f"AGENT_{i}", capability=random.uniform(0.6, 1.0),
              cost_per_hour=random.uniform(50, 150))
        for i in range(5)
    ]
    
    # Configure optimizer
    config = OptimizationConfig(
        population_size=50,
        max_generations=30,
        prefer_quality=True  # User preference
    )
    
    # Run optimization
    optimizer = MultiObjectiveOptimizer(config)
    pareto_front = optimizer.optimize(jobs, agents)
    
    # Select best solution based on preference
    best_solution = optimizer.select_solution_by_preference()
    
    print(f"\n{'='*60}")
    print(f"Pareto Front ({len(pareto_front)} solutions):")
    print(f"{'='*60}")
    for i, sol in enumerate(pareto_front[:5], 1):  # Show top 5
        print(f"{i}. {sol}")
    
    print(f"\n{'='*60}")
    print(f"Selected Solution (based on quality preference):")
    print(f"{'='*60}")
    print(best_solution)


if __name__ == "__main__":
    main()
