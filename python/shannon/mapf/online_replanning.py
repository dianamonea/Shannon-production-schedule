"""
Online Dynamic Replanning for L-HCBS

Handles runtime disruptions:
1. Machine breakdowns (CNC failure)
2. Urgent orders (new tasks arrive)
3. Quality issues (rework needed)
4. AGV delays (slower than expected)

Key innovation: Partial replanning preserves valid paths
while efficiently resolving new conflicts.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import copy
import math

from .heterogeneous_agent import HeterogeneousAgent, AgentType, AgentDict
from .heterogeneous_cbs import GridMap, Conflict
from .lhcbs import LearningGuidedHCBS, SearchStatistics


class DisruptionType(Enum):
    """Types of runtime disruptions."""
    MACHINE_BREAKDOWN = "machine_breakdown"
    URGENT_ORDER = "urgent_order"
    QUALITY_REWORK = "quality_rework"
    AGV_DELAY = "agv_delay"
    PATH_BLOCKED = "path_blocked"
    NEW_OBSTACLE = "new_obstacle"


@dataclass
class Disruption:
    """A runtime disruption event."""
    disruption_type: DisruptionType
    timestamp: float
    affected_agents: List[str]
    affected_positions: List[Tuple[int, int]]
    duration: Optional[float] = None  # For temporary disruptions
    new_goal: Optional[Tuple[int, int]] = None  # For urgent orders
    priority_boost: float = 0.0  # For urgent tasks
    
    def is_temporary(self) -> bool:
        return self.duration is not None and self.duration > 0


@dataclass
class ReplanningContext:
    """Context for replanning decision."""
    current_time: int
    current_paths: Dict[str, List[Tuple[int, int]]]
    current_positions: Dict[str, Tuple[int, int]]
    goals: Dict[str, Tuple[int, int]]
    pending_disruptions: List[Disruption]


class PathValidator:
    """Validates paths against current world state."""
    
    def __init__(self, grid_map: GridMap):
        self.grid = grid_map
        self.blocked_positions: Set[Tuple[int, int]] = set()
        self.blocked_until: Dict[Tuple[int, int], float] = {}
    
    def add_obstacle(self, pos: Tuple[int, int], 
                    duration: Optional[float] = None):
        """Add new obstacle."""
        self.blocked_positions.add(pos)
        if duration:
            self.blocked_until[pos] = time.time() + duration
    
    def remove_obstacle(self, pos: Tuple[int, int]):
        """Remove obstacle."""
        self.blocked_positions.discard(pos)
        self.blocked_until.pop(pos, None)
    
    def update_temporary_obstacles(self):
        """Remove expired temporary obstacles."""
        current_time = time.time()
        expired = [pos for pos, until in self.blocked_until.items()
                  if until <= current_time]
        for pos in expired:
            self.remove_obstacle(pos)
    
    def is_path_valid(self, path: List[Tuple[int, int]], 
                     start_time: int = 0) -> Tuple[bool, Optional[int]]:
        """
        Check if path is still valid.
        
        Returns:
            (valid, first_invalid_time)
        """
        self.update_temporary_obstacles()
        
        for t, pos in enumerate(path):
            if pos in self.blocked_positions:
                return False, start_time + t
        
        return True, None
    
    def validate_paths(self, paths: Dict[str, List[Tuple[int, int]]],
                      current_time: int) -> Dict[str, Tuple[bool, Optional[int]]]:
        """Validate all paths."""
        results = {}
        for agent_id, path in paths.items():
            # Skip already completed path segments
            remaining_path = path[current_time:] if current_time < len(path) else []
            results[agent_id] = self.is_path_valid(remaining_path, current_time)
        return results


class PartialReplanner:
    """
    Performs partial replanning when disruptions occur.
    
    Key optimization: Only replan affected agents, keep valid paths.
    """
    
    def __init__(self, base_solver: LearningGuidedHCBS):
        self.solver = base_solver
        self.path_validator = PathValidator(base_solver.grid)
    
    def identify_affected_agents(self, disruption: Disruption,
                                current_paths: Dict[str, List[Tuple[int, int]]],
                                current_time: int) -> Set[str]:
        """Identify which agents are affected by disruption."""
        affected = set(disruption.affected_agents)
        
        # Check if any agent's path goes through affected positions
        for agent_id, path in current_paths.items():
            remaining_path = path[current_time:]
            
            for pos in remaining_path:
                if pos in disruption.affected_positions:
                    affected.add(agent_id)
                    break
        
        return affected
    
    def extract_remaining_problem(self, 
                                 current_paths: Dict[str, List[Tuple[int, int]]],
                                 current_time: int,
                                 goals: Dict[str, Tuple[int, int]],
                                 affected_agents: Set[str]
                                 ) -> Tuple[Dict[str, Tuple[int, int]], 
                                           Dict[str, Tuple[int, int]],
                                           Dict[str, List[Tuple[int, int]]]]:
        """
        Extract remaining planning problem.
        
        Returns:
            (new_starts, new_goals, preserved_paths)
        """
        new_starts = {}
        new_goals = {}
        preserved_paths = {}
        
        for agent_id, path in current_paths.items():
            current_pos = path[min(current_time, len(path)-1)]
            goal = goals[agent_id]
            
            if agent_id in affected_agents:
                # Need to replan
                new_starts[agent_id] = current_pos
                new_goals[agent_id] = goal
            else:
                # Preserve remaining path
                remaining = path[current_time:]
                if remaining:
                    preserved_paths[agent_id] = remaining
        
        return new_starts, new_goals, preserved_paths
    
    def merge_paths(self, 
                   preserved_paths: Dict[str, List[Tuple[int, int]]],
                   new_paths: Dict[str, List[Tuple[int, int]]],
                   current_time: int) -> Dict[str, List[Tuple[int, int]]]:
        """Merge preserved and newly planned paths."""
        merged = {}
        
        # Add preserved paths
        for agent_id, path in preserved_paths.items():
            merged[agent_id] = path
        
        # Add new paths
        for agent_id, path in new_paths.items():
            merged[agent_id] = path
        
        return merged
    
    def replan(self, context: ReplanningContext,
              disruption: Disruption) -> Optional[Dict[str, List[Tuple[int, int]]]]:
        """
        Perform partial replanning for disruption.
        
        Returns:
            New paths from current time, or None if failed
        """
        # Update obstacles
        for pos in disruption.affected_positions:
            self.path_validator.add_obstacle(
                pos, 
                duration=disruption.duration
            )
        
        # Identify affected agents
        affected = self.identify_affected_agents(
            disruption, context.current_paths, context.current_time
        )
        
        if not affected:
            # No agents affected, keep current paths
            return context.current_paths
        
        # Extract remaining problem
        new_starts, new_goals, preserved_paths = self.extract_remaining_problem(
            context.current_paths, context.current_time,
            context.goals, affected
        )
        
        # Handle urgent order (new goal)
        if disruption.disruption_type == DisruptionType.URGENT_ORDER:
            if disruption.new_goal and disruption.affected_agents:
                for agent_id in disruption.affected_agents:
                    if agent_id in new_goals:
                        new_goals[agent_id] = disruption.new_goal
        
        # Solve reduced problem
        if new_starts:
            new_paths = self.solver.solve(new_starts, new_goals, 
                                         max_iterations=5000,
                                         timeout=10.0)
            
            if new_paths is None:
                return None
            
            return self.merge_paths(preserved_paths, new_paths, context.current_time)
        
        return {aid: path[context.current_time:] 
               for aid, path in context.current_paths.items()}


class OnlineExecutionManager:
    """
    Manages online execution with dynamic replanning.
    
    Simulates real-time execution with disruptions.
    """
    
    def __init__(self, solver: LearningGuidedHCBS, agents: AgentDict):
        self.solver = solver
        self.agents = agents
        self.replanner = PartialReplanner(solver)
        
        # Execution state
        self.current_time: int = 0
        self.current_paths: Dict[str, List[Tuple[int, int]]] = {}
        self.goals: Dict[str, Tuple[int, int]] = {}
        
        # Disruption queue
        self.disruption_queue: List[Tuple[float, Disruption]] = []
        
        # Statistics
        self.replanning_count: int = 0
        self.total_replanning_time: float = 0.0
        self.disruptions_handled: int = 0
    
    def initialize(self, starts: Dict[str, Tuple[int, int]],
                  goals: Dict[str, Tuple[int, int]]) -> bool:
        """Initialize execution with initial plan."""
        self.goals = goals
        self.current_time = 0
        
        paths = self.solver.solve(starts, goals)
        if paths is None:
            return False
        
        self.current_paths = paths
        return True
    
    def add_disruption(self, disruption: Disruption, 
                      trigger_time: Optional[float] = None):
        """Schedule a disruption."""
        trigger = trigger_time or disruption.timestamp
        self.disruption_queue.append((trigger, disruption))
        self.disruption_queue.sort(key=lambda x: x[0])
    
    def get_current_positions(self) -> Dict[str, Tuple[int, int]]:
        """Get current positions of all agents."""
        positions = {}
        for agent_id, path in self.current_paths.items():
            idx = min(self.current_time, len(path) - 1)
            positions[agent_id] = path[idx]
        return positions
    
    def step(self) -> bool:
        """
        Execute one time step.
        
        Returns:
            True if execution should continue
        """
        # Check for disruptions
        while self.disruption_queue and self.disruption_queue[0][0] <= self.current_time:
            _, disruption = self.disruption_queue.pop(0)
            
            if not self._handle_disruption(disruption):
                return False  # Failed to handle disruption
        
        # Check if all agents reached goal
        positions = self.get_current_positions()
        all_done = all(
            positions[aid] == self.goals[aid]
            for aid in self.agents
        )
        
        if all_done:
            return False  # Execution complete
        
        self.current_time += 1
        return True
    
    def _handle_disruption(self, disruption: Disruption) -> bool:
        """Handle a disruption event."""
        self.disruptions_handled += 1
        
        replan_start = time.time()
        
        context = ReplanningContext(
            current_time=self.current_time,
            current_paths=self.current_paths,
            current_positions=self.get_current_positions(),
            goals=self.goals,
            pending_disruptions=[disruption]
        )
        
        new_paths = self.replanner.replan(context, disruption)
        
        self.total_replanning_time += time.time() - replan_start
        self.replanning_count += 1
        
        if new_paths is None:
            return False
        
        self.current_paths = new_paths
        return True
    
    def execute_full(self, max_steps: int = 1000) -> bool:
        """
        Execute until completion or max steps.
        
        Returns:
            True if all agents reached goals
        """
        for _ in range(max_steps):
            if not self.step():
                # Check if successful
                positions = self.get_current_positions()
                return all(
                    positions[aid] == self.goals[aid]
                    for aid in self.agents
                )
        
        return False  # Timeout
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_time_steps': self.current_time,
            'replanning_count': self.replanning_count,
            'avg_replanning_time': (self.total_replanning_time / self.replanning_count
                                   if self.replanning_count > 0 else 0),
            'disruptions_handled': self.disruptions_handled
        }


class DisruptionGenerator:
    """Generates random disruptions for testing."""
    
    def __init__(self, grid_map: GridMap, agents: AgentDict):
        self.grid = grid_map
        self.agents = agents
        import random
        self.random = random
    
    def generate_machine_breakdown(self, 
                                   cnc_agents: List[str],
                                   time_range: Tuple[float, float],
                                   duration_range: Tuple[float, float]
                                   ) -> Disruption:
        """Generate random machine breakdown."""
        agent_id = self.random.choice(cnc_agents)
        agent = self.agents[agent_id]
        
        return Disruption(
            disruption_type=DisruptionType.MACHINE_BREAKDOWN,
            timestamp=self.random.uniform(*time_range),
            affected_agents=[agent_id],
            affected_positions=[(int(agent.position[0]), int(agent.position[1]))],
            duration=self.random.uniform(*duration_range)
        )
    
    def generate_urgent_order(self,
                             agv_agents: List[str],
                             time_range: Tuple[float, float],
                             goal_positions: List[Tuple[int, int]]
                             ) -> Disruption:
        """Generate urgent order disruption."""
        agent_id = self.random.choice(agv_agents)
        new_goal = self.random.choice(goal_positions)
        
        return Disruption(
            disruption_type=DisruptionType.URGENT_ORDER,
            timestamp=self.random.uniform(*time_range),
            affected_agents=[agent_id],
            affected_positions=[],
            new_goal=new_goal,
            priority_boost=1.5
        )
    
    def generate_path_blocked(self,
                             time_range: Tuple[float, float],
                             duration_range: Tuple[float, float]
                             ) -> Disruption:
        """Generate path blocked disruption."""
        # Random position on grid
        x = self.random.randint(0, self.grid.width - 1)
        y = self.random.randint(0, self.grid.height - 1)
        
        return Disruption(
            disruption_type=DisruptionType.PATH_BLOCKED,
            timestamp=self.random.uniform(*time_range),
            affected_agents=[],
            affected_positions=[(x, y)],
            duration=self.random.uniform(*duration_range)
        )
    
    def generate_scenario(self, 
                         num_breakdowns: int = 2,
                         num_urgent: int = 1,
                         num_blocks: int = 3,
                         time_range: Tuple[float, float] = (5.0, 50.0)
                         ) -> List[Disruption]:
        """Generate a complete disruption scenario."""
        disruptions = []
        
        # Find CNC and AGV agents
        cnc_agents = [aid for aid, a in self.agents.items() 
                     if a.agent_type == AgentType.CNC]
        agv_agents = [aid for aid, a in self.agents.items()
                     if a.agent_type == AgentType.AGV]
        
        # Machine breakdowns
        for _ in range(min(num_breakdowns, len(cnc_agents))):
            if cnc_agents:
                disruptions.append(
                    self.generate_machine_breakdown(
                        cnc_agents, time_range, (5.0, 15.0)
                    )
                )
        
        # Urgent orders
        if agv_agents:
            goal_positions = [(self.random.randint(0, self.grid.width-1),
                             self.random.randint(0, self.grid.height-1))
                            for _ in range(5)]
            
            for _ in range(num_urgent):
                disruptions.append(
                    self.generate_urgent_order(agv_agents, time_range, goal_positions)
                )
        
        # Path blocks
        for _ in range(num_blocks):
            disruptions.append(
                self.generate_path_blocked(time_range, (3.0, 10.0))
            )
        
        return disruptions
