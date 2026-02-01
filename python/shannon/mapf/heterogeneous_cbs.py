"""
Heterogeneous Conflict-Based Search (H-CBS)

Core algorithm for heterogeneous multi-agent path finding.
Extends standard CBS to handle agents with different kinematic constraints.

Key contributions:
1. Type-aware conflict detection (different collision radii)
2. Heterogeneous low-level planner (respects kinematics)
3. Constraint propagation across agent types

Reference: Based on CBS (Sharon et al., 2015) extended for heterogeneous agents.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, FrozenSet
from enum import Enum
import math
import copy

from .heterogeneous_agent import (
    HeterogeneousAgent, AgentType, AgentState, AgentDict
)


class ConflictType(Enum):
    """Types of conflicts in H-MAPF."""
    VERTEX = "vertex"      # Two agents at same position
    EDGE = "edge"          # Two agents swap positions
    FOLLOWING = "following"  # Agent too close behind another
    WORKSPACE = "workspace"  # Robot arm workspace overlap


@dataclass
class Conflict:
    """A conflict between two agents."""
    agent1: str
    agent2: str
    conflict_type: ConflictType
    time: int
    position1: Tuple[int, int]
    position2: Tuple[int, int]
    
    def __hash__(self):
        return hash((self.agent1, self.agent2, self.time, 
                    self.position1, self.position2))


@dataclass
class Constraint:
    """A constraint on an agent's path."""
    agent_id: str
    time: int
    position: Tuple[int, int]
    is_vertex: bool = True  # False for edge constraint
    next_position: Optional[Tuple[int, int]] = None  # For edge constraints


@dataclass
class CTNode:
    """Constraint Tree Node for CBS."""
    constraints: Dict[str, List[Constraint]]
    paths: Dict[str, List[Tuple[int, int]]]
    cost: float
    conflicts: List[Conflict] = field(default_factory=list)
    
    def __lt__(self, other):
        return self.cost < other.cost


class GridMap:
    """Grid-based map for path planning."""
    
    def __init__(self, width: int, height: int, 
                 obstacles: Optional[Set[Tuple[int, int]]] = None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
        
        # Workspace zones for different agent types
        self.cnc_zones: Set[Tuple[int, int]] = set()
        self.agv_paths: Set[Tuple[int, int]] = set()
        self.robot_workspaces: Dict[str, Set[Tuple[int, int]]] = {}
    
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.obstacles
    
    def passable_for_agent(self, x: int, y: int, 
                           agent: HeterogeneousAgent) -> bool:
        """Check if position is passable for specific agent type."""
        if not self.passable(x, y):
            return False
        
        # CNC machines are static
        if agent.agent_type == AgentType.CNC:
            return (x, y) == (int(agent.position[0]), int(agent.position[1]))
        
        return True
    
    def neighbors(self, x: int, y: int, 
                  agent: Optional[HeterogeneousAgent] = None,
                  include_wait: bool = True) -> List[Tuple[int, int]]:
        """Get neighboring positions including wait action."""
        results = []
        
        # Wait action (stay in place)
        if include_wait:
            results.append((x, y))
        
        # Movement actions
        if agent is None or agent.can_move():
            candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            
            # Add diagonal movement for holonomic agents
            if agent and agent.kinematics.is_holonomic:
                candidates.extend([(x+1, y+1), (x+1, y-1), 
                                  (x-1, y+1), (x-1, y-1)])
            
            for nx, ny in candidates:
                if self.in_bounds(nx, ny) and self.passable(nx, ny):
                    if agent is None or self.passable_for_agent(nx, ny, agent):
                        results.append((nx, ny))
        
        return results


class HeterogeneousLowLevelPlanner:
    """
    Low-level planner for individual agents in H-CBS.
    Respects agent-specific kinematic constraints.
    """
    
    def __init__(self, grid_map: GridMap, agent: HeterogeneousAgent):
        self.grid = grid_map
        self.agent = agent
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Admissible heuristic considering agent kinematics."""
        distance = math.hypot(pos[0] - goal[0], pos[1] - goal[1])
        
        if self.agent.kinematics.max_velocity > 0:
            return distance / self.agent.kinematics.max_velocity
        return distance
    
    def movement_cost(self, from_pos: Tuple[int, int], 
                      to_pos: Tuple[int, int]) -> float:
        """Cost of moving between positions."""
        if from_pos == to_pos:
            return 1.0  # Wait cost
        
        distance = math.hypot(to_pos[0] - from_pos[0], 
                             to_pos[1] - from_pos[1])
        
        if self.agent.kinematics.max_velocity > 0:
            return distance / self.agent.kinematics.max_velocity
        return 1.0
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int],
             constraints: List[Constraint],
             max_time: int = 200) -> Optional[List[Tuple[int, int]]]:
        """
        A* search with constraints for single agent.
        Returns time-indexed path.
        """
        # Build constraint lookup
        vertex_constraints: Dict[Tuple[int, Tuple[int, int]], bool] = {}
        edge_constraints: Set[Tuple[int, Tuple[int, int], Tuple[int, int]]] = set()
        
        for c in constraints:
            if c.is_vertex:
                vertex_constraints[(c.time, c.position)] = True
            else:
                edge_constraints.add((c.time, c.position, c.next_position))
        
        # A* search in space-time
        # State: (x, y, t)
        start_state = (start[0], start[1], 0)
        
        # Priority queue: (f, g, state)
        frontier = [(self.heuristic(start, goal), 0.0, start_state)]
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
            start_state: None
        }
        g_score: Dict[Tuple[int, int, int], float] = {start_state: 0.0}
        
        while frontier:
            _, g, current = heapq.heappop(frontier)
            x, y, t = current
            
            # Goal reached
            if (x, y) == goal:
                return self._reconstruct_path(came_from, current)
            
            if t >= max_time:
                continue
            
            # Expand neighbors
            for nx, ny in self.grid.neighbors(x, y, self.agent):
                nt = t + 1
                next_state = (nx, ny, nt)
                
                # Check vertex constraint
                if (nt, (nx, ny)) in vertex_constraints:
                    continue
                
                # Check edge constraint (swap)
                if (nt, (x, y), (nx, ny)) in edge_constraints:
                    continue
                if (nt, (nx, ny), (x, y)) in edge_constraints:
                    continue
                
                # Calculate cost
                move_cost = self.movement_cost((x, y), (nx, ny))
                new_g = g + move_cost
                
                if next_state not in g_score or new_g < g_score[next_state]:
                    g_score[next_state] = new_g
                    f = new_g + self.heuristic((nx, ny), goal)
                    heapq.heappush(frontier, (f, new_g, next_state))
                    came_from[next_state] = current
        
        return None  # No path found
    
    def _reconstruct_path(self, came_from: Dict, end_state: Tuple[int, int, int]
                         ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = []
        current = end_state
        
        while current is not None:
            x, y, _ = current
            path.append((x, y))
            current = came_from.get(current)
        
        path.reverse()
        return path


class HeterogeneousCBS:
    """
    Heterogeneous Conflict-Based Search.
    
    Key differences from standard CBS:
    1. Type-aware conflict detection
    2. Different collision radii for different agent types
    3. Kinematic-aware low-level planning
    """
    
    def __init__(self, grid_map: GridMap, agents: AgentDict):
        self.grid = grid_map
        self.agents = agents
        
        # Collision radii for different agent type pairs
        self.collision_radii = {
            (AgentType.AGV, AgentType.AGV): 1.0,
            (AgentType.AGV, AgentType.ROBOT): 1.5,
            (AgentType.ROBOT, AgentType.ROBOT): 2.0,
            (AgentType.CNC, AgentType.AGV): 1.5,
            (AgentType.CNC, AgentType.ROBOT): 2.0,
        }
    
    def get_collision_radius(self, type1: AgentType, type2: AgentType) -> float:
        """Get collision radius for agent type pair."""
        key = (type1, type2) if type1.value <= type2.value else (type2, type1)
        return self.collision_radii.get(key, 1.0)
    
    def detect_conflicts(self, paths: Dict[str, List[Tuple[int, int]]]
                        ) -> List[Conflict]:
        """
        Detect all conflicts between paths.
        Considers heterogeneous collision radii.
        """
        conflicts = []
        agents_list = list(paths.keys())
        
        # Find maximum path length
        max_len = max(len(p) for p in paths.values()) if paths else 0
        
        for i, agent1_id in enumerate(agents_list):
            for agent2_id in agents_list[i+1:]:
                path1 = paths[agent1_id]
                path2 = paths[agent2_id]
                
                agent1 = self.agents[agent1_id]
                agent2 = self.agents[agent2_id]
                collision_radius = self.get_collision_radius(
                    agent1.agent_type, agent2.agent_type
                )
                
                # Check each timestep
                for t in range(max_len):
                    pos1 = path1[min(t, len(path1)-1)]
                    pos2 = path2[min(t, len(path2)-1)]
                    
                    # Vertex conflict
                    dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                    if dist < collision_radius:
                        conflicts.append(Conflict(
                            agent1=agent1_id,
                            agent2=agent2_id,
                            conflict_type=ConflictType.VERTEX,
                            time=t,
                            position1=pos1,
                            position2=pos2
                        ))
                        break  # One conflict per pair for efficiency
                    
                    # Edge conflict (swap)
                    if t > 0:
                        prev_pos1 = path1[min(t-1, len(path1)-1)]
                        prev_pos2 = path2[min(t-1, len(path2)-1)]
                        
                        if pos1 == prev_pos2 and pos2 == prev_pos1:
                            conflicts.append(Conflict(
                                agent1=agent1_id,
                                agent2=agent2_id,
                                conflict_type=ConflictType.EDGE,
                                time=t,
                                position1=pos1,
                                position2=pos2
                            ))
                            break
        
        return conflicts
    
    def create_constraints_from_conflict(self, conflict: Conflict
                                        ) -> Tuple[List[Constraint], List[Constraint]]:
        """
        Create constraints to resolve a conflict.
        Returns constraints for both agents.
        """
        constraints1 = []
        constraints2 = []
        
        if conflict.conflict_type == ConflictType.VERTEX:
            # Prevent agent1 from being at position at time t
            constraints1.append(Constraint(
                agent_id=conflict.agent1,
                time=conflict.time,
                position=conflict.position1,
                is_vertex=True
            ))
            # Prevent agent2 from being at position at time t
            constraints2.append(Constraint(
                agent_id=conflict.agent2,
                time=conflict.time,
                position=conflict.position2,
                is_vertex=True
            ))
        
        elif conflict.conflict_type == ConflictType.EDGE:
            # Edge constraints for swapping
            constraints1.append(Constraint(
                agent_id=conflict.agent1,
                time=conflict.time,
                position=conflict.position2,
                is_vertex=False,
                next_position=conflict.position1
            ))
            constraints2.append(Constraint(
                agent_id=conflict.agent2,
                time=conflict.time,
                position=conflict.position1,
                is_vertex=False,
                next_position=conflict.position2
            ))
        
        return constraints1, constraints2
    
    def compute_paths_cost(self, paths: Dict[str, List[Tuple[int, int]]]) -> float:
        """Compute sum of costs (SOC) for all paths."""
        return sum(len(path) - 1 for path in paths.values())
    
    def solve(self, starts: Dict[str, Tuple[int, int]],
              goals: Dict[str, Tuple[int, int]],
              max_iterations: int = 10000) -> Optional[Dict[str, List[Tuple[int, int]]]]:
        """
        Solve H-MAPF using Heterogeneous CBS.
        
        Args:
            starts: Agent ID -> start position
            goals: Agent ID -> goal position
            max_iterations: Maximum CBS iterations
        
        Returns:
            Dictionary of agent ID -> path, or None if no solution
        """
        # Compute initial paths (ignoring other agents)
        initial_paths = {}
        for agent_id in starts:
            agent = self.agents[agent_id]
            planner = HeterogeneousLowLevelPlanner(self.grid, agent)
            path = planner.plan(starts[agent_id], goals[agent_id], [])
            
            if path is None:
                return None  # No individual path exists
            initial_paths[agent_id] = path
        
        # Create root node
        root = CTNode(
            constraints={aid: [] for aid in starts},
            paths=initial_paths,
            cost=self.compute_paths_cost(initial_paths)
        )
        root.conflicts = self.detect_conflicts(root.paths)
        
        # Priority queue (min-heap by cost)
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            
            # Get lowest cost node
            current = heapq.heappop(open_list)
            
            # Check if solution found (no conflicts)
            if not current.conflicts:
                return current.paths
            
            # Pick first conflict to resolve
            conflict = current.conflicts[0]
            
            # Create two child nodes (one constraint each)
            constraints1, constraints2 = self.create_constraints_from_conflict(conflict)
            
            for new_constraints, constrained_agent in [
                (constraints1, conflict.agent1),
                (constraints2, conflict.agent2)
            ]:
                # Copy parent constraints
                child_constraints = {
                    aid: list(cs) for aid, cs in current.constraints.items()
                }
                child_constraints[constrained_agent].extend(new_constraints)
                
                # Copy paths
                child_paths = dict(current.paths)
                
                # Replan for constrained agent
                agent = self.agents[constrained_agent]
                planner = HeterogeneousLowLevelPlanner(self.grid, agent)
                new_path = planner.plan(
                    starts[constrained_agent],
                    goals[constrained_agent],
                    child_constraints[constrained_agent]
                )
                
                if new_path is not None:
                    child_paths[constrained_agent] = new_path
                    
                    child = CTNode(
                        constraints=child_constraints,
                        paths=child_paths,
                        cost=self.compute_paths_cost(child_paths)
                    )
                    child.conflicts = self.detect_conflicts(child_paths)
                    
                    heapq.heappush(open_list, child)
        
        return None  # No solution found within iterations


def solve_heterogeneous_mapf(
    grid_map: GridMap,
    agents: AgentDict,
    starts: Dict[str, Tuple[int, int]],
    goals: Dict[str, Tuple[int, int]]
) -> Optional[Dict[str, List[Tuple[int, int]]]]:
    """
    Convenience function to solve H-MAPF.
    
    Args:
        grid_map: The environment grid
        agents: Dictionary of heterogeneous agents
        starts: Start positions
        goals: Goal positions
    
    Returns:
        Paths for all agents, or None if unsolvable
    """
    solver = HeterogeneousCBS(grid_map, agents)
    return solver.solve(starts, goals)
