"""
Heterogeneous Agent Definitions for H-MAPF

Defines different agent types with their kinematic constraints:
- CNC Machine: Static, only processes tasks
- AGV: Omnidirectional movement, variable speed
- Robot Arm: Limited workspace, pick-and-place operations

This is a core contribution for the paper: first formalization of
heterogeneous kinematic constraints in MAPF for manufacturing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Callable
import math


class AgentType(Enum):
    """Types of agents in manufacturing environment."""
    CNC = "cnc"           # Static machine tool
    AGV = "agv"           # Mobile transport vehicle
    ROBOT = "robot"       # Robotic arm for loading/unloading


@dataclass
class KinematicConstraints:
    """Kinematic constraints for different agent types."""
    max_velocity: float           # m/s
    max_acceleration: float       # m/s^2
    max_angular_velocity: float   # rad/s (for AGV)
    turning_radius: float         # m (0 for omnidirectional)
    workspace_radius: float       # m (for robot arm)
    is_holonomic: bool           # Can move in any direction instantly
    
    @classmethod
    def cnc_constraints(cls) -> 'KinematicConstraints':
        """CNC machines are static - no movement."""
        return cls(
            max_velocity=0.0,
            max_acceleration=0.0,
            max_angular_velocity=0.0,
            turning_radius=0.0,
            workspace_radius=0.0,
            is_holonomic=True
        )
    
    @classmethod
    def agv_constraints(cls, speed: float = 1.0) -> 'KinematicConstraints':
        """AGV with omnidirectional or differential drive."""
        return cls(
            max_velocity=speed,
            max_acceleration=0.5,
            max_angular_velocity=math.pi,
            turning_radius=0.0,  # Omnidirectional
            workspace_radius=0.0,
            is_holonomic=True
        )
    
    @classmethod
    def robot_constraints(cls, reach: float = 1.5) -> 'KinematicConstraints':
        """Robot arm with limited workspace."""
        return cls(
            max_velocity=0.5,  # End-effector speed
            max_acceleration=1.0,
            max_angular_velocity=2 * math.pi,
            turning_radius=0.0,
            workspace_radius=reach,
            is_holonomic=True  # In task space
        )


@dataclass
class HeterogeneousAgent:
    """
    A heterogeneous agent with type-specific constraints.
    
    Key insight: Different agent types have fundamentally different
    motion models, which affects path planning complexity.
    """
    agent_id: str
    agent_type: AgentType
    position: Tuple[float, float]
    orientation: float  # radians
    kinematics: KinematicConstraints
    
    # Task-related
    current_task: Optional[str] = None
    task_queue: List[str] = field(default_factory=list)
    
    # State
    is_busy: bool = False
    payload: Optional[str] = None  # For AGV carrying items
    
    def can_move(self) -> bool:
        """Check if agent can physically move."""
        return self.agent_type != AgentType.CNC
    
    def movement_cost(self, from_pos: Tuple[float, float], 
                      to_pos: Tuple[float, float]) -> float:
        """
        Calculate movement cost considering kinematic constraints.
        This is crucial for heterogeneous CBS.
        """
        if not self.can_move():
            return float('inf') if from_pos != to_pos else 0.0
        
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        distance = math.hypot(dx, dy)
        
        # Time = distance / velocity
        if self.kinematics.max_velocity > 0:
            base_time = distance / self.kinematics.max_velocity
        else:
            return float('inf')
        
        # Add turning cost for non-holonomic agents
        if not self.kinematics.is_holonomic and self.kinematics.turning_radius > 0:
            angle_diff = abs(math.atan2(dy, dx) - self.orientation)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            turn_time = angle_diff / max(self.kinematics.max_angular_velocity, 1e-6)
            base_time += turn_time
        
        return base_time
    
    def reachable_positions(self, grid_map, time_budget: float) -> List[Tuple[int, int]]:
        """
        Get all positions reachable within time budget.
        Used for computing safe intervals in CBS.
        """
        if not self.can_move():
            return [(int(self.position[0]), int(self.position[1]))]
        
        max_distance = self.kinematics.max_velocity * time_budget
        reachable = []
        
        cx, cy = int(self.position[0]), int(self.position[1])
        search_radius = int(max_distance) + 1
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = cx + dx, cy + dy
                if grid_map.in_bounds(nx, ny) and grid_map.passable(nx, ny):
                    dist = math.hypot(dx, dy)
                    if dist <= max_distance:
                        reachable.append((nx, ny))
        
        return reachable


@dataclass  
class AgentState:
    """State of an agent at a specific time."""
    agent_id: str
    position: Tuple[float, float]
    orientation: float
    velocity: Tuple[float, float]
    time: float
    
    def distance_to(self, other: 'AgentState') -> float:
        return math.hypot(
            self.position[0] - other.position[0],
            self.position[1] - other.position[1]
        )


class HeterogeneousAgentFactory:
    """Factory for creating agents of different types."""
    
    @staticmethod
    def create_agent(agent_id: str, agent_type: AgentType, 
                    position: Tuple[float, float], **kwargs) -> HeterogeneousAgent:
        """Generic factory method for any agent type."""
        if agent_type == AgentType.CNC:
            return HeterogeneousAgentFactory.create_cnc(agent_id, position)
        elif agent_type == AgentType.AGV:
            speed = kwargs.get('speed', 1.0)
            return HeterogeneousAgentFactory.create_agv(agent_id, position, speed)
        elif agent_type == AgentType.ROBOT:
            reach = kwargs.get('reach', 1.5)
            return HeterogeneousAgentFactory.create_robot(agent_id, position, reach)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def create_cnc(agent_id: str, position: Tuple[float, float]) -> HeterogeneousAgent:
        return HeterogeneousAgent(
            agent_id=agent_id,
            agent_type=AgentType.CNC,
            position=position,
            orientation=0.0,
            kinematics=KinematicConstraints.cnc_constraints()
        )
    
    @staticmethod
    def create_agv(agent_id: str, position: Tuple[float, float], 
                   speed: float = 1.0) -> HeterogeneousAgent:
        return HeterogeneousAgent(
            agent_id=agent_id,
            agent_type=AgentType.AGV,
            position=position,
            orientation=0.0,
            kinematics=KinematicConstraints.agv_constraints(speed)
        )
    
    @staticmethod
    def create_robot(agent_id: str, position: Tuple[float, float],
                     reach: float = 1.5) -> HeterogeneousAgent:
        return HeterogeneousAgent(
            agent_id=agent_id,
            agent_type=AgentType.ROBOT,
            position=position,
            orientation=0.0,
            kinematics=KinematicConstraints.robot_constraints(reach)
        )


# Type alias for convenience
AgentDict = Dict[str, HeterogeneousAgent]
