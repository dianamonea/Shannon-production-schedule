"""
Heterogeneous Embodiment Coordinator for Multi-Robot Systems

Coordinates task execution across robots with different embodiments:
- AGV (Automated Guided Vehicle): Material transport
- Manipulator (Robotic Arm): Pick-and-place, assembly
- Dual-Arm: Coordinated bimanual manipulation
- Mobile Manipulator: Combined mobility + manipulation

Handles:
1. Task decomposition for composite operations
2. Synchronization protocols (handoff, coordination)
3. Deadlock detection and resolution
4. Capability-based task allocation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .path_planning import GridMap, plan_multi_agent_paths
except Exception:
    GridMap = None
    plan_multi_agent_paths = None

try:
    from .decentralized_allocation import CBBAAllocator, Task as CBTask, Agent as CBAgent
except Exception:
    CBBAAllocator = None
    CBTask = None
    CBAgent = None


class EmbodimentType(Enum):
    """Robot embodiment types"""
    AGV = "AGV"
    SINGLE_ARM = "SINGLE_ARM"
    DUAL_ARM = "DUAL_ARM"
    MOBILE_MANIPULATOR = "MOBILE_MANIPULATOR"
    INSPECTION_ROBOT = "INSPECTION_ROBOT"


class TaskType(Enum):
    """Task types requiring coordination"""
    TRANSPORT = "TRANSPORT"
    PICK_AND_PLACE = "PICK_AND_PLACE"
    BIMANUAL_ASSEMBLY = "BIMANUAL_ASSEMBLY"
    MOBILE_MANIPULATION = "MOBILE_MANIPULATION"
    HANDOFF = "HANDOFF"
    COORDINATED_INSPECTION = "COORDINATED_INSPECTION"


@dataclass
class Capability:
    """Robot capability specification"""
    embodiment: EmbodimentType
    max_payload: float  # kg
    reach: float  # meters
    mobility: bool
    dual_arm: bool
    vision_system: bool
    force_sensing: bool


@dataclass
class Robot:
    """Robot agent representation"""
    robot_id: str
    embodiment: EmbodimentType
    capability: Capability
    current_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_busy: bool = False
    current_task_id: Optional[str] = None
    locked_resources: Set[str] = field(default_factory=set)


@dataclass
class CompositeTask:
    """Task requiring multiple robots"""
    task_id: str
    task_type: TaskType
    subtasks: List['Subtask'] = field(default_factory=list)
    required_capabilities: List[EmbodimentType] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "PENDING"  # PENDING, DECOMPOSED, IN_PROGRESS, COMPLETED, FAILED


@dataclass
class Subtask:
    """Individual subtask for a single robot"""
    subtask_id: str
    parent_task_id: str
    assigned_robot_id: Optional[str] = None
    required_embodiment: EmbodimentType = EmbodimentType.SINGLE_ARM
    target_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    payload: float = 0.0
    duration_estimate: float = 0.0  # seconds
    dependencies: List[str] = field(default_factory=list)  # Subtask IDs
    status: str = "PENDING"


@dataclass
class Handoff:
    """Handoff operation between robots"""
    handoff_id: str
    from_robot_id: str
    to_robot_id: str
    handoff_position: Tuple[float, float, float]
    object_id: str
    scheduled_time: datetime
    tolerance_window: timedelta = timedelta(seconds=5)
    status: str = "SCHEDULED"  # SCHEDULED, IN_PROGRESS, COMPLETED, FAILED


class EmbodimentCoordinator:
    """
    Coordinates task execution across heterogeneous robots
    """
    
    def __init__(self):
        self.robots: Dict[str, Robot] = {}
        self.tasks: Dict[str, CompositeTask] = {}
        self.handoffs: Dict[str, Handoff] = {}
        self.resource_locks: Dict[str, str] = {}  # resource_id -> robot_id
        
        # Deadlock detection
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)

        # Optional grid map for AGV planning
        self.grid_map: Optional[GridMap] = None
        
    def register_robot(self, robot: Robot):
        """Register a robot with the coordinator"""
        self.robots[robot.robot_id] = robot
        logger.info(f"Registered robot {robot.robot_id} ({robot.embodiment.value})")
        
    def decompose_task(self, task: CompositeTask) -> List[Subtask]:
        """
        Decompose composite task into subtasks based on task type
        """
        subtasks = []
        
        if task.task_type == TaskType.MOBILE_MANIPULATION:
            # Decompose: AGV transport → Manipulator picks up from AGV → Manipulator places
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_TRANSPORT",
                parent_task_id=task.task_id,
                required_embodiment=EmbodimentType.AGV,
                target_position=(5.0, 3.0, 0.0),
                duration_estimate=30.0,
                dependencies=[]
            ))
            
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_HANDOFF_PICKUP",
                parent_task_id=task.task_id,
                required_embodiment=EmbodimentType.SINGLE_ARM,
                target_position=(5.0, 3.0, 0.5),
                payload=2.0,
                duration_estimate=10.0,
                dependencies=[f"{task.task_id}_TRANSPORT"]
            ))
            
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_PLACE",
                parent_task_id=task.task_id,
                required_embodiment=EmbodimentType.SINGLE_ARM,
                target_position=(10.0, 5.0, 0.8),
                duration_estimate=15.0,
                dependencies=[f"{task.task_id}_HANDOFF_PICKUP"]
            ))
            
        elif task.task_type == TaskType.BIMANUAL_ASSEMBLY:
            # Dual-arm coordinated task
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_DUAL_ARM_ASSEMBLY",
                parent_task_id=task.task_id,
                required_embodiment=EmbodimentType.DUAL_ARM,
                target_position=(8.0, 4.0, 0.6),
                duration_estimate=45.0,
                dependencies=[]
            ))
            
        elif task.task_type == TaskType.HANDOFF:
            # AGV → Manipulator handoff
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_AGV_DELIVER",
                parent_task_id=task.task_id,
                required_embodiment=EmbodimentType.AGV,
                target_position=(5.0, 3.0, 0.0),
                duration_estimate=20.0,
                dependencies=[]
            ))
            
            subtasks.append(Subtask(
                subtask_id=f"{task.task_id}_ARM_RECEIVE",
                parent_task_id=task.task_id,
                required_embodiment=EmbodimentType.SINGLE_ARM,
                target_position=(5.0, 3.0, 0.5),
                duration_estimate=10.0,
                dependencies=[f"{task.task_id}_AGV_DELIVER"]
            ))
        
        task.subtasks = subtasks
        task.status = "DECOMPOSED"
        logger.info(f"Decomposed task {task.task_id} into {len(subtasks)} subtasks")
        
        return subtasks
    
    def allocate_robots(self, subtasks: List[Subtask]) -> Dict[str, str]:
        """
        Allocate robots to subtasks based on capabilities and availability
        
        Returns:
            Dict mapping subtask_id -> robot_id
        """
        allocation = {}
        
        for subtask in subtasks:
            # Find available robots with required embodiment
            candidates = [
                robot for robot in self.robots.values()
                if robot.embodiment == subtask.required_embodiment
                and not robot.is_busy
                and robot.capability.max_payload >= subtask.payload
            ]
            
            if not candidates:
                logger.warning(f"No available robot for subtask {subtask.subtask_id}")
                continue
            
            # Select closest robot
            best_robot = min(
                candidates,
                key=lambda r: self._distance(r.current_position, subtask.target_position)
            )
            
            allocation[subtask.subtask_id] = best_robot.robot_id
            subtask.assigned_robot_id = best_robot.robot_id
            best_robot.is_busy = True
            best_robot.current_task_id = subtask.parent_task_id
            
            logger.info(f"Allocated robot {best_robot.robot_id} to subtask {subtask.subtask_id}")
        
        return allocation
    
    def schedule_handoff(self, from_robot_id: str, to_robot_id: str, 
                        object_id: str, position: Tuple[float, float, float]) -> Handoff:
        """
        Schedule a handoff operation between two robots
        """
        handoff = Handoff(
            handoff_id=f"HANDOFF_{datetime.now().timestamp()}",
            from_robot_id=from_robot_id,
            to_robot_id=to_robot_id,
            handoff_position=position,
            object_id=object_id,
            scheduled_time=datetime.now() + timedelta(seconds=10)
        )
        
        self.handoffs[handoff.handoff_id] = handoff
        logger.info(f"Scheduled handoff {handoff.handoff_id} at position {position}")
        
        return handoff
    
    async def execute_handoff(self, handoff: Handoff) -> bool:
        """
        Execute synchronized handoff operation
        """
        logger.info(f"Executing handoff {handoff.handoff_id}")
        handoff.status = "IN_PROGRESS"
        
        from_robot = self.robots[handoff.from_robot_id]
        to_robot = self.robots[handoff.to_robot_id]
        
        # Simulate synchronized movement to handoff position
        await asyncio.sleep(2.0)  # Movement time
        
        # Check synchronization (both robots within tolerance)
        from_distance = self._distance(from_robot.current_position, handoff.handoff_position)
        to_distance = self._distance(to_robot.current_position, handoff.handoff_position)
        
        if from_distance < 0.1 and to_distance < 0.1:
            # Successful handoff
            from_robot.current_position = handoff.handoff_position
            to_robot.current_position = handoff.handoff_position
            handoff.status = "COMPLETED"
            logger.info(f"Handoff {handoff.handoff_id} completed successfully")
            return True
        else:
            handoff.status = "FAILED"
            logger.error(f"Handoff {handoff.handoff_id} failed: robots not synchronized")
            return False
    
    def detect_deadlock(self) -> Optional[List[str]]:
        """
        Detect circular dependencies in subtask execution
        
        Returns:
            List of subtask IDs in deadlock cycle, or None
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, path.copy())
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
            
            rec_stack.remove(node)
            return None
        
        for node in self.dependency_graph:
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    logger.warning(f"Deadlock detected: {cycle}")
                    return cycle
        
        return None
    
    def resolve_deadlock(self, cycle: List[str]):
        """
        Resolve deadlock by breaking the weakest dependency
        """
        # Simple strategy: break the last dependency in cycle
        if len(cycle) < 2:
            return
        
        weak_link = (cycle[-1], cycle[0])
        if weak_link[0] in self.dependency_graph:
            self.dependency_graph[weak_link[0]].discard(weak_link[1])
            logger.info(f"Broke dependency {weak_link[0]} -> {weak_link[1]} to resolve deadlock")
    
    def lock_resource(self, robot_id: str, resource_id: str) -> bool:
        """
        Acquire exclusive lock on shared resource
        """
        if resource_id in self.resource_locks:
            logger.warning(f"Resource {resource_id} already locked by {self.resource_locks[resource_id]}")
            return False
        
        self.resource_locks[resource_id] = robot_id
        self.robots[robot_id].locked_resources.add(resource_id)
        logger.info(f"Robot {robot_id} locked resource {resource_id}")
        return True
    
    def release_resource(self, robot_id: str, resource_id: str):
        """Release resource lock"""
        if resource_id in self.resource_locks and self.resource_locks[resource_id] == robot_id:
            del self.resource_locks[resource_id]
            self.robots[robot_id].locked_resources.discard(resource_id)
            logger.info(f"Robot {robot_id} released resource {resource_id}")

    def set_grid_map(self, width: int, height: int, obstacles: Set[Tuple[int, int]]):
        """Initialize grid map for AGV path planning"""
        if GridMap is None:
            logger.warning("GridMap unavailable; path planning disabled")
            return
        self.grid_map = GridMap(width=width, height=height, obstacles=obstacles)

    def plan_agv_paths(self, starts: Dict[str, Tuple[int, int]], goals: Dict[str, Tuple[int, int]]):
        """Plan collision-free paths for AGVs using prioritized planning"""
        if self.grid_map is None or plan_multi_agent_paths is None:
            logger.warning("Path planner not configured")
            return {}
        return plan_multi_agent_paths(self.grid_map, starts, goals)

    def decentralized_allocate(self, tasks: List[Dict], agents: List[Dict]) -> Dict[str, List[str]]:
        """Decentralized task allocation using CBBA (if available)."""
        if CBBAAllocator is None:
            logger.warning("CBBA allocator unavailable")
            return {}

        cb_tasks = [
            CBTask(
                task_id=t["task_id"],
                reward=t.get("reward", 1.0),
                duration=t.get("duration", 1.0),
                location=tuple(t.get("location", (0.0, 0.0))),
            )
            for t in tasks
        ]

        cb_agents = [
            CBAgent(
                agent_id=a["agent_id"],
                position=tuple(a.get("position", (0.0, 0.0))),
                speed=a.get("speed", 1.0),
                max_tasks=a.get("max_tasks", 3),
            )
            for a in agents
        ]

        allocator = CBBAAllocator(cb_tasks, cb_agents)
        return allocator.allocate()
    
    def _distance(self, pos1: Tuple[float, float, float], 
                  pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between positions"""
        return ((pos1[0] - pos2[0])**2 + 
                (pos1[1] - pos2[1])**2 + 
                (pos1[2] - pos2[2])**2) ** 0.5
    
    async def execute_composite_task(self, task: CompositeTask) -> bool:
        """
        Execute a composite task with full coordination
        """
        logger.info(f"Starting composite task {task.task_id} ({task.task_type.value})")
        task.start_time = datetime.now()
        task.status = "IN_PROGRESS"
        
        # 1. Decompose task
        subtasks = self.decompose_task(task)
        
        # 2. Allocate robots
        allocation = self.allocate_robots(subtasks)
        
        if len(allocation) < len(subtasks):
            logger.error(f"Failed to allocate all subtasks for {task.task_id}")
            task.status = "FAILED"
            return False
        
        # 3. Build dependency graph
        for subtask in subtasks:
            for dep in subtask.dependencies:
                self.dependency_graph[dep].add(subtask.subtask_id)
        
        # 4. Check for deadlocks
        cycle = self.detect_deadlock()
        if cycle:
            self.resolve_deadlock(cycle)
        
        # 5. Execute subtasks in dependency order
        completed = set()
        
        while len(completed) < len(subtasks):
            # Find subtasks ready to execute
            ready = [
                st for st in subtasks
                if st.status == "PENDING"
                and all(dep in completed for dep in st.dependencies)
            ]
            
            if not ready:
                logger.error("No subtasks ready - possible deadlock")
                break
            
            # Execute ready subtasks in parallel
            tasks_to_execute = []
            for subtask in ready:
                subtask.status = "IN_PROGRESS"
                tasks_to_execute.append(self._execute_subtask(subtask))
            
            results = await asyncio.gather(*tasks_to_execute)
            
            for subtask, success in zip(ready, results):
                if success:
                    subtask.status = "COMPLETED"
                    completed.add(subtask.subtask_id)
                    
                    # Release robot
                    robot = self.robots[subtask.assigned_robot_id]
                    robot.is_busy = False
                    robot.current_task_id = None
                else:
                    subtask.status = "FAILED"
                    logger.error(f"Subtask {subtask.subtask_id} failed")
        
        # Clean up dependency graph
        for subtask in subtasks:
            if subtask.subtask_id in self.dependency_graph:
                del self.dependency_graph[subtask.subtask_id]
        
        task.end_time = datetime.now()
        task.status = "COMPLETED" if len(completed) == len(subtasks) else "FAILED"
        
        logger.info(f"Task {task.task_id} {task.status.lower()}: {len(completed)}/{len(subtasks)} subtasks completed")
        
        return task.status == "COMPLETED"
    
    async def _execute_subtask(self, subtask: Subtask) -> bool:
        """Execute a single subtask (simulation)"""
        logger.info(f"Executing subtask {subtask.subtask_id} on robot {subtask.assigned_robot_id}")
        await asyncio.sleep(subtask.duration_estimate / 10.0)  # Simulated execution
        return True


# Example usage
async def main():
    coordinator = EmbodimentCoordinator()
    
    # Register heterogeneous robots
    coordinator.register_robot(Robot(
        robot_id="AGV_001",
        embodiment=EmbodimentType.AGV,
        capability=Capability(
            embodiment=EmbodimentType.AGV,
            max_payload=50.0,
            reach=0.0,
            mobility=True,
            dual_arm=False,
            vision_system=True,
            force_sensing=False
        )
    ))
    
    coordinator.register_robot(Robot(
        robot_id="ARM_001",
        embodiment=EmbodimentType.SINGLE_ARM,
        capability=Capability(
            embodiment=EmbodimentType.SINGLE_ARM,
            max_payload=10.0,
            reach=1.5,
            mobility=False,
            dual_arm=False,
            vision_system=True,
            force_sensing=True
        )
    ))
    
    coordinator.register_robot(Robot(
        robot_id="DUAL_ARM_001",
        embodiment=EmbodimentType.DUAL_ARM,
        capability=Capability(
            embodiment=EmbodimentType.DUAL_ARM,
            max_payload=15.0,
            reach=1.2,
            mobility=False,
            dual_arm=True,
            vision_system=True,
            force_sensing=True
        )
    ))
    
    # Create composite task
    task = CompositeTask(
        task_id="TASK_001",
        task_type=TaskType.MOBILE_MANIPULATION,
        required_capabilities=[EmbodimentType.AGV, EmbodimentType.SINGLE_ARM]
    )
    
    # Execute task
    success = await coordinator.execute_composite_task(task)
    print(f"\nTask execution {'succeeded' if success else 'failed'}")


if __name__ == "__main__":
    asyncio.run(main())
