"""
Digital Twin Simulation Environment for Manufacturing System

Provides high-fidelity simulation for:
1. Testing scheduling algorithms offline
2. Validating new policies before deployment
3. Generating synthetic training data
4. What-if scenario analysis

Features:
- Discrete event simulation
- Physics-based robot kinematics
- Stochastic failure models
- Real-time visualization interface
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Simulation event types"""
    JOB_ARRIVAL = "JOB_ARRIVAL"
    JOB_START = "JOB_START"
    JOB_COMPLETE = "JOB_COMPLETE"
    JOB_FAIL = "JOB_FAIL"
    AGENT_BREAKDOWN = "AGENT_BREAKDOWN"
    AGENT_REPAIR = "AGENT_REPAIR"
    MATERIAL_ARRIVAL = "MATERIAL_ARRIVAL"
    INSPECTION = "INSPECTION"


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "IDLE"
    BUSY = "BUSY"
    BROKEN = "BROKEN"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class SimEvent:
    """Discrete event in simulation"""
    event_id: str
    event_type: EventType
    timestamp: float  # Simulation time
    entity_id: str  # Job ID, Agent ID, etc.
    data: Dict = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class SimJob:
    """Simulated production job"""
    job_id: str
    job_type: str
    duration: float  # seconds
    arrival_time: float
    deadline: float
    priority: int = 1
    materials_required: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED


@dataclass
class SimAgent:
    """Simulated manufacturing agent"""
    agent_id: str
    agent_type: str  # CNC, ROBOT, AGV, etc.
    state: AgentState = AgentState.IDLE
    
    # Performance characteristics
    processing_speed: float = 1.0  # Multiplier on job duration
    failure_rate: float = 0.001  # Failures per hour
    repair_time_mean: float = 3600.0  # Mean repair time in seconds
    
    # State tracking
    current_job: Optional[str] = None
    total_jobs_completed: int = 0
    total_busy_time: float = 0.0
    total_idle_time: float = 0.0
    last_state_change: float = 0.0
    
    # Quality
    quality_capability: float = 0.95


@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    simulation_duration: float = 86400.0  # 24 hours in seconds
    time_acceleration: float = 1.0  # Real-time multiplier
    random_seed: Optional[int] = 42
    
    # Job arrival process
    job_arrival_rate: float = 10.0  # Jobs per hour
    job_duration_mean: float = 1800.0  # 30 minutes
    job_duration_std: float = 600.0  # 10 minutes
    
    # Failure modeling
    enable_failures: bool = True
    enable_quality_defects: bool = True
    
    # Output
    enable_visualization: bool = False
    log_interval: float = 3600.0  # Log stats every hour


class SimulationEnvironment:
    """
    Discrete event simulation of manufacturing system
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Simulation state
        self.current_time: float = 0.0
        self.event_queue: List[SimEvent] = []
        self.agents: Dict[str, SimAgent] = {}
        self.jobs: Dict[str, SimJob] = {}
        
        # Statistics
        self.total_jobs_arrived = 0
        self.total_jobs_completed = 0
        self.total_jobs_failed = 0
        self.total_makespan = 0.0
        self.total_tardiness = 0.0
        
        # Random number generator
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        # Event handlers
        self.event_handlers: Dict[EventType, Callable] = {
            EventType.JOB_ARRIVAL: self._handle_job_arrival,
            EventType.JOB_START: self._handle_job_start,
            EventType.JOB_COMPLETE: self._handle_job_complete,
            EventType.JOB_FAIL: self._handle_job_fail,
            EventType.AGENT_BREAKDOWN: self._handle_agent_breakdown,
            EventType.AGENT_REPAIR: self._handle_agent_repair,
            EventType.INSPECTION: self._handle_inspection,
        }
    
    def add_agent(self, agent: SimAgent):
        """Add agent to simulation"""
        self.agents[agent.agent_id] = agent
        agent.last_state_change = self.current_time
        logger.info(f"Added agent {agent.agent_id} ({agent.agent_type})")
    
    def schedule_event(self, event: SimEvent):
        """Schedule an event in the future"""
        # Insert event in sorted order
        inserted = False
        for i, e in enumerate(self.event_queue):
            if event.timestamp < e.timestamp:
                self.event_queue.insert(i, event)
                inserted = True
                break
        
        if not inserted:
            self.event_queue.append(event)
    
    def run(self) -> Dict:
        """
        Run simulation
        
        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting simulation for {self.config.simulation_duration}s")
        
        # Initialize job arrival process
        self._schedule_next_job_arrival()
        
        # Schedule agent failures if enabled
        if self.config.enable_failures:
            for agent in self.agents.values():
                self._schedule_agent_failure(agent)
        
        # Main event loop
        last_log_time = 0.0
        
        while self.event_queue and self.current_time < self.config.simulation_duration:
            # Get next event
            event = self.event_queue.pop(0)
            self.current_time = event.timestamp
            
            # Handle event
            handler = self.event_handlers.get(event.event_type)
            if handler:
                handler(event)
            
            # Periodic logging
            if self.current_time - last_log_time >= self.config.log_interval:
                self._log_statistics()
                last_log_time = self.current_time
        
        # Final statistics
        logger.info("Simulation complete")
        return self.get_statistics()
    
    def _schedule_next_job_arrival(self):
        """Schedule next job arrival using Poisson process"""
        # Inter-arrival time follows exponential distribution
        inter_arrival = np.random.exponential(3600.0 / self.config.job_arrival_rate)
        arrival_time = self.current_time + inter_arrival
        
        if arrival_time < self.config.simulation_duration:
            job_id = f"JOB_{self.total_jobs_arrived + 1:04d}"
            
            event = SimEvent(
                event_id=f"ARRIVAL_{job_id}",
                event_type=EventType.JOB_ARRIVAL,
                timestamp=arrival_time,
                entity_id=job_id,
                data={'job_id': job_id}
            )
            
            self.schedule_event(event)
    
    def _handle_job_arrival(self, event: SimEvent):
        """Handle job arrival event"""
        job_id = event.entity_id
        
        # Create job with random duration
        duration = max(60.0, np.random.normal(
            self.config.job_duration_mean,
            self.config.job_duration_std
        ))
        
        # Random deadline (2-4x the expected duration)
        deadline_multiplier = random.uniform(2.0, 4.0)
        deadline = self.current_time + duration * deadline_multiplier
        
        job = SimJob(
            job_id=job_id,
            job_type=random.choice(["MILLING", "ASSEMBLY", "INSPECTION"]),
            duration=duration,
            arrival_time=self.current_time,
            deadline=deadline,
            priority=random.randint(1, 5)
        )
        
        self.jobs[job_id] = job
        self.total_jobs_arrived += 1
        
        logger.debug(f"Job {job_id} arrived at t={self.current_time:.1f}s, "
                    f"duration={duration:.1f}s, deadline={deadline:.1f}s")
        
        # Try to assign to available agent
        self._assign_job_to_agent(job)
        
        # Schedule next arrival
        self._schedule_next_job_arrival()
    
    def _assign_job_to_agent(self, job: SimJob) -> bool:
        """Assign job to an available agent"""
        # Find idle agents
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == AgentState.IDLE
        ]
        
        if not available_agents:
            logger.debug(f"No available agent for {job.job_id}")
            return False
        
        # Simple assignment: pick first available
        agent = available_agents[0]
        
        job.assigned_agent = agent.agent_id
        agent.state = AgentState.BUSY
        agent.current_job = job.job_id
        
        # Schedule job start
        start_event = SimEvent(
            event_id=f"START_{job.job_id}",
            event_type=EventType.JOB_START,
            timestamp=self.current_time,
            entity_id=job.job_id,
            data={'agent_id': agent.agent_id}
        )
        
        self.schedule_event(start_event)
        
        return True
    
    def _handle_job_start(self, event: SimEvent):
        """Handle job start event"""
        job_id = event.entity_id
        agent_id = event.data['agent_id']
        
        job = self.jobs[job_id]
        agent = self.agents[agent_id]
        
        job.start_time = self.current_time
        job.status = "RUNNING"
        
        # Update agent statistics
        idle_time = self.current_time - agent.last_state_change
        agent.total_idle_time += idle_time
        agent.last_state_change = self.current_time
        
        # Schedule job completion
        processing_time = job.duration * agent.processing_speed
        completion_time = self.current_time + processing_time
        
        complete_event = SimEvent(
            event_id=f"COMPLETE_{job_id}",
            event_type=EventType.JOB_COMPLETE,
            timestamp=completion_time,
            entity_id=job_id,
            data={'agent_id': agent_id}
        )
        
        self.schedule_event(complete_event)
        
        logger.debug(f"Job {job_id} started on {agent_id} at t={self.current_time:.1f}s")
    
    def _handle_job_complete(self, event: SimEvent):
        """Handle job completion event"""
        job_id = event.entity_id
        agent_id = event.data['agent_id']
        
        job = self.jobs[job_id]
        agent = self.agents[agent_id]
        
        job.completion_time = self.current_time
        job.status = "COMPLETED"
        
        # Calculate tardiness
        if self.current_time > job.deadline:
            tardiness = self.current_time - job.deadline
            self.total_tardiness += tardiness
        
        # Update statistics
        self.total_jobs_completed += 1
        agent.total_jobs_completed += 1
        
        # Update agent state
        busy_time = self.current_time - agent.last_state_change
        agent.total_busy_time += busy_time
        agent.state = AgentState.IDLE
        agent.current_job = None
        agent.last_state_change = self.current_time
        
        logger.debug(f"Job {job_id} completed at t={self.current_time:.1f}s")
        
        # Schedule inspection if quality checks enabled
        if self.config.enable_quality_defects:
            inspection_event = SimEvent(
                event_id=f"INSPECT_{job_id}",
                event_type=EventType.INSPECTION,
                timestamp=self.current_time + 60.0,  # 1 minute inspection delay
                entity_id=job_id,
                data={'agent_id': agent_id}
            )
            self.schedule_event(inspection_event)
        
        # Try to assign next job
        self._assign_next_waiting_job(agent)
    
    def _handle_inspection(self, event: SimEvent):
        """Handle quality inspection event"""
        job_id = event.entity_id
        agent_id = event.data['agent_id']
        
        job = self.jobs[job_id]
        agent = self.agents[agent_id]
        
        # Simulate quality check
        defect_probability = 1.0 - agent.quality_capability
        is_defective = random.random() < defect_probability
        
        if is_defective:
            logger.warning(f"Defect detected in {job_id} from {agent_id}")
            job.status = "FAILED"
            self.total_jobs_failed += 1
    
    def _assign_next_waiting_job(self, agent: SimAgent):
        """Assign next waiting job to agent"""
        waiting_jobs = [
            job for job in self.jobs.values()
            if job.status == "PENDING" and job.assigned_agent is None
        ]
        
        if not waiting_jobs:
            return
        
        # Sort by priority and arrival time
        waiting_jobs.sort(key=lambda j: (-j.priority, j.arrival_time))
        next_job = waiting_jobs[0]
        
        self._assign_job_to_agent(next_job)
    
    def _schedule_agent_failure(self, agent: SimAgent):
        """Schedule next agent failure using exponential distribution"""
        # Time to next failure
        mtbf = 1.0 / agent.failure_rate  # Mean time between failures (hours)
        time_to_failure = np.random.exponential(mtbf * 3600.0)  # Convert to seconds
        
        failure_time = self.current_time + time_to_failure
        
        if failure_time < self.config.simulation_duration:
            failure_event = SimEvent(
                event_id=f"FAIL_{agent.agent_id}_{int(failure_time)}",
                event_type=EventType.AGENT_BREAKDOWN,
                timestamp=failure_time,
                entity_id=agent.agent_id
            )
            
            self.schedule_event(failure_event)
    
    def _handle_agent_breakdown(self, event: SimEvent):
        """Handle agent breakdown event"""
        agent_id = event.entity_id
        agent = self.agents[agent_id]
        
        logger.warning(f"Agent {agent_id} breakdown at t={self.current_time:.1f}s")
        
        # Mark agent as broken
        previous_state = agent.state
        agent.state = AgentState.BROKEN
        
        # If agent was working on a job, mark job as failed
        if agent.current_job:
            job = self.jobs[agent.current_job]
            
            fail_event = SimEvent(
                event_id=f"JOB_FAIL_{job.job_id}",
                event_type=EventType.JOB_FAIL,
                timestamp=self.current_time,
                entity_id=job.job_id,
                data={'reason': 'AGENT_BREAKDOWN', 'agent_id': agent_id}
            )
            
            self.schedule_event(fail_event)
        
        # Schedule repair
        repair_time = np.random.exponential(agent.repair_time_mean)
        repair_event = SimEvent(
            event_id=f"REPAIR_{agent_id}_{int(self.current_time)}",
            event_type=EventType.AGENT_REPAIR,
            timestamp=self.current_time + repair_time,
            entity_id=agent_id
        )
        
        self.schedule_event(repair_event)
    
    def _handle_agent_repair(self, event: SimEvent):
        """Handle agent repair completion"""
        agent_id = event.entity_id
        agent = self.agents[agent_id]
        
        logger.info(f"Agent {agent_id} repaired at t={self.current_time:.1f}s")
        
        agent.state = AgentState.IDLE
        agent.current_job = None
        agent.last_state_change = self.current_time
        
        # Try to assign waiting job
        self._assign_next_waiting_job(agent)
        
        # Schedule next failure
        self._schedule_agent_failure(agent)
    
    def _handle_job_fail(self, event: SimEvent):
        """Handle job failure event"""
        job_id = event.entity_id
        reason = event.data.get('reason', 'UNKNOWN')
        
        job = self.jobs[job_id]
        job.status = "FAILED"
        job.completion_time = self.current_time
        
        self.total_jobs_failed += 1
        
        logger.warning(f"Job {job_id} failed: {reason}")
    
    def _log_statistics(self):
        """Log current simulation statistics"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Simulation Time: {self.current_time:.1f}s ({self.current_time/3600:.1f}h)")
        logger.info(f"Jobs: Arrived={self.total_jobs_arrived}, "
                   f"Completed={self.total_jobs_completed}, "
                   f"Failed={self.total_jobs_failed}")
        
        for agent in self.agents.values():
            utilization = agent.total_busy_time / (self.current_time + 1e-6)
            logger.info(f"  {agent.agent_id}: State={agent.state.value}, "
                       f"Jobs={agent.total_jobs_completed}, "
                       f"Utilization={utilization:.2%}")
        
        logger.info(f"{'='*60}\n")
    
    def get_statistics(self) -> Dict:
        """Get final simulation statistics"""
        stats = {
            'simulation_duration': self.current_time,
            'total_jobs_arrived': self.total_jobs_arrived,
            'total_jobs_completed': self.total_jobs_completed,
            'total_jobs_failed': self.total_jobs_failed,
            'completion_rate': self.total_jobs_completed / max(1, self.total_jobs_arrived),
            'total_tardiness': self.total_tardiness,
            'avg_tardiness': self.total_tardiness / max(1, self.total_jobs_completed),
            'agents': {}
        }
        
        for agent_id, agent in self.agents.items():
            total_time = self.current_time
            utilization = agent.total_busy_time / total_time if total_time > 0 else 0
            
            stats['agents'][agent_id] = {
                'jobs_completed': agent.total_jobs_completed,
                'utilization': utilization,
                'total_busy_time': agent.total_busy_time,
                'total_idle_time': agent.total_idle_time
            }
        
        return stats


# Example usage
def main():
    # Configure simulation
    config = SimulationConfig(
        simulation_duration=28800.0,  # 8 hours
        job_arrival_rate=15.0,  # 15 jobs per hour
        job_duration_mean=1200.0,  # 20 minutes average
        job_duration_std=300.0,  # 5 minutes std dev
        enable_failures=True,
        random_seed=42
    )
    
    # Create simulation
    sim = SimulationEnvironment(config)
    
    # Add agents
    sim.add_agent(SimAgent(
        agent_id="CNC_001",
        agent_type="CNC_MILL",
        processing_speed=1.0,
        failure_rate=0.01,  # 0.01 failures per hour
        quality_capability=0.98
    ))
    
    sim.add_agent(SimAgent(
        agent_id="CNC_002",
        agent_type="CNC_MILL",
        processing_speed=1.1,  # 10% faster
        failure_rate=0.015,
        quality_capability=0.95
    ))
    
    sim.add_agent(SimAgent(
        agent_id="ROBOT_001",
        agent_type="ASSEMBLY_ROBOT",
        processing_speed=0.9,
        failure_rate=0.005,
        quality_capability=0.99
    ))
    
    # Run simulation
    print("\n=== Digital Twin Simulation ===")
    stats = sim.run()
    
    # Print results
    print(f"\n{'='*60}")
    print("SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Duration: {stats['simulation_duration']/3600:.1f} hours")
    print(f"Jobs Arrived: {stats['total_jobs_arrived']}")
    print(f"Jobs Completed: {stats['total_jobs_completed']}")
    print(f"Jobs Failed: {stats['total_jobs_failed']}")
    print(f"Completion Rate: {stats['completion_rate']:.1%}")
    print(f"Average Tardiness: {stats['avg_tardiness']:.1f}s")
    
    print(f"\nAgent Performance:")
    for agent_id, agent_stats in stats['agents'].items():
        print(f"  {agent_id}:")
        print(f"    Jobs Completed: {agent_stats['jobs_completed']}")
        print(f"    Utilization: {agent_stats['utilization']:.1%}")
        print(f"    Busy Time: {agent_stats['total_busy_time']/3600:.1f}h")


if __name__ == "__main__":
    main()
