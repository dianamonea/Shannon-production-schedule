"""
Decentralized Task Allocation (CBBA-style)

Implements a lightweight consensus-based bundle algorithm for multi-agent
allocation without central coordinator.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class Task:
    task_id: str
    reward: float
    duration: float
    location: Tuple[float, float]


@dataclass
class Agent:
    agent_id: str
    position: Tuple[float, float]
    speed: float
    max_tasks: int


@dataclass
class BundleState:
    bundle: List[str] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    bids: Dict[str, float] = field(default_factory=dict)
    winners: Dict[str, str] = field(default_factory=dict)


class CBBAAllocator:
    """Consensus-Based Bundle Algorithm (simplified)."""

    def __init__(self, tasks: List[Task], agents: List[Agent]):
        self.tasks = {t.task_id: t for t in tasks}
        self.agents = {a.agent_id: a for a in agents}
        self.states = {a.agent_id: BundleState() for a in agents}

    def allocate(self, rounds: int = 10) -> Dict[str, List[str]]:
        for _ in range(rounds):
            for agent_id in self.agents:
                self._build_bundle(agent_id)
            self._consensus_phase()

        return {aid: state.path for aid, state in self.states.items()}

    def _build_bundle(self, agent_id: str):
        agent = self.agents[agent_id]
        state = self.states[agent_id]

        while len(state.bundle) < agent.max_tasks:
            best_task = None
            best_bid = -math.inf

            for task_id, task in self.tasks.items():
                if task_id in state.bundle:
                    continue
                if task_id in state.winners and state.winners[task_id] != agent_id:
                    continue

                bid = self._compute_bid(agent, task)
                if bid > best_bid:
                    best_bid = bid
                    best_task = task_id

            if best_task is None:
                break

            state.bundle.append(best_task)
            state.path.append(best_task)
            state.bids[best_task] = best_bid
            state.winners[best_task] = agent_id

    def _compute_bid(self, agent: Agent, task: Task) -> float:
        dist = math.hypot(agent.position[0] - task.location[0], agent.position[1] - task.location[1])
        travel_time = dist / max(agent.speed, 1e-6)
        cost = travel_time + task.duration
        return task.reward - cost

    def _consensus_phase(self):
        # Resolve conflicts by highest bid
        all_bids = []
        for agent_id, state in self.states.items():
            for task_id, bid in state.bids.items():
                all_bids.append((task_id, bid, agent_id))

        # Determine winners
        winners: Dict[str, Tuple[float, str]] = {}
        for task_id, bid, agent_id in all_bids:
            if task_id not in winners or bid > winners[task_id][0]:
                winners[task_id] = (bid, agent_id)

        # Update states
        for task_id, (_, winner) in winners.items():
            for agent_id, state in self.states.items():
                if agent_id != winner and task_id in state.bundle:
                    state.bundle.remove(task_id)
                    if task_id in state.path:
                        state.path.remove(task_id)
                state.winners[task_id] = winner
