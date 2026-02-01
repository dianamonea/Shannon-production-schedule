"""
Multi-Agent Path Planning (MAPF) with Prioritized Planning

Implements:
- Grid-based A* planner
- Reservation table for time-expanded collision avoidance
- Prioritized planning for multiple AGVs

This is a lightweight, dependency-free implementation suitable for
integration with the embodiment coordinator.
"""

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set


@dataclass
class GridMap:
    width: int
    height: int
    obstacles: Set[Tuple[int, int]]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.obstacles

    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        candidates = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        results = []
        for nx, ny in candidates:
            if self.in_bounds(nx, ny) and self.passable(nx, ny):
                results.append((nx, ny))
        return results


class ReservationTable:
    """Time-expanded reservation table to avoid collisions."""

    def __init__(self):
        self.reservations: Dict[int, Set[Tuple[int, int]]] = {}
        self.edge_reservations: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}

    def reserve(self, path: List[Tuple[int, int]]):
        for t, pos in enumerate(path):
            if t not in self.reservations:
                self.reservations[t] = set()
            self.reservations[t].add(pos)
            if t > 0:
                if t not in self.edge_reservations:
                    self.edge_reservations[t] = set()
                self.edge_reservations[t].add((path[t - 1], pos))

    def is_reserved(self, pos: Tuple[int, int], t: int) -> bool:
        return t in self.reservations and pos in self.reservations[t]

    def is_edge_reserved(self, prev: Tuple[int, int], curr: Tuple[int, int], t: int) -> bool:
        # Prevent swap conflicts: another agent moves curr->prev at same timestep
        return t in self.edge_reservations and (curr, prev) in self.edge_reservations[t]


class AStarPlanner:
    def __init__(self, grid: GridMap, reservation: Optional[ReservationTable] = None):
        self.grid = grid
        self.reservation = reservation

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int], max_time: int = 200) -> Optional[List[Tuple[int, int]]]:
        frontier = []
        heapq.heappush(frontier, (0, 0, start))
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}
        cost_so_far: Dict[Tuple[int, int, int], int] = {}

        start_state = (start[0], start[1], 0)
        came_from[start_state] = None
        cost_so_far[start_state] = 0

        while frontier:
            _, t, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct_path(came_from, (current[0], current[1], t))

            if t >= max_time:
                continue

            for nx, ny in self.grid.neighbors(current[0], current[1]):
                nt = t + 1
                if self.reservation and self.reservation.is_reserved((nx, ny), nt):
                    continue
                if self.reservation and self.reservation.is_edge_reserved((current[0], current[1]), (nx, ny), nt):
                    continue

                next_state = (nx, ny, nt)
                new_cost = cost_so_far[(current[0], current[1], t)] + 1
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.heuristic((nx, ny), goal)
                    heapq.heappush(frontier, (priority, nt, (nx, ny)))
                    came_from[next_state] = (current[0], current[1], t)

        return None

    def _reconstruct_path(self, came_from, end_state) -> List[Tuple[int, int]]:
        path = []
        current = end_state
        while current is not None:
            x, y, _ = current
            path.append((x, y))
            current = came_from.get(current)
        path.reverse()
        return path


def plan_multi_agent_paths(
    grid: GridMap,
    starts: Dict[str, Tuple[int, int]],
    goals: Dict[str, Tuple[int, int]],
) -> Dict[str, List[Tuple[int, int]]]:
    """Prioritized planning for multiple agents."""
    reservation = ReservationTable()
    paths = {}

    for agent_id in sorted(starts.keys()):
        planner = AStarPlanner(grid, reservation)
        path = planner.plan(starts[agent_id], goals[agent_id])
        if path is None:
            raise RuntimeError(f"No path for agent {agent_id}")
        reservation.reserve(path)
        paths[agent_id] = path

    return paths
