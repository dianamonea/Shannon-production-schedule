"""
Graph Neural Network for Conflict Prediction

Uses GNN to predict:
1. Which agent pairs are likely to conflict
2. Conflict severity (how hard to resolve)
3. Best branching order in CBS search tree

This enables learning-guided search to dramatically reduce CBS iterations.

Architecture: GraphSAGE-based encoder + MLP decoder
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import random
from collections import defaultdict

# Type aliases
Position = Tuple[int, int]
Path = List[Position]


@dataclass
class AgentFeatures:
    """Features extracted for each agent (GNN node features)."""
    position: Tuple[float, float]
    goal: Tuple[float, float]
    velocity: float
    agent_type_onehot: List[float]  # [CNC, AGV, ROBOT]
    path_length: float
    remaining_distance: float
    congestion_level: float  # Local density of agents
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector."""
        return [
            self.position[0], self.position[1],
            self.goal[0], self.goal[1],
            self.velocity,
            *self.agent_type_onehot,
            self.path_length,
            self.remaining_distance,
            self.congestion_level
        ]


@dataclass 
class EdgeFeatures:
    """Features for agent pair (GNN edge features)."""
    distance: float
    relative_velocity: float
    path_intersection_count: int
    time_to_closest_approach: float
    collision_risk: float  # Heuristic risk score
    
    def to_vector(self) -> List[float]:
        return [
            self.distance,
            self.relative_velocity,
            self.path_intersection_count,
            self.time_to_closest_approach,
            self.collision_risk
        ]


class ConflictPredictorNetwork:
    """
    Simplified GNN for conflict prediction.
    
    In production, this would use PyTorch Geometric or similar.
    Here we implement a basic version for demonstration.
    
    Architecture:
    - Input: Agent features (node) + Pair features (edge)
    - 2-layer GraphSAGE aggregation
    - MLP decoder for conflict probability
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize weights (simplified - would use proper initialization)
        self.node_input_dim = 11  # Size of AgentFeatures vector
        self.edge_input_dim = 5   # Size of EdgeFeatures vector
        
        # Weights initialized randomly (would be learned in practice)
        random.seed(42)
        self._init_weights()
        
        # Training statistics (for normalization)
        self.feature_means: Optional[List[float]] = None
        self.feature_stds: Optional[List[float]] = None
    
    def _init_weights(self):
        """Initialize network weights."""
        # Layer 1 weights
        self.W1 = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)] 
                   for _ in range(self.node_input_dim)]
        self.W1_neighbor = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
                           for _ in range(self.node_input_dim)]
        
        # Layer 2 weights
        self.W2 = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
                   for _ in range(self.hidden_dim)]
        
        # Edge MLP weights
        self.W_edge = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
                      for _ in range(self.edge_input_dim + 2 * self.hidden_dim)]
        self.W_out = [random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
    
    def _relu(self, x: float) -> float:
        return max(0, x)
    
    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def _matmul(self, vec: List[float], mat: List[List[float]]) -> List[float]:
        """Vector-matrix multiplication."""
        return [sum(v * m for v, m in zip(vec, col)) 
                for col in zip(*mat)]
    
    def _aggregate_neighbors(self, node_features: List[List[float]], 
                            edges: List[Tuple[int, int]]) -> List[List[float]]:
        """Mean aggregation of neighbor features."""
        n = len(node_features)
        neighbor_sums = [[0.0] * len(node_features[0]) for _ in range(n)]
        neighbor_counts = [0] * n
        
        for i, j in edges:
            for k in range(len(node_features[0])):
                neighbor_sums[i][k] += node_features[j][k]
                neighbor_sums[j][k] += node_features[i][k]
            neighbor_counts[i] += 1
            neighbor_counts[j] += 1
        
        # Mean aggregation
        result = []
        for i in range(n):
            if neighbor_counts[i] > 0:
                result.append([s / neighbor_counts[i] for s in neighbor_sums[i]])
            else:
                result.append([0.0] * len(node_features[0]))
        
        return result
    
    def forward(self, node_features: List[AgentFeatures],
               edge_features: Dict[Tuple[int, int], EdgeFeatures],
               edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Forward pass to predict conflict probabilities.
        
        Args:
            node_features: Features for each agent
            edge_features: Features for each agent pair
            edges: List of (agent_i, agent_j) pairs to predict
        
        Returns:
            Dictionary of (i, j) -> conflict probability
        """
        # Convert to vectors
        X = [nf.to_vector() for nf in node_features]
        n = len(X)
        
        # Layer 1: Node embedding
        H1 = []
        neighbor_agg = self._aggregate_neighbors(X, edges)
        
        for i in range(n):
            self_transform = self._matmul(X[i], self.W1)
            neighbor_transform = self._matmul(neighbor_agg[i], self.W1_neighbor)
            combined = [self._relu(s + n) for s, n in zip(self_transform, neighbor_transform)]
            H1.append(combined)
        
        # Layer 2: Deeper embedding
        neighbor_agg2 = self._aggregate_neighbors(H1, edges)
        H2 = []
        for i in range(n):
            self_transform = self._matmul(H1[i], self.W2)
            combined = [self._relu(s + n) for s, n in zip(self_transform, neighbor_agg2[i])]
            H2.append(combined)
        
        # Edge-level predictions
        predictions = {}
        
        for (i, j) in edge_features.keys():
            # Concatenate: node_i embedding, node_j embedding, edge features
            edge_input = H2[i] + H2[j] + edge_features[(i, j)].to_vector()
            
            # Pad/truncate to match weight dimensions
            while len(edge_input) < len(self.W_edge):
                edge_input.append(0.0)
            edge_input = edge_input[:len(self.W_edge)]
            
            # MLP for edge
            hidden = self._matmul(edge_input, self.W_edge)
            hidden = [self._relu(h) for h in hidden]
            
            # Output
            score = sum(h * w for h, w in zip(hidden, self.W_out))
            predictions[(i, j)] = self._sigmoid(score)
        
        return predictions
    
    def predict_conflict_priority(self, 
                                  node_features: List[AgentFeatures],
                                  edge_features: Dict[Tuple[int, int], EdgeFeatures],
                                  edges: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """
        Predict and rank conflicts by priority (severity).
        
        Returns: List of (agent_i, agent_j, priority_score) sorted by priority
        """
        probs = self.forward(node_features, edge_features, edges)
        
        # Sort by conflict probability (descending)
        ranked = [(i, j, p) for (i, j), p in probs.items()]
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        return ranked


class ConflictFeatureExtractor:
    """Extract features for GNN from MAPF instance."""
    
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def _agent_type_to_onehot(self, agent_type_str: str) -> List[float]:
        """Convert agent type to one-hot encoding."""
        types = ['CNC', 'AGV', 'ROBOT']
        return [1.0 if t == agent_type_str else 0.0 for t in types]
    
    def _compute_congestion(self, pos: Position, 
                           all_positions: List[Position],
                           radius: float = 3.0) -> float:
        """Compute local congestion around position."""
        count = sum(1 for p in all_positions 
                   if math.hypot(p[0] - pos[0], p[1] - pos[1]) <= radius)
        return count / len(all_positions) if all_positions else 0.0
    
    def _path_intersections(self, path1: Path, path2: Path) -> int:
        """Count number of time steps where paths intersect."""
        max_len = max(len(path1), len(path2))
        count = 0
        
        for t in range(max_len):
            p1 = path1[min(t, len(path1)-1)]
            p2 = path2[min(t, len(path2)-1)]
            if p1 == p2:
                count += 1
        
        return count
    
    def _time_to_closest_approach(self, path1: Path, path2: Path) -> float:
        """Find time of closest approach between two paths."""
        min_dist = float('inf')
        min_time = 0
        max_len = max(len(path1), len(path2))
        
        for t in range(max_len):
            p1 = path1[min(t, len(path1)-1)]
            p2 = path2[min(t, len(path2)-1)]
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            
            if dist < min_dist:
                min_dist = dist
                min_time = t
        
        return float(min_time)
    
    def extract_features(self, 
                        agents: Dict[str, dict],
                        positions: Dict[str, Position],
                        goals: Dict[str, Position],
                        paths: Optional[Dict[str, Path]] = None
                        ) -> Tuple[List[AgentFeatures], 
                                  Dict[Tuple[int, int], EdgeFeatures],
                                  List[Tuple[int, int]],
                                  Dict[str, int]]:
        """
        Extract GNN features from MAPF instance.
        
        Args:
            agents: Agent info (type, velocity, etc.)
            positions: Current positions
            goals: Goal positions
            paths: Optional computed paths
        
        Returns:
            node_features, edge_features, edges, agent_id_to_idx mapping
        """
        agent_ids = list(agents.keys())
        agent_id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        
        all_positions = list(positions.values())
        
        # Node features
        node_features = []
        for aid in agent_ids:
            agent = agents[aid]
            pos = positions[aid]
            goal = goals[aid]
            
            remaining_dist = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
            path_len = len(paths.get(aid, [])) if paths else remaining_dist
            
            nf = AgentFeatures(
                position=(float(pos[0]), float(pos[1])),
                goal=(float(goal[0]), float(goal[1])),
                velocity=agent.get('velocity', 1.0),
                agent_type_onehot=self._agent_type_to_onehot(
                    agent.get('type', 'AGV')
                ),
                path_length=float(path_len),
                remaining_distance=remaining_dist,
                congestion_level=self._compute_congestion(pos, all_positions)
            )
            node_features.append(nf)
        
        # Edge features (for all pairs)
        edge_features = {}
        edges = []
        
        for i, aid1 in enumerate(agent_ids):
            for j, aid2 in enumerate(agent_ids[i+1:], i+1):
                pos1 = positions[aid1]
                pos2 = positions[aid2]
                
                distance = math.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1])
                
                v1 = agents[aid1].get('velocity', 1.0)
                v2 = agents[aid2].get('velocity', 1.0)
                rel_vel = abs(v1 - v2)
                
                # Path-based features if available
                if paths and aid1 in paths and aid2 in paths:
                    intersections = self._path_intersections(paths[aid1], paths[aid2])
                    ttca = self._time_to_closest_approach(paths[aid1], paths[aid2])
                else:
                    intersections = 0
                    ttca = distance / max(v1 + v2, 0.1)
                
                # Heuristic collision risk
                collision_risk = max(0, 1.0 - distance / 5.0) * (1 + intersections * 0.1)
                
                ef = EdgeFeatures(
                    distance=distance,
                    relative_velocity=rel_vel,
                    path_intersection_count=intersections,
                    time_to_closest_approach=ttca,
                    collision_risk=collision_risk
                )
                
                edge_features[(i, j)] = ef
                edges.append((i, j))
        
        return node_features, edge_features, edges, agent_id_to_idx


@dataclass
class TrainingExample:
    """Training example for GNN."""
    node_features: List[AgentFeatures]
    edge_features: Dict[Tuple[int, int], EdgeFeatures]
    edges: List[Tuple[int, int]]
    labels: Dict[Tuple[int, int], int]  # 1 = conflict, 0 = no conflict


class ConflictPredictorTrainer:
    """
    Trainer for the conflict prediction GNN.
    
    Uses online learning from CBS search experience.
    """
    
    def __init__(self, network: ConflictPredictorNetwork, 
                 learning_rate: float = 0.01):
        self.network = network
        self.lr = learning_rate
        self.experience_buffer: List[TrainingExample] = []
        self.buffer_size = 1000
    
    def add_experience(self, example: TrainingExample):
        """Add training example from CBS search."""
        self.experience_buffer.append(example)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def train_step(self, batch_size: int = 32) -> float:
        """
        One training step using experience buffer.
        Returns average loss.
        
        Note: This is a simplified training loop. In practice,
        would use PyTorch with proper backpropagation.
        """
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        total_loss = 0.0
        for example in batch:
            predictions = self.network.forward(
                example.node_features,
                example.edge_features,
                example.edges
            )
            
            # Binary cross-entropy loss
            for (i, j), label in example.labels.items():
                if (i, j) in predictions:
                    pred = predictions[(i, j)]
                    # Clamp for numerical stability
                    pred = max(1e-7, min(1 - 1e-7, pred))
                    
                    if label == 1:
                        loss = -math.log(pred)
                    else:
                        loss = -math.log(1 - pred)
                    
                    total_loss += loss
        
        return total_loss / (batch_size * len(batch[0].labels))
    
    def extract_labels_from_conflicts(self, 
                                     conflicts: List[Tuple[str, str]],
                                     agent_id_to_idx: Dict[str, int],
                                     all_edges: List[Tuple[int, int]]
                                     ) -> Dict[Tuple[int, int], int]:
        """Convert detected conflicts to training labels."""
        labels = {e: 0 for e in all_edges}
        
        for aid1, aid2 in conflicts:
            i = agent_id_to_idx.get(aid1)
            j = agent_id_to_idx.get(aid2)
            
            if i is not None and j is not None:
                key = (min(i, j), max(i, j))
                if key in labels:
                    labels[key] = 1
        
        return labels
