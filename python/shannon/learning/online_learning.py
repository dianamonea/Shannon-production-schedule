"""
Online Learning System for Manufacturing Agents

Implements continuous learning from production data to improve:
1. Scheduling decisions (reinforcement learning)
2. Quality predictions (supervised learning)
3. Anomaly detection (unsupervised learning)
4. Agent coordination strategies (multi-agent learning)

Features:
- Experience replay buffer
- Incremental model updates
- A/B testing for policy rollout
- Performance monitoring
"""

import logging
import numpy as np
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning algorithm types"""
    REINFORCEMENT = "REINFORCEMENT"  # For decision-making
    SUPERVISED = "SUPERVISED"  # For prediction
    UNSUPERVISED = "UNSUPERVISED"  # For anomaly detection


@dataclass
class Experience:
    """Single experience tuple for RL"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Metrics for tracking learning progress"""
    episode: int
    avg_reward: float
    loss: float
    epsilon: float  # For epsilon-greedy exploration
    model_version: int
    timestamp: datetime = field(default_factory=datetime.now)


class ExperienceReplayBuffer:
    """
    Replay buffer for storing and sampling experiences
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Prioritized sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: Path):
        """Save buffer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({'buffer': list(self.buffer), 'priorities': list(self.priorities)}, f)
        logger.info(f"Saved replay buffer with {len(self.buffer)} experiences to {filepath}")
    
    def load(self, filepath: Path):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data['buffer'], maxlen=self.capacity)
            self.priorities = deque(data['priorities'], maxlen=self.capacity)
        logger.info(f"Loaded replay buffer with {len(self.buffer)} experiences")


class QNetwork:
    """
    Q-Network for approximating action-value function
    (Simplified implementation - in production, use PyTorch/TensorFlow)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize simple linear approximation (mock)
        self.weights = {
            'W1': np.random.randn(state_dim, hidden_dim) * 0.01,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, action_dim) * 0.01,
            'b2': np.zeros(action_dim)
        }
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions"""
        # Forward pass
        h1 = np.maximum(0, state @ self.weights['W1'] + self.weights['b1'])  # ReLU
        q_values = h1 @ self.weights['W2'] + self.weights['b2']
        return q_values
    
    def update(self, states: np.ndarray, targets: np.ndarray, learning_rate: float = 0.001):
        """Update network weights (simplified gradient descent)"""
        # In production, would use proper backpropagation
        batch_size = states.shape[0]
        
        # Forward pass
        h1 = np.maximum(0, states @ self.weights['W1'] + self.weights['b1'])
        predictions = h1 @ self.weights['W2'] + self.weights['b2']
        
        # Compute loss
        loss = np.mean((predictions - targets) ** 2)
        
        # Simplified weight update (mock)
        grad_scale = learning_rate * (predictions - targets).mean()
        self.weights['W2'] -= grad_scale * 0.01
        
        return loss


class OnlineLearner:
    """
    Online learning system for manufacturing optimization
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 model_dir: Path, mode: LearningMode = LearningMode.REINFORCEMENT):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        
        # Initialize components
        self.replay_buffer = ExperienceReplayBuffer(capacity=100000)
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim=256)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim=256)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 100
        
        # Tracking
        self.training_step = 0
        self.episode = 0
        self.model_version = 1
        self.metrics_history: List[TrainingMetrics] = []
        
        # A/B testing
        self.baseline_performance = 0.0
        self.current_performance = 0.0
        self.ab_test_episodes = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            explore: Whether to use exploration
        
        Returns:
            Selected action index
        """
        if explore and np.random.rand() < self.epsilon:
            # Random exploration
            return np.random.randint(self.action_dim)
        else:
            # Greedy exploitation
            q_values = self.q_network.predict(state)
            return int(np.argmax(q_values))
    
    def add_experience(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool, metadata: Dict = None):
        """Add experience to replay buffer"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            metadata=metadata or {}
        )
        
        # Calculate priority based on TD error (simplified)
        q_current = self.q_network.predict(state)[action]
        q_next = np.max(self.q_network.predict(next_state)) if not done else 0
        td_error = abs(reward + self.gamma * q_next - q_current)
        priority = td_error + 1e-6
        
        self.replay_buffer.add(experience, priority)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step
        
        Returns:
            Training loss, or None if not enough data
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample mini-batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Compute targets using Double DQN
        current_q = self.q_network.predict(states)
        next_q_online = self.q_network.predict(next_states)
        next_q_target = self.target_network.predict(next_states)
        
        targets = current_q.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # Double DQN: use online network to select action, target network to evaluate
                best_action = np.argmax(next_q_online[i])
                targets[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_action]
        
        # Update Q-network
        loss = self.q_network.update(states, targets, self.learning_rate)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def _update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.weights = {
            k: v.copy() for k, v in self.q_network.weights.items()
        }
        logger.info("Updated target network")
    
    def train_episode(self, episode_experiences: List[Experience]) -> TrainingMetrics:
        """
        Train on a full episode of experiences
        
        Args:
            episode_experiences: List of experiences from one episode
        
        Returns:
            Training metrics
        """
        # Add all experiences to buffer
        for exp in episode_experiences:
            self.add_experience(
                exp.state, exp.action, exp.reward, exp.next_state, exp.done, exp.metadata
            )
        
        # Perform multiple training steps
        losses = []
        for _ in range(len(episode_experiences)):
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
        
        # Calculate metrics
        avg_reward = np.mean([e.reward for e in episode_experiences])
        avg_loss = np.mean(losses) if losses else 0.0
        
        self.episode += 1
        
        metrics = TrainingMetrics(
            episode=self.episode,
            avg_reward=avg_reward,
            loss=avg_loss,
            epsilon=self.epsilon,
            model_version=self.model_version
        )
        
        self.metrics_history.append(metrics)
        
        logger.info(f"Episode {self.episode}: avg_reward={avg_reward:.3f}, "
                   f"loss={avg_loss:.4f}, epsilon={self.epsilon:.3f}")
        
        return metrics
    
    def save_model(self, version: Optional[int] = None):
        """Save model to disk"""
        if version is None:
            version = self.model_version
        
        model_path = self.model_dir / f"model_v{version}.pkl"
        
        model_data = {
            'q_network_weights': self.q_network.weights,
            'target_network_weights': self.target_network.weights,
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode': self.episode,
            'model_version': version
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved model version {version} to {model_path}")
        
        # Save replay buffer
        buffer_path = self.model_dir / f"replay_buffer_v{version}.pkl"
        self.replay_buffer.save(buffer_path)
    
    def load_model(self, version: int):
        """Load model from disk"""
        model_path = self.model_dir / f"model_v{version}.pkl"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_network.weights = model_data['q_network_weights']
        self.target_network.weights = model_data['target_network_weights']
        self.epsilon = model_data['epsilon']
        self.training_step = model_data['training_step']
        self.episode = model_data['episode']
        self.model_version = model_data['model_version']
        
        logger.info(f"Loaded model version {version}")
        
        # Load replay buffer
        buffer_path = self.model_dir / f"replay_buffer_v{version}.pkl"
        if buffer_path.exists():
            self.replay_buffer.load(buffer_path)
        
        return True
    
    def ab_test_new_model(self, num_episodes: int = 100) -> bool:
        """
        Run A/B test to compare new model against baseline
        
        Args:
            num_episodes: Number of episodes for testing
        
        Returns:
            True if new model is better
        """
        logger.info(f"Starting A/B test for {num_episodes} episodes")
        
        # Save current model as candidate
        self.save_model(version=self.model_version + 1)
        
        # Simulate A/B test (in production, would run actual episodes)
        # For now, use metrics history
        if len(self.metrics_history) < num_episodes:
            logger.warning("Not enough metrics history for A/B test")
            return False
        
        recent_rewards = [m.avg_reward for m in self.metrics_history[-num_episodes:]]
        self.current_performance = np.mean(recent_rewards)
        
        # Compare to baseline
        if self.baseline_performance == 0.0:
            # First model - set as baseline
            self.baseline_performance = self.current_performance
            improvement = 0.0
        else:
            improvement = (self.current_performance - self.baseline_performance) / abs(self.baseline_performance)
        
        logger.info(f"A/B Test Results: baseline={self.baseline_performance:.3f}, "
                   f"current={self.current_performance:.3f}, "
                   f"improvement={improvement*100:.1f}%")
        
        # Require 5% improvement to promote model
        if improvement > 0.05:
            logger.info("New model promoted to production")
            self.model_version += 1
            self.baseline_performance = self.current_performance
            return True
        else:
            logger.info("New model rejected - keeping baseline")
            return False
    
    def get_performance_summary(self) -> Dict:
        """Get summary of learning performance"""
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-100:]
        
        return {
            'total_episodes': self.episode,
            'model_version': self.model_version,
            'recent_avg_reward': np.mean([m.avg_reward for m in recent]),
            'recent_avg_loss': np.mean([m.loss for m in recent]),
            'current_epsilon': self.epsilon,
            'replay_buffer_size': len(self.replay_buffer),
            'baseline_performance': self.baseline_performance
        }

    def export_weight_suggestions(self) -> Dict[str, float]:
        """Export suggested weights for scheduling (heuristic)."""
        if not self.metrics_history:
            return {
                'weight_duration': 0.3,
                'weight_cost': 0.25,
                'weight_quality': 0.25,
                'weight_load': 0.15,
                'weight_tool_health': 0.1,
            }

        recent = self.metrics_history[-20:]
        avg_reward = np.mean([m.avg_reward for m in recent])
        avg_loss = np.mean([m.loss for m in recent])

        # Simple heuristic: if loss is high, emphasize quality; if reward is low, emphasize duration
        weight_quality = 0.25 + min(max(avg_loss, 0.0), 1.0) * 0.1
        weight_duration = 0.3 + (0.2 if avg_reward < 0 else 0.0)
        weight_cost = 0.25
        weight_load = 0.15
        weight_tool_health = 0.1

        total = weight_quality + weight_duration + weight_cost + weight_load + weight_tool_health
        return {
            'weight_duration': weight_duration / total,
            'weight_cost': weight_cost / total,
            'weight_quality': weight_quality / total,
            'weight_load': weight_load / total,
            'weight_tool_health': weight_tool_health / total,
        }


class AnomalyDetector:
    """
    Unsupervised learning for anomaly detection in production
    """
    
    def __init__(self, feature_dim: int, contamination: float = 0.05):
        self.feature_dim = feature_dim
        self.contamination = contamination
        
        # Simple statistics-based detector
        self.mean = np.zeros(feature_dim)
        self.std = np.ones(feature_dim)
        self.threshold = 3.0  # 3-sigma rule
        
        self.samples_seen = 0
        self.anomalies_detected = 0
    
    def fit(self, data: np.ndarray):
        """Fit detector on normal data"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.samples_seen = data.shape[0]
        
        logger.info(f"Fitted anomaly detector on {self.samples_seen} samples")
    
    def predict(self, sample: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if sample is anomalous
        
        Returns:
            (is_anomaly, anomaly_score)
        """
        # Calculate z-score
        z_scores = np.abs((sample - self.mean) / (self.std + 1e-6))
        anomaly_score = np.max(z_scores)
        
        is_anomaly = anomaly_score > self.threshold
        
        if is_anomaly:
            self.anomalies_detected += 1
            logger.warning(f"Anomaly detected! Score: {anomaly_score:.3f}")
        
        return is_anomaly, float(anomaly_score)
    
    def update_online(self, sample: np.ndarray):
        """Update statistics incrementally"""
        self.samples_seen += 1
        
        # Incremental mean update
        delta = sample - self.mean
        self.mean += delta / self.samples_seen
        
        # Incremental variance update (simplified)
        self.std = np.sqrt((self.std ** 2 * (self.samples_seen - 1) + delta ** 2) / self.samples_seen)


# Example usage
def main():
    # Initialize learner
    model_dir = Path("models/online_learning")
    learner = OnlineLearner(
        state_dim=10,  # Example: agent utilization, WIP level, etc.
        action_dim=5,  # Example: 5 scheduling strategies
        model_dir=model_dir
    )
    
    # Simulate learning episodes
    print("\n=== Online Learning Simulation ===")
    
    for episode in range(100):
        # Simulate episode
        episode_experiences = []
        state = np.random.randn(10)
        
        for step in range(50):
            action = learner.select_action(state)
            
            # Simulate environment response
            reward = np.random.randn() + (100 - episode) * 0.01  # Improving rewards
            next_state = np.random.randn(10)
            done = step == 49
            
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            episode_experiences.append(experience)
            state = next_state
        
        # Train on episode
        metrics = learner.train_episode(episode_experiences)
        
        # Save model periodically
        if (episode + 1) % 20 == 0:
            learner.save_model()
        
        # Run A/B test
        if (episode + 1) % 50 == 0:
            learner.ab_test_new_model(num_episodes=50)
    
    # Print summary
    print("\n=== Performance Summary ===")
    summary = learner.get_performance_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Anomaly detection example
    print("\n=== Anomaly Detection ===")
    detector = AnomalyDetector(feature_dim=5)
    
    # Fit on normal data
    normal_data = np.random.randn(1000, 5)
    detector.fit(normal_data)
    
    # Test on normal and anomalous samples
    normal_sample = np.random.randn(5)
    anomaly_sample = np.random.randn(5) * 5  # Outlier
    
    is_anomaly1, score1 = detector.predict(normal_sample)
    is_anomaly2, score2 = detector.predict(anomaly_sample)
    
    print(f"Normal sample: anomaly={is_anomaly1}, score={score1:.3f}")
    print(f"Anomaly sample: anomaly={is_anomaly2}, score={score2:.3f}")


if __name__ == "__main__":
    main()
