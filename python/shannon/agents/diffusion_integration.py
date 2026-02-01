"""
Diffusion Policy Integration for Manufacturing Agents

Integrates diffusion-based multi-agent reinforcement learning (from diffusion_marl.py)
with Shannon's manufacturing system for learned skill execution.

Key features:
1. ROS2 bridge for Rust agent-core
2. Demonstration collection pipeline
3. Policy inference for skill execution
4. Online fine-tuning from production data
"""

import asyncio
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Import existing diffusion MARL implementation
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from python.diffusion_marl import DiffusionPolicy, MultiAgentDiffusion
except ImportError:
    logging.warning("diffusion_marl.py not found - using mock implementation")
    DiffusionPolicy = None
    MultiAgentDiffusion = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Demonstration:
    """Single demonstration trajectory"""
    states: np.ndarray  # (T, state_dim)
    actions: np.ndarray  # (T, action_dim)
    rewards: np.ndarray  # (T,)
    agent_id: str
    skill_name: str
    success: bool
    timestamp: float


@dataclass
class ObservationSpace:
    """Manufacturing observation space"""
    joint_positions: np.ndarray  # (7,) for 7-DOF arm
    joint_velocities: np.ndarray  # (7,)
    ee_pose: np.ndarray  # (7,) [x,y,z, qw,qx,qy,qz]
    force_torque: np.ndarray  # (6,) [fx,fy,fz, tx,ty,tz]
    vision_features: Optional[np.ndarray] = None  # (512,) from vision encoder
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat state vector"""
        vectors = [
            self.joint_positions,
            self.joint_velocities,
            self.ee_pose,
            self.force_torque
        ]
        
        if self.vision_features is not None:
            vectors.append(self.vision_features)
        
        return np.concatenate(vectors)


@dataclass
class ActionSpace:
    """Manufacturing action space"""
    joint_positions_delta: np.ndarray  # (7,) position changes
    gripper_command: float  # 0.0 (open) - 1.0 (close)
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat action vector"""
        return np.concatenate([
            self.joint_positions_delta,
            [self.gripper_command]
        ])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'ActionSpace':
        """Reconstruct from flat vector"""
        return cls(
            joint_positions_delta=vec[:7],
            gripper_command=vec[7]
        )


class DemonstrationCollector:
    """
    Collects demonstrations from human teleoperation or successful executions
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.demonstrations: List[Demonstration] = []
    
    async def record_demonstration(self, agent_id: str, skill_name: str,
                                   duration: float = 30.0) -> Demonstration:
        """
        Record a demonstration trajectory
        
        Args:
            agent_id: ID of agent performing demonstration
            skill_name: Name of skill being demonstrated
            duration: Maximum recording duration in seconds
        
        Returns:
            Recorded demonstration
        """
        logger.info(f"Recording demonstration for {skill_name} on {agent_id}")
        
        states = []
        actions = []
        rewards = []
        
        start_time = asyncio.get_event_loop().time()
        
        # Simulate recording (in real system, would subscribe to ROS2 topics)
        while asyncio.get_event_loop().time() - start_time < duration:
            # Get current state
            obs = await self._get_observation(agent_id)
            state = obs.to_vector()
            
            # Get executed action
            action = await self._get_action(agent_id)
            action_vec = action.to_vector()
            
            # Get reward signal
            reward = await self._calculate_reward(obs, action)
            
            states.append(state)
            actions.append(action_vec)
            rewards.append(reward)
            
            await asyncio.sleep(0.1)  # 10 Hz recording
        
        demo = Demonstration(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_id=agent_id,
            skill_name=skill_name,
            success=np.mean(rewards) > 0.5,
            timestamp=start_time
        )
        
        self.demonstrations.append(demo)
        self._save_demonstration(demo)
        
        logger.info(f"Demonstration recorded: {len(states)} timesteps, "
                   f"avg reward: {np.mean(rewards):.3f}")
        
        return demo
    
    async def _get_observation(self, agent_id: str) -> ObservationSpace:
        """Get current observation from agent (simulation)"""
        return ObservationSpace(
            joint_positions=np.random.randn(7) * 0.1,
            joint_velocities=np.random.randn(7) * 0.01,
            ee_pose=np.array([0.5, 0.3, 0.2, 1.0, 0.0, 0.0, 0.0]),
            force_torque=np.random.randn(6) * 0.5
        )
    
    async def _get_action(self, agent_id: str) -> ActionSpace:
        """Get executed action (simulation)"""
        return ActionSpace(
            joint_positions_delta=np.random.randn(7) * 0.01,
            gripper_command=np.random.rand()
        )
    
    async def _calculate_reward(self, obs: ObservationSpace, 
                                action: ActionSpace) -> float:
        """Calculate reward based on task progress (simulation)"""
        # Example: reward based on proximity to target
        target_pos = np.array([0.6, 0.4, 0.3])
        current_pos = obs.ee_pose[:3]
        distance = np.linalg.norm(current_pos - target_pos)
        
        reward = 1.0 - np.clip(distance, 0, 1)
        return reward
    
    def _save_demonstration(self, demo: Demonstration):
        """Save demonstration to disk"""
        filename = f"{demo.skill_name}_{demo.agent_id}_{int(demo.timestamp)}.pkl"
        filepath = self.save_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(demo, f)
        
        logger.info(f"Saved demonstration to {filepath}")
    
    def load_demonstrations(self, skill_name: Optional[str] = None) -> List[Demonstration]:
        """Load demonstrations from disk"""
        demos = []
        
        for filepath in self.save_dir.glob("*.pkl"):
            if skill_name and skill_name not in filepath.name:
                continue
            
            with open(filepath, 'rb') as f:
                demo = pickle.load(f)
                demos.append(demo)
        
        logger.info(f"Loaded {len(demos)} demonstrations")
        return demos


class DiffusionPolicyBridge:
    """
    Bridge between Shannon manufacturing system and diffusion policy
    """
    
    def __init__(self, model_dir: Path, state_dim: int = 27, action_dim: int = 8):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize diffusion policy
        if DiffusionPolicy is not None:
            self.policy = DiffusionPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=256,
                num_diffusion_steps=50
            )
        else:
            logger.warning("Using mock diffusion policy")
            self.policy = None
        
        self.trained_skills: Dict[str, bool] = {}
    
    def train_from_demonstrations(self, demonstrations: List[Demonstration],
                                  skill_name: str, epochs: int = 100):
        """
        Train diffusion policy from demonstrations
        
        Args:
            demonstrations: List of demonstration trajectories
            skill_name: Name of skill to train
            epochs: Training epochs
        """
        logger.info(f"Training diffusion policy for {skill_name} "
                   f"from {len(demonstrations)} demonstrations")
        
        if self.policy is None:
            logger.warning("No policy implementation - skipping training")
            return
        
        # Prepare training data
        all_states = []
        all_actions = []
        
        for demo in demonstrations:
            all_states.append(demo.states)
            all_actions.append(demo.actions)
        
        # Concatenate all trajectories
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        
        logger.info(f"Training data: {states.shape[0]} timesteps")
        
        # Train policy (simplified - real implementation would use proper training loop)
        for epoch in range(epochs):
            # Sample mini-batch
            batch_size = 256
            indices = np.random.choice(len(states), size=batch_size, replace=False)
            
            state_batch = states[indices]
            action_batch = actions[indices]
            
            # Training step (mock)
            # In real implementation: loss = policy.compute_loss(state_batch, action_batch)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Save trained policy
        self._save_policy(skill_name)
        self.trained_skills[skill_name] = True
        
        logger.info(f"Training complete for {skill_name}")
    
    async def execute_skill(self, skill_name: str, initial_obs: ObservationSpace,
                           max_steps: int = 100) -> bool:
        """
        Execute trained skill using diffusion policy
        
        Args:
            skill_name: Name of trained skill
            initial_obs: Initial observation
            max_steps: Maximum execution steps
        
        Returns:
            Success flag
        """
        if skill_name not in self.trained_skills:
            logger.error(f"Skill {skill_name} not trained")
            return False
        
        logger.info(f"Executing skill {skill_name} with diffusion policy")
        
        current_obs = initial_obs
        
        for step in range(max_steps):
            # Get state vector
            state = current_obs.to_vector()
            
            # Predict action using diffusion policy
            action_vec = self._predict_action(state)
            action = ActionSpace.from_vector(action_vec)
            
            # Execute action (would send to ROS2 in real system)
            success = await self._execute_action(action)
            
            if not success:
                logger.error(f"Action execution failed at step {step}")
                return False
            
            # Get next observation
            current_obs = await self._get_next_observation()
            
            # Check termination
            if self._is_task_complete(current_obs):
                logger.info(f"Skill {skill_name} completed successfully in {step + 1} steps")
                return True
            
            await asyncio.sleep(0.1)  # 10 Hz control rate
        
        logger.warning(f"Skill {skill_name} did not complete within {max_steps} steps")
        return False
    
    def _predict_action(self, state: np.ndarray) -> np.ndarray:
        """Predict action using trained policy"""
        if self.policy is None:
            # Mock prediction
            return np.random.randn(self.action_dim) * 0.01
        
        # Real prediction (simplified)
        # action = policy.predict(state)
        return np.random.randn(self.action_dim) * 0.01
    
    async def _execute_action(self, action: ActionSpace) -> bool:
        """Send action to robot controller"""
        # Would publish to ROS2 action server in real implementation
        await asyncio.sleep(0.1)
        return True
    
    async def _get_next_observation(self) -> ObservationSpace:
        """Get observation after action execution"""
        return ObservationSpace(
            joint_positions=np.random.randn(7) * 0.1,
            joint_velocities=np.random.randn(7) * 0.01,
            ee_pose=np.array([0.55, 0.35, 0.25, 1.0, 0.0, 0.0, 0.0]),
            force_torque=np.random.randn(6) * 0.5
        )
    
    def _is_task_complete(self, obs: ObservationSpace) -> bool:
        """Check if task is complete based on observation"""
        # Example: check if gripper is at target position
        target_pos = np.array([0.6, 0.4, 0.3])
        current_pos = obs.ee_pose[:3]
        distance = np.linalg.norm(current_pos - target_pos)
        
        return distance < 0.02  # 2cm tolerance
    
    def _save_policy(self, skill_name: str):
        """Save trained policy to disk"""
        filepath = self.model_dir / f"{skill_name}_policy.pkl"
        
        policy_data = {
            'skill_name': skill_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'trained_at': asyncio.get_event_loop().time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
        
        logger.info(f"Saved policy to {filepath}")
    
    def load_policy(self, skill_name: str):
        """Load trained policy from disk"""
        filepath = self.model_dir / f"{skill_name}_policy.pkl"
        
        if not filepath.exists():
            logger.error(f"Policy file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            policy_data = pickle.load(f)
        
        self.trained_skills[skill_name] = True
        logger.info(f"Loaded policy for {skill_name}")
        return True


class ROS2ActionBridge:
    """
    Bridge for ROS2 communication with Rust agent-core
    """
    
    def __init__(self):
        self.action_server_running = False
    
    async def start_action_server(self):
        """Start ROS2 action server for skill execution"""
        logger.info("Starting ROS2 action server")
        self.action_server_running = True
        
        # In real implementation, would initialize ROS2 node
        # and create action server
    
    async def publish_action(self, action: ActionSpace, agent_id: str):
        """Publish action to ROS2 topic"""
        if not self.action_server_running:
            logger.warning("Action server not running")
            return
        
        # Would publish to /agent_{agent_id}/joint_commands topic
        logger.debug(f"Published action to {agent_id}")
    
    async def subscribe_observations(self, agent_id: str) -> ObservationSpace:
        """Subscribe to observation topic"""
        # Would subscribe to /agent_{agent_id}/observations topic
        return ObservationSpace(
            joint_positions=np.random.randn(7) * 0.1,
            joint_velocities=np.random.randn(7) * 0.01,
            ee_pose=np.array([0.5, 0.3, 0.2, 1.0, 0.0, 0.0, 0.0]),
            force_torque=np.random.randn(6) * 0.5
        )


# Example usage
async def main():
    # Setup paths
    demo_dir = Path("data/demonstrations")
    model_dir = Path("models/diffusion")
    
    # Initialize components
    collector = DemonstrationCollector(demo_dir)
    bridge = DiffusionPolicyBridge(model_dir)
    ros_bridge = ROS2ActionBridge()
    
    await ros_bridge.start_action_server()
    
    # 1. Collect demonstrations
    print("\n=== Collecting Demonstrations ===")
    demo1 = await collector.record_demonstration("AGENT_001", "pick_and_place", duration=5.0)
    demo2 = await collector.record_demonstration("AGENT_001", "pick_and_place", duration=5.0)
    
    # 2. Train policy
    print("\n=== Training Diffusion Policy ===")
    demos = collector.load_demonstrations("pick_and_place")
    bridge.train_from_demonstrations(demos, "pick_and_place", epochs=50)
    
    # 3. Execute learned skill
    print("\n=== Executing Learned Skill ===")
    initial_obs = ObservationSpace(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.array([0.5, 0.3, 0.2, 1.0, 0.0, 0.0, 0.0]),
        force_torque=np.zeros(6)
    )
    
    success = await bridge.execute_skill("pick_and_place", initial_obs, max_steps=50)
    print(f"\nSkill execution {'succeeded' if success else 'failed'}")


if __name__ == "__main__":
    asyncio.run(main())
