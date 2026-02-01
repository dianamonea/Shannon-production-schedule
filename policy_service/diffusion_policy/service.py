"""
Diffusion Policy 推理服务
实现 PolicyBackendInterface 接口
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

from .model import DiffusionPolicy, create_diffusion_policy
from ..common.schemas_v2 import (
    UnifiedPolicyRequest,
    UnifiedPolicyResponse,
    EnhancedRobotAction,
    ActionCandidate,
    PolicyBackendInterface,
    StateObservation,
)
from ..common.vectorizer import StateVectorizer


class DiffusionPolicyBackend(PolicyBackendInterface):
    """
    Diffusion Policy 后端实现
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        version: str = "v1.0",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.version = version
        
        self.model: Optional[DiffusionPolicy] = None
        self.config: Optional[Dict] = None
        self.vectorizer: Optional[StateVectorizer] = None
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载模型检查点"""
        if not self.checkpoint_path.exists():
            print(f"Warning: Checkpoint {self.checkpoint_path} not found")
            # 创建 dummy 模型用于测试
            self.config = {
                'model': {
                    'max_robots': 10,
                    'max_jobs': 50,
                    'max_stations': 20,
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'num_heads': 4,
                    'dropout': 0.1,
                    'num_diffusion_steps': 10,
                }
            }
            self.model = create_diffusion_policy(self.config['model']).to(self.device)
            self.model.eval()
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.config = checkpoint['config']
            self.model = create_diffusion_policy(self.config['model']).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Loaded diffusion checkpoint from {self.checkpoint_path}")
        
        # 创建 vectorizer
        self.vectorizer = StateVectorizer(
            max_robots=self.config['model']['max_robots'],
            max_jobs=self.config['model']['max_jobs'],
            max_stations=self.config['model']['max_stations'],
        )
    
    def act(self, request: UnifiedPolicyRequest) -> UnifiedPolicyResponse:
        """
        推理接口
        
        Diffusion 只需要最后一个时间步的状态
        """
        start_time = time.time()
        
        # 取最后一个观测
        if len(request.trajectory) == 0:
            raise ValueError("Empty trajectory")
        
        current_obs = request.trajectory[-1]
        
        # Vectorize state
        robot_feats, job_feats, station_feats = self._vectorize_state(current_obs)
        
        # 转换为 tensor 并添加 batch 维度
        robot_feats = torch.tensor(robot_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        job_feats = torch.tensor(job_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        station_feats = torch.tensor(station_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 采样多个候选
        with torch.no_grad():
            actions_tensor, scores_tensor = self.model.sample(
                robot_feats=robot_feats,
                job_feats=job_feats,
                station_feats=station_feats,
                num_samples=request.num_candidates,
                temperature=request.temperature,
                seed=request.seed,
            )
        
        # [1, num_samples, num_robots] -> [num_samples, num_robots]
        actions_tensor = actions_tensor.squeeze(0).cpu().numpy()
        scores_tensor = scores_tensor.squeeze(0).cpu().numpy()
        
        # 排序候选（按 score 升序）
        sorted_indices = np.argsort(scores_tensor)
        actions_tensor = actions_tensor[sorted_indices]
        scores_tensor = scores_tensor[sorted_indices]
        
        # 转换为 EnhancedRobotAction
        candidates = []
        for rank, (action_tokens, score) in enumerate(zip(actions_tensor, scores_tensor)):
            robot_actions = self._tokens_to_actions(action_tokens, current_obs.robots)
            candidates.append(ActionCandidate(
                actions=robot_actions,
                score=float(score),
                rank=rank + 1,
            ))
        
        # 主推荐动作是排名第一的
        best_actions = candidates[0].actions if candidates else []
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # 构建响应
        response = UnifiedPolicyResponse(
            actions=best_actions,
            meta={
                'backend': 'diffusion',
                'version': self.version,
                'model_id': self.checkpoint_path.stem,
                'inference_time_ms': inference_time_ms,
                'num_candidates': len(candidates),
                'num_diffusion_steps': self.model.num_diffusion_steps,
            },
            candidates=candidates if request.num_candidates > 1 else None,
        )
        
        return response
    
    def _vectorize_state(self, obs: StateObservation) -> tuple:
        """
        将 StateObservation 转换为 feature vectors
        """
        # 转换为 vectorizer 期望的格式
        state_dict = {
            't': obs.t,
            'robots': [r.dict() for r in obs.robots],
            'jobs': [j.dict() for j in obs.jobs],
            'stations': [s.dict() for s in obs.stations],
            'global_time': obs.global_time,
        }
        
        return self.vectorizer.vectorize_state(state_dict)
    
    def _tokens_to_actions(
        self,
        action_tokens: np.ndarray,
        robots: List,
    ) -> List[EnhancedRobotAction]:
        """
        将 action tokens 转换为 EnhancedRobotAction 列表
        
        Args:
            action_tokens: [num_robots]，每个值是 job_index（0=idle, 1+=job）
            robots: robot states
        
        Returns:
            actions: List[EnhancedRobotAction]
        """
        actions = []
        
        for i, token in enumerate(action_tokens):
            if i >= len(robots):
                break
            
            robot = robots[i]
            token = int(token)
            
            if token == 0:
                # Idle
                action = EnhancedRobotAction(
                    robot_id=robot.robot_id,
                    action_type='idle',
                    assign_job_id=None,
                )
            else:
                # Assign job (token - 1 是 job index)
                job_idx = token - 1
                job_id = f"j{job_idx}"  # 简单格式，实际应从 obs 映射
                
                action = EnhancedRobotAction(
                    robot_id=robot.robot_id,
                    action_type='assign_job',
                    assign_job_id=job_id,
                )
            
            actions.append(action)
        
        return actions
    
    def get_backend_name(self) -> str:
        return "diffusion"
    
    def get_version(self) -> str:
        return self.version


def create_diffusion_backend(
    checkpoint_path: str = "./checkpoints/best_diffusion_model.pt",
    device: str = "cpu",
    version: str = "v1.0",
) -> DiffusionPolicyBackend:
    """工厂函数：创建 Diffusion 后端"""
    return DiffusionPolicyBackend(
        checkpoint_path=checkpoint_path,
        device=device,
        version=version,
    )
