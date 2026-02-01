"""
Decision Transformer 后端适配器
将现有 DT 实现包装为 PolicyBackendInterface
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time

from .training.model import DecisionTransformer
from .common.schemas_v2 import (
    UnifiedPolicyRequest,
    UnifiedPolicyResponse,
    EnhancedRobotAction,
    PolicyBackendInterface,
)
from .common.vectorizer import StateVectorizer, ActionVectorizer


class DTPolicyBackend(PolicyBackendInterface):
    """
    Decision Transformer 后端实现
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
        
        self.model: Optional[DecisionTransformer] = None
        self.config: Optional[Dict] = None
        self.state_vectorizer: Optional[StateVectorizer] = None
        self.action_vectorizer = ActionVectorizer()
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载模型检查点"""
        if not self.checkpoint_path.exists():
            print(f"Warning: DT checkpoint {self.checkpoint_path} not found")
            # 创建 dummy 配置
            self.config = {
                'model': {
                    'max_robots': 10,
                    'max_jobs': 50,
                    'max_stations': 20,
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'num_heads': 8,
                    'dropout': 0.1,
                }
            }
            self._create_dummy_model()
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.config = checkpoint.get('config', {})
            
            # 创建模型
            model_config = self.config.get('model', {})
            max_stations = model_config.get('max_stations', 20)
            state_vec_dim = (
                model_config['max_robots'] * model_config['hidden_dim'] +
                model_config['max_jobs'] * model_config['hidden_dim'] +
                max_stations * model_config['hidden_dim'] +
                model_config['hidden_dim']
            )
            
            self.model = DecisionTransformer(
                state_vec_dim=state_vec_dim,
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                num_heads=model_config['num_heads'],
                dropout=model_config['dropout'],
                max_robots=model_config['max_robots'],
                max_actions=model_config['max_jobs'] + 1,
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Loaded DT checkpoint from {self.checkpoint_path}")
        
        # 创建 vectorizer
        self.state_vectorizer = StateVectorizer(
            max_robots=self.config['model']['max_robots'],
            max_jobs=self.config['model']['max_jobs'],
            max_stations=self.config['model']['max_stations'],
        )
    
    def _create_dummy_model(self):
        """创建 dummy 模型（用于测试）"""
        model_config = self.config['model']
        max_stations = model_config.get('max_stations', 20)
        state_vec_dim = (
            model_config['max_robots'] * model_config['hidden_dim'] +
            model_config['max_jobs'] * model_config['hidden_dim'] +
            max_stations * model_config['hidden_dim'] +
            model_config['hidden_dim']
        )
        
        self.model = DecisionTransformer(
            state_vec_dim=state_vec_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            max_robots=model_config['max_robots'],
            max_actions=model_config['max_jobs'] + 1,
        ).to(self.device)
        
        self.model.eval()
    
    def act(self, request: UnifiedPolicyRequest) -> UnifiedPolicyResponse:
        """
        推理接口（DT 需要 K 步历史）
        """
        start_time = time.time()
        
        if len(request.trajectory) == 0:
            raise ValueError("Empty trajectory")
        
        # Vectorize trajectory
        state_vecs = []
        for obs in request.trajectory:
            state_dict = {
                't': obs.t,
                'robots': [r.dict() for r in obs.robots],
                'jobs': [j.dict() for j in obs.jobs],
                'stations': [s.dict() for s in obs.stations],
                'global_time': obs.global_time,
            }
            
            robot_feats, job_feats, station_feats = self.state_vectorizer.vectorize_state(state_dict)
            
            # 拼接为单个向量（与 DT 训练时一致）
            state_vec = np.concatenate([
                robot_feats.flatten(),
                job_feats.flatten(),
                station_feats.flatten(),
                np.array([obs.global_time]),
            ])
            state_vecs.append(state_vec)
        
        # 转换为 tensor [1, K, state_dim]
        state_tensor = torch.tensor(state_vecs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            action_logits = self.model(state_tensor)  # [1, K, num_robots, num_actions]
        
        # 取最后一个时间步
        last_logits = action_logits[0, -1]  # [num_robots, num_actions]
        
        # 贪心解码
        action_indices = torch.argmax(last_logits, dim=-1).cpu().numpy()  # [num_robots]
        
        # 转换为 EnhancedRobotAction
        current_obs = request.trajectory[-1]
        actions = []
        
        for i, action_idx in enumerate(action_indices):
            if i >= len(current_obs.robots):
                break
            
            robot = current_obs.robots[i]
            
            if action_idx == 0:
                # Idle
                action = EnhancedRobotAction(
                    robot_id=robot.robot_id,
                    action_type='idle',
                    assign_job_id=None,
                )
            else:
                # Assign job
                job_idx = action_idx - 1
                job_id = f"j{job_idx}"
                
                action = EnhancedRobotAction(
                    robot_id=robot.robot_id,
                    action_type='assign_job',
                    assign_job_id=job_id,
                )
            
            actions.append(action)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # 构建响应
        logits_dict = None
        if request.return_logits:
            logits_dict = {
                f"robot_{i}": last_logits[i].cpu().tolist()
                for i in range(len(current_obs.robots))
            }
        
        response = UnifiedPolicyResponse(
            actions=actions,
            meta={
                'backend': 'dt',
                'version': self.version,
                'model_id': self.checkpoint_path.stem,
                'inference_time_ms': inference_time_ms,
                'trajectory_length': len(request.trajectory),
            },
            logits=logits_dict,
        )
        
        return response
    
    def get_backend_name(self) -> str:
        return "dt"
    
    def get_version(self) -> str:
        return self.version


def create_dt_backend(
    checkpoint_path: str = "./checkpoints/best_model.pt",
    device: str = "cpu",
    version: str = "v1.0",
) -> DTPolicyBackend:
    """工厂函数：创建 DT 后端"""
    return DTPolicyBackend(
        checkpoint_path=checkpoint_path,
        device=device,
        version=version,
    )
