"""
FastAPI 推理服务
提供 /policy/act 端点用于推理
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json

from common.schemas import (
    PolicyActRequest, PolicyActResponse, RobotAction, ActionDistribution,
    StepObservation,
)
from common.vectorizer import StateVectorizer, ActionVectorizer
from training.model import DecisionTransformer


class PolicyServiceConfig:
    """推理服务配置"""
    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/best_model.pt",
        device: str = "cpu",
        version: str = "v1.0",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.version = version


class PolicyService:
    """推理服务封装"""
    
    def __init__(self, config: PolicyServiceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载模型和配置
        self.model = None
        self.model_config = None
        self.state_vectorizer = None
        self.action_vectorizer = ActionVectorizer()
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载检查点"""
        if not self.config.checkpoint_path.exists():
            # 使用虚拟模型进行演示
            print(f"Warning: Checkpoint {self.config.checkpoint_path} not found, using dummy model")
            self.model_config = {
                'max_robots': 10,
                'max_jobs': 50,
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1,
            }
            self._create_dummy_model()
        else:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            
            # 提取配置
            config_dict = checkpoint['config']
            self.model_config = config_dict['model']
            
            # 创建模型
            max_stations = self.model_config.get('max_stations', 20)
            state_vec_dim = (
                self.model_config['max_robots'] * self.model_config['hidden_dim'] +
                self.model_config['max_jobs'] * self.model_config['hidden_dim'] +
                max_stations * self.model_config['hidden_dim'] +
                self.model_config['hidden_dim']
            )
            
            self.model = DecisionTransformer(
                state_vec_dim=state_vec_dim,
                hidden_dim=self.model_config['hidden_dim'],
                num_layers=self.model_config['num_layers'],
                num_heads=self.model_config['num_heads'],
                dropout=self.model_config['dropout'],
                max_robots=self.model_config['max_robots'],
                max_actions=self.model_config['max_jobs'] + 1,
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Loaded checkpoint from {self.config.checkpoint_path}")
    
    def _create_dummy_model(self):
        """创建虚拟模型（演示用）"""
        max_stations = self.model_config.get('max_stations', 20)
        state_vec_dim = (
            self.model_config['max_robots'] * self.model_config['hidden_dim'] +
            self.model_config['max_jobs'] * self.model_config['hidden_dim'] +
            max_stations * self.model_config['hidden_dim'] +
            self.model_config['hidden_dim']
        )
        
        self.model = DecisionTransformer(
            state_vec_dim=state_vec_dim,
            hidden_dim=self.model_config['hidden_dim'],
            num_layers=self.model_config['num_layers'],
            num_heads=self.model_config['num_heads'],
            dropout=self.model_config['dropout'],
            max_robots=self.model_config['max_robots'],
            max_actions=self.model_config['max_jobs'] + 1,
        ).to(self.device)
        
        self.model.eval()
        print("Using dummy (random) model for demonstration")
    
    def _lazy_init_vectorizer(self, obs: StepObservation):
        """延迟初始化向量化器"""
        if self.state_vectorizer is None:
            self.state_vectorizer = StateVectorizer(
                max_robots=self.model_config['max_robots'],
                max_jobs=self.model_config['max_jobs'],
                max_stations=self.model_config.get('max_stations', 20),
                embed_dim=self.model_config['hidden_dim'],
            )
    
    def act(self, request: PolicyActRequest) -> PolicyActResponse:
        """
        推理：从观测序列生成动作
        
        Args:
            request: PolicyActRequest
        
        Returns:
            PolicyActResponse
        """
        try:
            # 初始化向量化器
            self._lazy_init_vectorizer(request.trajectory[0])
            
            # 向量化观测序列
            state_vectors = []
            robot_ids = None
            last_obs = request.trajectory[-1]
            
            for obs in request.trajectory:
                vec_state = self.state_vectorizer.vectorize_step(obs)
                
                # 拼接为单个向量
                flat_vec = []
                flat_vec.extend(vec_state.robot_embeddings)
                flat_vec.extend(vec_state.job_embeddings)
                flat_vec.extend(vec_state.station_embeddings)
                flat_vec.append(vec_state.time_embedding)
                
                flat_vec_array = np.concatenate([
                    np.array(v, dtype=np.float32) if isinstance(v, list) else v
                    for v in flat_vec
                ])
                state_vectors.append(flat_vec_array)
                
                # 收集机器人和任务信息
                if robot_ids is None:
                    robot_ids = [r.robot_id for r in obs.robots]
                
            
            state_vectors = np.stack(state_vectors, axis=0)  # [seq_len, state_vec_dim]
            
            # 转换为 torch 张量
            state_tensor = torch.from_numpy(state_vectors).unsqueeze(0).to(self.device)  # [1, seq_len, state_vec_dim]
            
            # Robot mask
            robot_mask = torch.ones(1, self.model_config['max_robots'], device=self.device)
            if robot_ids:
                robot_mask[0, len(robot_ids):] = 0
            
            # 模型推理
            with torch.no_grad():
                seq_mask = torch.ones(1, state_tensor.shape[1], device=self.device)
                logits = self.model(state_tensor, robot_mask, seq_mask)  # [1, max_robots, max_actions]
            
            logits = logits[0].cpu().numpy()  # [max_robots, max_actions]
            
            # 生成动作（固定动作空间 + mask）
            job_id_order = self.action_vectorizer.build_job_id_order(
                last_obs.jobs,
                self.model_config['max_jobs'],
            )
            action_mask = self.action_vectorizer.build_action_mask(job_id_order)
            available_jobs = [jid for jid in job_id_order if jid is not None]
            
            actions = []
            action_distributions = [] if request.return_logits else None
            
            for i, robot_id in enumerate(robot_ids):
                # 获取该机器人的 logits
                robot_logits = logits[i]  # [max_actions]
                
                # mask 无效动作后取 argmax
                masked_logits = np.where(action_mask > 0, robot_logits, -1e8)
                action_idx = int(np.argmax(masked_logits))

                # 创建动作
                if action_idx >= len(job_id_order) or job_id_order[action_idx] is None:
                    action = RobotAction(
                        robot_id=robot_id,
                        action_type="idle",
                    )
                else:
                    action = RobotAction(
                        robot_id=robot_id,
                        action_type="assign_job",
                        assign_job_id=job_id_order[action_idx],
                    )
                
                actions.append(action)
                
                # 可选：记录分布
                if request.return_logits:
                    logits_dict = {
                        job_id_order[j]: float(robot_logits[j])
                        for j in range(len(job_id_order)) if job_id_order[j] is not None
                    }
                    logits_dict["idle"] = float(robot_logits[len(job_id_order)])
                    
                    action_distributions.append(ActionDistribution(
                        robot_id=robot_id,
                        action_type=action.action_type,
                        assign_job_id=action.assign_job_id,
                        logits=logits_dict,
                        confidence=float(1.0 / (1.0 + np.exp(-float(robot_logits[action_idx])))),
                    ))
            
            # 元数据
            meta = {
                'policy_version': self.config.version,
                'model_device': str(self.device),
                'num_robots': len(robot_ids),
                'num_available_jobs': len(available_jobs),
            }
            
            return PolicyActResponse(
                actions=actions,
                action_distributions=action_distributions,
                meta=meta,
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="MADT Policy Service",
    description="Multi-Agent Decision Transformer Policy Service",
    version="v1.0",
)

# 全局服务实例
policy_service: Optional[PolicyService] = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global policy_service
    config = PolicyServiceConfig(
        checkpoint_path="./checkpoints/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        version="v1.0",
    )
    policy_service = PolicyService(config)
    print("Policy service initialized")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "madt_policy_service",
        "version": "v1.0",
    }


@app.post("/policy/act")
async def policy_act(request: PolicyActRequest) -> PolicyActResponse:
    """
    推理端点：从观测序列生成动作
    
    Args:
        request: 包含 K 步观测序列的请求
    
    Returns:
        PolicyActResponse: 所有机器人的动作
    """
    if policy_service is None:
        raise HTTPException(status_code=503, detail="Policy service not initialized")
    
    return policy_service.act(request)


@app.post("/policy/act_batch")
async def policy_act_batch(requests: list[PolicyActRequest]) -> list[PolicyActResponse]:
    """
    批量推理端点
    
    Args:
        requests: 多个推理请求
    
    Returns:
        多个推理响应
    """
    if policy_service is None:
        raise HTTPException(status_code=503, detail="Policy service not initialized")
    
    responses = []
    for request in requests:
        try:
            response = policy_service.act(request)
            responses.append(response)
        except HTTPException as e:
            # 如果某个请求失败，返回错误响应
            responses.append({
                "error": str(e.detail),
                "status_code": e.status_code,
            })
    
    return responses


@app.get("/policy/info")
async def policy_info() -> Dict[str, Any]:
    """获取策略信息"""
    if policy_service is None:
        raise HTTPException(status_code=503, detail="Policy service not initialized")
    
    return {
        "version": policy_service.config.version,
        "device": str(policy_service.device),
        "model_config": policy_service.model_config,
        "checkpoint": str(policy_service.config.checkpoint_path),
    }


# 错误处理
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )
