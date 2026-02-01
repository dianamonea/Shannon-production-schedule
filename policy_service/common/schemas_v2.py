"""
统一策略接口 Schema (v2)
支持 DT 和 Diffusion 两种后端，并可扩展
"""

from typing import List, Dict, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum

from .schemas import (
    RobotState, JobSpec, StationState, LaneInfo,
    RobotStatus, JobStatus, StationType, RobotAction
)


# ============================================================
# 策略后端类型
# ============================================================

class PolicyBackend(str, Enum):
    """策略后端类型"""
    DT = "dt"  # Decision Transformer
    DIFFUSION = "diffusion"  # Diffusion Policy
    HEURISTIC = "heuristic"  # 启发式（baseline）


# ============================================================
# 单步观测（支持 DT 历史序列和 Diffusion 当前状态）
# ============================================================

class StateObservation(BaseModel):
    """
    单个时间步的完整状态
    兼容 DT 的 StepObservation
    """
    t: int = Field(description="时间步索引")
    robots: List[RobotState] = Field(description="所有机器人的状态")
    jobs: List[JobSpec] = Field(description="所有待调度任务")
    stations: List[StationState] = Field(description="所有工作站的状态")
    lanes: Optional[List[LaneInfo]] = Field(default=None, description="车道信息（可选）")
    global_time: float = Field(default=0.0, description="当前全局时间")
    
    class Config:
        use_enum_values = False


# ============================================================
# 扩展动作定义（预留交接点、时间窗）
# ============================================================

class EnhancedRobotAction(BaseModel):
    """
    增强的机器人动作
    v1: 只使用 assign_job_id
    v2+: 扩展支持 handoff_point, time_window
    """
    robot_id: str
    action_type: str = Field(default="assign_job", description="assign_job | idle | charge | handoff")
    assign_job_id: Optional[str] = Field(None, description="分配的任务 ID，None 表示 idle")
    
    # v2+ 预留字段（当前可为空）
    handoff_point: Optional[Dict[str, float]] = Field(None, description="交接点坐标 {x, y} (v2+)")
    time_window: Optional[Dict[str, float]] = Field(None, description="时间窗 {start, end} (v2+)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="动作置信度")
    
    class Config:
        use_enum_values = False


# ============================================================
# 候选方案（用于 Diffusion 多样性采样）
# ============================================================

class ActionCandidate(BaseModel):
    """
    一个候选动作方案
    """
    actions: List[EnhancedRobotAction] = Field(description="所有机器人的联合动作")
    score: float = Field(description="方案评分（越低越好，如预计完成时间）")
    rank: int = Field(description="排名（1=最优）")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")


# ============================================================
# 统一策略请求
# ============================================================

class UnifiedPolicyRequest(BaseModel):
    """
    统一策略推理请求
    支持两种输入模式：
    1. 序列模式（DT）：trajectory = K 步历史
    2. 单步模式（Diffusion）：trajectory = [当前状态]
    """
    trajectory: List[StateObservation] = Field(
        description="状态序列：DT 需要 K 步历史，Diffusion 只需最后一步"
    )
    backend: PolicyBackend = Field(
        default=PolicyBackend.DT,
        description="策略后端：dt | diffusion | heuristic"
    )
    
    # Diffusion 专属参数
    num_candidates: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Diffusion 采样候选数（只对 diffusion 生效）"
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=0.0,
        description="引导强度（v2+，预留）"
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="采样温度（v2+，预留）"
    )
    seed: Optional[int] = Field(
        default=None,
        description="随机种子（用于可复现采样）"
    )
    
    # DT 专属参数
    return_logits: bool = Field(
        default=False,
        description="是否返回 logits（DT 调试用）"
    )
    
    class Config:
        use_enum_values = False


# ============================================================
# 统一策略响应
# ============================================================

class UnifiedPolicyResponse(BaseModel):
    """
    统一策略推理响应
    """
    actions: List[EnhancedRobotAction] = Field(
        description="主推荐动作（最优方案）"
    )
    
    # 元信息
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="元信息：backend, version, model_id, inference_time_ms 等"
    )
    
    # Diffusion 专属：候选方案
    candidates: Optional[List[ActionCandidate]] = Field(
        default=None,
        description="候选动作方案列表（仅 diffusion 返回，按 score 升序）"
    )
    
    # DT 专属：logits
    logits: Optional[Dict[str, Any]] = Field(
        default=None,
        description="动作 logits（仅 DT 在 return_logits=True 时返回）"
    )
    
    class Config:
        use_enum_values = False


# ============================================================
# 策略后端抽象接口
# ============================================================

class PolicyBackendInterface:
    """
    策略后端抽象基类
    所有后端（DT, Diffusion, Heuristic）必须实现此接口
    """
    
    def act(self, request: UnifiedPolicyRequest) -> UnifiedPolicyResponse:
        """
        推理接口
        
        Args:
            request: 统一请求格式
        
        Returns:
            response: 统一响应格式
        """
        raise NotImplementedError
    
    def get_backend_name(self) -> str:
        """返回后端名称"""
        raise NotImplementedError
    
    def get_version(self) -> str:
        """返回版本号"""
        raise NotImplementedError


# ============================================================
# 训练数据格式（共享）
# ============================================================

class EpisodeStep(BaseModel):
    """
    单步训练数据
    兼容 DT 和 Diffusion 训练
    """
    obs: StateObservation = Field(description="观测状态")
    action: List[EnhancedRobotAction] = Field(description="专家动作（联合动作）")
    reward: float = Field(description="即时奖励")
    done: bool = Field(description="是否结束")
    info: Optional[Dict[str, Any]] = Field(default=None, description="额外信息")


class Episode(BaseModel):
    """
    完整 episode
    """
    episode_id: str
    steps: List[EpisodeStep]
    total_reward: float
    meta: Optional[Dict[str, Any]] = Field(default=None)
