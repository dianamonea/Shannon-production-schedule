"""
Pydantic schemas for MADT Policy Service
定义请求、响应及中间格式的数据结构
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================
# 枚举定义
# ============================================================

class RobotStatus(str, Enum):
    """机器人状态"""
    IDLE = "idle"
    WORKING = "working"
    CHARGING = "charging"
    ERROR = "error"


class JobStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StationType(str, Enum):
    """工作站类型"""
    ASSEMBLY = "assembly"
    QUALITY_CHECK = "quality_check"
    PACKAGING = "packaging"
    STORAGE = "storage"


# ============================================================
# 基础状态定义
# ============================================================

class RobotState(BaseModel):
    """机器人状态"""
    robot_id: str
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    status: RobotStatus = RobotStatus.IDLE
    current_job_id: Optional[str] = None
    battery_level: float = Field(ge=0.0, le=100.0, default=100.0)
    load_capacity: float = Field(default=0.0)
    
    class Config:
        use_enum_values = False


class JobSpec(BaseModel):
    """任务规格"""
    job_id: str
    job_type: str
    source_station_id: str
    target_station_id: str
    deadline: float = Field(default=float('inf'))  # 时间步
    priority: int = Field(ge=0, le=100, default=50)
    required_capacity: float = Field(default=0.0)
    
    class Config:
        use_enum_values = False


class StationState(BaseModel):
    """工作站状态"""
    station_id: str
    station_type: StationType
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    is_available: bool = True
    queued_jobs: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = False


class LaneInfo(BaseModel):
    """车道信息（可选，用于多层工厂）"""
    lane_id: str
    level: int = Field(ge=0, default=0)
    robots_on_lane: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = False


# ============================================================
# 时间步状态（K 步序列的一个时间片）
# ============================================================

class StepObservation(BaseModel):
    """单个时间步的观测"""
    t: int = Field(description="时间步索引")
    robots: List[RobotState] = Field(description="所有机器人的状态")
    jobs: List[JobSpec] = Field(description="所有待调度任务")
    stations: List[StationState] = Field(description="所有工作站的状态")
    lanes: Optional[List[LaneInfo]] = Field(default=None, description="车道信息（可选）")
    global_time: float = Field(default=0.0, description="当前全局时间")
    
    class Config:
        use_enum_values = False


# ============================================================
# 动作定义
# ============================================================

class RobotAction(BaseModel):
    """单个机器人的动作"""
    robot_id: str
    action_type: str = Field(default="assign_job", description="assign_job | idle | charge | ...")
    assign_job_id: Optional[str] = None  # action_type == "assign_job" 时有效
    target_position: Optional[Dict[str, float]] = None  # 可选
    
    class Config:
        use_enum_values = False


class ActionDistribution(BaseModel):
    """动作分布（含 logits，用于训练和解释）"""
    robot_id: str
    action_type: str
    assign_job_id: Optional[str] = None
    logits: Optional[Dict[str, float]] = None  # job_id -> logit（可选，用于调试）
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# ============================================================
# API 请求/响应
# ============================================================

class PolicyActRequest(BaseModel):
    """
    Policy Service 推理请求
    输入：K 步的观测序列
    """
    trajectory: List[StepObservation] = Field(
        description="K 步的观测序列（time-ordered）"
    )
    return_logits: bool = Field(
        default=False,
        description="是否返回 logits（用于调试）"
    )
    
    @validator("trajectory")
    def trajectory_not_empty(cls, v):
        if not v:
            raise ValueError("trajectory must not be empty")
        return v


class PolicyActResponse(BaseModel):
    """
    Policy Service 推理响应
    输出：所有机器人的动作
    """
    actions: List[RobotAction] = Field(
        description="每个机器人的动作"
    )
    action_distributions: Optional[List[ActionDistribution]] = Field(
        default=None,
        description="动作分布及 logits（可选）"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据：policy_version, timestamp, etc."
    )
    
    class Config:
        use_enum_values = False


# ============================================================
# 训练数据格式
# ============================================================

class TrajectoryStep(BaseModel):
    """训练数据：单步 (obs_t, act_t, reward_t, done_t)"""
    obs: StepObservation
    action: List[RobotAction]  # 所有机器人的动作
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = False


class Episode(BaseModel):
    """完整轨迹（用于 JSONL 保存）"""
    episode_id: str
    steps: List[TrajectoryStep]
    total_reward: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = False


class DatasetConfig(BaseModel):
    """数据集配置"""
    episode_dir: str = Field(description="存放 episode jsonl 的目录")
    sequence_length: int = Field(default=4, ge=1, description="K：滑窗长度")
    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1)
    shuffle: bool = Field(default=True)


class ModelConfig(BaseModel):
    """模型配置"""
    hidden_dim: int = Field(default=256)
    num_layers: int = Field(default=4, ge=1)
    num_heads: int = Field(default=8, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    max_robots: int = Field(default=10, description="最大机器人数（用于 padding）")
    max_jobs: int = Field(default=50, description="最大任务数（用于 padding）")
    max_stations: int = Field(default=20, description="最大工作站数")


class TrainingConfig(BaseModel):
    """训练配置"""
    lr: float = Field(default=1e-4)
    epochs: int = Field(default=100)
    warmup_steps: int = Field(default=1000)
    weight_decay: float = Field(default=1e-5)
    device: str = Field(default="cpu")  # "cpu" or "cuda"
    checkpoint_dir: str = Field(default="./checkpoints")
    log_interval: int = Field(default=10)


# ============================================================
# 向量化中间格式
# ============================================================

class VectorizedState(BaseModel):
    """向量化后的状态（中间格式）"""
    robot_embeddings: List[List[float]]  # [num_robots_padded, embed_dim]
    robot_mask: List[int]  # [num_robots_padded] (0=padding, 1=valid)
    
    job_embeddings: List[List[float]]  # [num_jobs_padded, embed_dim]
    job_mask: List[int]  # [num_jobs_padded]
    
    station_embeddings: List[List[float]]  # [num_stations_padded, embed_dim]
    station_mask: List[int]  # [num_stations_padded]
    
    time_embedding: List[float]  # [time_embed_dim]
    
    class Config:
        use_enum_values = False
