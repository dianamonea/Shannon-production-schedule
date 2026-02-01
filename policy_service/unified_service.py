"""
统一策略服务（支持 DT 和 Diffusion 两种后端）
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Optional
import os

from common.schemas_v2 import (
    UnifiedPolicyRequest,
    UnifiedPolicyResponse,
    PolicyBackend,
    PolicyBackendInterface,
)
from dt_backend import create_dt_backend
from diffusion_policy import create_diffusion_backend


app = FastAPI(
    title="Shannon Multi-Agent Policy Service",
    description="统一多智能体调度策略服务（支持 DT 和 Diffusion）",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolicyServiceManager:
    """策略服务管理器"""
    
    def __init__(self):
        self.backends: Dict[str, PolicyBackendInterface] = {}
        self.default_backend = PolicyBackend.DT
        self._load_backends()
    
    def _load_backends(self):
        """加载所有后端"""
        device = os.getenv("POLICY_DEVICE", "cpu")
        
        # 加载 DT 后端
        dt_checkpoint = os.getenv("DT_CHECKPOINT_PATH", "./checkpoints/best_model.pt")
        try:
            self.backends[PolicyBackend.DT] = create_dt_backend(
                checkpoint_path=dt_checkpoint,
                device=device,
                version="1.0",
            )
            print(f"✓ Loaded DT backend from {dt_checkpoint}")
        except Exception as e:
            print(f"✗ Failed to load DT backend: {e}")
        
        # 加载 Diffusion 后端
        diffusion_checkpoint = os.getenv("DIFFUSION_CHECKPOINT_PATH", "./checkpoints/best_diffusion_model.pt")
        try:
            self.backends[PolicyBackend.DIFFUSION] = create_diffusion_backend(
                checkpoint_path=diffusion_checkpoint,
                device=device,
                version="1.0",
            )
            print(f"✓ Loaded Diffusion backend from {diffusion_checkpoint}")
        except Exception as e:
            print(f"✗ Failed to load Diffusion backend: {e}")
        
        # 设置默认后端
        default_backend_name = os.getenv("DEFAULT_POLICY_BACKEND", "dt")
        if default_backend_name == "diffusion" and PolicyBackend.DIFFUSION in self.backends:
            self.default_backend = PolicyBackend.DIFFUSION
        elif PolicyBackend.DT in self.backends:
            self.default_backend = PolicyBackend.DT
        
        print(f"Default backend: {self.default_backend}")
    
    def get_backend(self, backend: PolicyBackend) -> PolicyBackendInterface:
        """获取指定后端"""
        if backend not in self.backends:
            raise ValueError(f"Backend {backend} not available. Available: {list(self.backends.keys())}")
        return self.backends[backend]


# 全局管理器
manager = PolicyServiceManager()


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Shannon Multi-Agent Policy Service",
        "version": "2.0.0",
        "available_backends": list(manager.backends.keys()),
        "default_backend": manager.default_backend,
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "backends": {
            backend: {
                "loaded": True,
                "name": impl.get_backend_name(),
                "version": impl.get_version(),
            }
            for backend, impl in manager.backends.items()
        }
    }


@app.post("/policy/act", response_model=UnifiedPolicyResponse)
async def policy_act(request: UnifiedPolicyRequest) -> UnifiedPolicyResponse:
    """
    统一策略推理接口
    
    - 支持通过 request.backend 指定后端（dt | diffusion）
    - 支持 DT 的历史序列输入
    - 支持 Diffusion 的多候选采样
    """
    try:
        # 获取后端
        backend = request.backend or manager.default_backend
        policy = manager.get_backend(backend)
        
        # 推理
        response = policy.act(request)
        
        return response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/policy/dt/act", response_model=UnifiedPolicyResponse)
async def dt_act(request: UnifiedPolicyRequest) -> UnifiedPolicyResponse:
    """DT 专用端点（向后兼容）"""
    request.backend = PolicyBackend.DT
    return await policy_act(request)


@app.post("/policy/diffusion/act", response_model=UnifiedPolicyResponse)
async def diffusion_act(request: UnifiedPolicyRequest) -> UnifiedPolicyResponse:
    """Diffusion 专用端点"""
    request.backend = PolicyBackend.DIFFUSION
    return await policy_act(request)


@app.get("/backends")
async def list_backends():
    """列出所有可用后端"""
    return {
        "available": [
            {
                "name": backend,
                "backend_name": impl.get_backend_name(),
                "version": impl.get_version(),
            }
            for backend, impl in manager.backends.items()
        ],
        "default": manager.default_backend,
    }


if __name__ == "__main__":
    port = int(os.getenv("POLICY_SERVICE_PORT", "8000"))
    host = os.getenv("POLICY_SERVICE_HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
