# MADT Policy Service - Quick Start Guide

## 概述

**Multi-Agent Decision Transformer (MADT) Policy Service** 是一个集中式多智能体强化学习决策系统。通过 FastAPI 提供推理服务，支持离线行为克隆（BC）训练。

## 核心特性

- 🎯 **集中式决策**：单一 Transformer 模型为所有机器人生成协调动作
- 📊 **可扩展架构**：支持可变数量的机器人、任务、工作站（通过 masking）
- 🔄 **闭环学习**：运行时数据自动记录为训练样本
- 🚀 **快速推理**：CPU 推理 <100ms（可选 GPU 加速）
- 📈 **预留升级空间**：v1.5 RTG、v2 事件序列、v3 协作动作、v4 Graph pooling

## 目录结构

```
policy_service/
├── app.py                      # FastAPI 推理服务入口
├── test_madt.py               # 单元测试和 E2E 演示
├── configs/
│   └── v1_bc.yaml            # v1 行为克隆配置
├── common/
│   ├── __init__.py
│   ├── schemas.py             # Pydantic 数据定义
│   └── vectorizer.py          # 状态/动作向量化
├── training/
│   ├── __init__.py
│   ├── model.py               # Decision Transformer 实现
│   ├── dataset.py             # 数据加载器 (JSONL/Parquet)
│   └── train.py               # 训练脚本（行为克隆）
└── README.md                   # 本文件
```

## 快速开始

### 前置要求

```bash
pip install fastapi uvicorn torch numpy pydantic pyyaml
```

### 1️⃣ 运行单元测试

```bash
cd policy_service
python test_madt.py
```

**输出示例**：
```
=== Test 1: Schema Validation ===
✓ Created valid StepObservation
✓ Correctly caught validation error

=== Test 2: Vectorizer ===
✓ Vectorized step observation
✓ Robot mask correctly applied
...

✓ All tests passed!
```

### 2️⃣ 启动推理服务

```bash
# 方式 1: 直接启动（使用虚拟模型演示）
cd policy_service
uvicorn app:app --host 0.0.0.0 --port 8000

# 方式 2: 使用 Python（便于调试）
python -c "from app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

**输出**：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### 3️⃣ 测试推理 API

```bash
# 健康检查
curl http://localhost:8000/health

# 查看策略信息
curl http://localhost:8000/policy/info

# 推理请求（见下方示例）
curl -X POST http://localhost:8000/policy/act \
  -H "Content-Type: application/json" \
  -d '{"trajectory": [...], "return_logits": true}'
```

## API 使用示例

### 请求格式

```json
POST /policy/act

{
  "trajectory": [
    {
      "t": 0,
      "global_time": 0.0,
      "robots": [
        {
          "robot_id": "robot_0",
          "position": {"x": 10.0, "y": 20.0},
          "status": "idle",
          "battery_level": 85.5,
          "load_capacity": 0.0
        }
      ],
      "jobs": [
        {
          "job_id": "job_0",
          "job_type": "assembly",
          "source_station_id": "station_0",
          "target_station_id": "station_1",
          "deadline": 100.0,
          "priority": 75
        }
      ],
      "stations": [
        {
          "station_id": "station_0",
          "station_type": "assembly",
          "position": {"x": 0.0, "y": 0.0},
          "is_available": true,
          "queued_jobs": []
        }
      ],
      "lanes": null
    },
    // ... K 步（默认 K=4）
  ],
  "return_logits": true  # 可选：返回动作分布用于调试
}
```

### 响应格式

```json
{
  "actions": [
    {
      "robot_id": "robot_0",
      "action_type": "assign_job",
      "assign_job_id": "job_0",
      "target_position": null
    }
  ],
  "action_distributions": [
    {
      "robot_id": "robot_0",
      "action_type": "assign_job",
      "assign_job_id": "job_0",
      "logits": {
        "job_0": 2.34,
        "job_1": 1.23,
        "idle": 0.45
      },
      "confidence": 0.91
    }
  ],
  "meta": {
    "policy_version": "v1.0",
    "model_device": "cpu",
    "num_robots": 1,
    "num_available_jobs": 2
  }
}
```

## 训练

### 数据准备

期望数据格式（JSONL，逐行一个 episode）：

```json
{
  "episode_id": "ep_001",
  "steps": [
    {
      "obs": {
        "t": 0,
        "robots": [...],
        "jobs": [...],
        "stations": [...],
        "global_time": 0.0,
        "lanes": null
      },
      "action": [
        {"robot_id": "robot_0", "action_type": "assign_job", "assign_job_id": "job_0"}
      ],
      "reward": 0.5,
      "done": false,
      "info": {}
    },
    ...
  ],
  "total_reward": 10.5,
  "metadata": {}
}
```

将 episode 文件放入 `data/episodes/` 目录。

### 启动训练

```bash
python -m training.train --config configs/v1_bc.yaml
```

**参数调整**：编辑 `configs/v1_bc.yaml`：

```yaml
model:
  hidden_dim: 256        # 隐层维度
  num_layers: 4          # Transformer 层数
  num_heads: 8           # 多头数
  dropout: 0.1

training:
  lr: 1.0e-4            # 学习率
  epochs: 50            # 轮数
  batch_size: 32        # 批大小
  device: "cpu"         # "cpu" 或 "cuda"

dataset:
  sequence_length: 4    # K 步窗口
  train_split: 0.8      # 训练集比例
```

## 数据闭环

### 1. 记录推理和执行

在 Runtime/Simulator 中：

```python
from app import PolicyService, PolicyServiceConfig
from common.schemas import PolicyActRequest, StepObservation, TrajectoryStep

# 初始化服务
config = PolicyServiceConfig()
service = PolicyService(config)

# 推理
request = PolicyActRequest(trajectory=[obs_0, obs_1, obs_2, obs_3])
response = service.act(request)

# 执行动作并记录
executed_reward = execute_actions(response.actions)

# 保存为训练数据
step = TrajectoryStep(
    obs=obs_3,
    action=response.actions,
    reward=executed_reward,
    done=done_flag,
    info={}
)
```

### 2. 转换为训练样本

数据加载器自动：
- 加载所有 JSONL episode
- 用滑窗大小 K 切片
- 配对 (obs_sequence, target_action)
- 向量化并分批

### 3. 定期重训练

```bash
# 每 N 个 episode 后
python -m training.train --config configs/v1_bc.yaml
```

## 模型架构

### Decision Transformer (v1)

```
Input: state_seq [batch, K, state_vec_dim]
  ↓
State Embedding + Positional Encoding
  ↓
Transformer Encoder (4 layers × 8 heads × 256 hidden)
  ↓
Last Hidden State [batch, hidden_dim]
  ↓
Multi-Head Action Classifiers (per robot)
  ↓
Output: action_logits [batch, max_robots, max_actions]
```

### 状态向量化

```
state_vector = concat([
  robot_embeddings,      # [max_robots, 128]
  job_embeddings,        # [max_jobs, 128]
  station_embeddings,    # [max_stations, 128]
  time_embedding         # [128]
])
```

## 升级路线图

### v1.0 (当前) ✅
- ✅ 行为克隆 (BC)
- ✅ 集中式决策
- ✅ FastAPI 推理服务

### v1.5 (预留接口)
```python
# Return-To-Go (RTG) 条件化
class RTGDecisionTransformer(DecisionTransformer):
    def forward(self, state_seq, rtg, robot_mask=None):
        # RTG 作为额外条件
        rtg_embedding = self.rtg_encoder(rtg)
        ...
```

### v2 (预留)
- 事件序列（异步 token + delta_t embedding）
- 支持动态任务插入和取消

### v3 (预留)
- 协作动作（协作对象、交接点、时间窗）
- 可行性修复和安全过滤

### v4 (预留)
- Agent-wise DT（每个机器人一个 mini-DT）
- Graph pooling for 协调

## 常见问题

**Q: 推理速度如何？**  
A: CPU 推理 ~50-100ms（K=4, max_robots=10），GPU 推理 <10ms

**Q: 如何处理可变机器人数？**  
A: 通过 `robot_mask` 支持，padding 部分被掩码为 0

**Q: 可以实时更新模型吗？**  
A: 支持在线学习（见 v2 预留）。当前建议离线重训练后热更新

**Q: 是否支持多工厂或多楼层？**  
A: 支持通过 `lanes` 字段扩展，v2 会完整实现

## 测试覆盖

- ✅ Schema 验证 (Pydantic)
- ✅ 向量化正确性
- ✅ 模型前向 pass (shape 检查)
- ✅ API 端到端 (request → response)
- ✅ Baseline 启发式 (EDF、最近距离)

运行所有测试：
```bash
python test_madt.py
```

## 故障排除

| 问题 | 解决方案 |
|------|--------|
| `ModuleNotFoundError` | 确保 `PYTHONPATH` 包含 `policy_service` 目录 |
| `cuda out of memory` | 改用 CPU (`device: "cpu"` in YAML) 或减小 `batch_size` |
| `No training data found` | 检查 `episode_dir` 路径和 JSONL 格式 |
| API 返回 `idle` 动作 | 正常！表示无合适的任务分配 |

## 参考资源

- Decision Transformer (Chen et al., 2021): https://arxiv.org/abs/2106.01021
- MARL Survey: https://arxiv.org/abs/2109.11044
- Behavior Cloning: https://arxiv.org/abs/1805.01954

## License

MIT

---

**欢迎贡献和反馈！** 📬
