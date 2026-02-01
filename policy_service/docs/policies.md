# Shannon 多智能体调度策略文档

## 概述

Shannon 支持两种并行的多智能体调度策略：

1. **Decision Transformer (DT)**: 基于 Transformer 的序列决策模型
2. **Diffusion Policy**: 基于扩散模型的生成式策略

两种策略通过统一接口集成，可根据场景需求灵活切换。

---

## 策略对比

| 特性 | Decision Transformer | Diffusion Policy |
|------|---------------------|------------------|
| **模型类型** | Transformer 序列建模 | 扩散生成模型 |
| **输入** | K 步历史序列 | 当前状态（单步） |
| **输出** | 确定性动作 | 多个候选方案 |
| **推理速度** | 快（~10-50ms） | 中等（~50-200ms） |
| **多样性** | 低（单一输出） | 高（多候选采样） |
| **适用场景** | 实时在线决策 | 规划/探索/对比 |
| **训练数据** | 需要完整轨迹 | 只需状态-动作对 |

---

## 快速开始

### 1. 环境配置

```bash
# 设置默认策略后端
export DEFAULT_POLICY_BACKEND=dt  # 或 diffusion

# 设置模型检查点路径
export DT_CHECKPOINT_PATH=./checkpoints/best_model.pt
export DIFFUSION_CHECKPOINT_PATH=./checkpoints/best_diffusion_model.pt

# 设置推理设备
export POLICY_DEVICE=cpu  # 或 cuda
```

### 2. 启动服务

```bash
# 方式 1：使用统一服务（推荐）
python -m uvicorn policy_service.unified_service:app --port 8000

# 方式 2：使用原有服务（仅 DT）
python -m uvicorn policy_service.app:app --port 8000
```

### 3. 运行演示

```bash
# 对比两种策略
python policy_service/scripts/run_demo.py --mode compare

# 只测试 DT
python policy_service/scripts/run_demo.py --mode dt

# 只测试 Diffusion（生成 5 个候选）
python policy_service/scripts/run_demo.py --mode diffusion --num_candidates 5
```

---

## 使用指南

### API 接口

#### 1. 统一推理接口

**端点**: `POST /policy/act`

**请求格式**:
```json
{
  "trajectory": [
    {
      "t": 0,
      "robots": [...],
      "jobs": [...],
      "stations": [...],
      "global_time": 0.0
    }
  ],
  "backend": "dt",  // 或 "diffusion"
  "num_candidates": 5,  // Diffusion 专属
  "temperature": 1.0,   // Diffusion 专属
  "seed": 42,           // 可选，用于可复现
  "return_logits": false  // DT 专属
}
```

**响应格式**:
```json
{
  "actions": [
    {
      "robot_id": "r0",
      "action_type": "assign_job",
      "assign_job_id": "j3",
      "handoff_point": null,  // v2+ 预留
      "time_window": null,    // v2+ 预留
      "confidence": 1.0
    },
    ...
  ],
  "meta": {
    "backend": "diffusion",
    "version": "1.0",
    "model_id": "best_diffusion_model",
    "inference_time_ms": 85.3,
    "num_candidates": 5
  },
  "candidates": [  // 仅 Diffusion 返回
    {
      "actions": [...],
      "score": 12.5,
      "rank": 1
    },
    ...
  ]
}
```

#### 2. DT 专用接口

**端点**: `POST /policy/dt/act`

向后兼容，等同于设置 `backend=dt`。

#### 3. Diffusion 专用接口

**端点**: `POST /policy/diffusion/act`

等同于设置 `backend=diffusion`。

#### 4. 健康检查

**端点**: `GET /health`

**响应**:
```json
{
  "status": "healthy",
  "backends": {
    "dt": {
      "loaded": true,
      "name": "dt",
      "version": "1.0"
    },
    "diffusion": {
      "loaded": true,
      "name": "diffusion",
      "version": "1.0"
    }
  }
}
```

---

## Decision Transformer (DT)

### 特点

- **历史依赖**: 需要 K 步（通常 3-10 步）历史观测
- **确定性**: 每次推理返回唯一最优动作
- **快速**: 单次前向传播，推理快
- **适合**: 实时在线决策、需要快速响应的场景

### 输入要求

```python
trajectory = [
    obs_t_k,  # K 步前
    obs_t_k1,
    ...
    obs_t,    # 当前时刻
]
```

每个 `obs` 包含：
- `t`: 时间步
- `robots`: 所有机器人状态
- `jobs`: 所有任务
- `stations`: 所有工作站

### 训练

```bash
# 使用现有训练脚本
python policy_service/training/train.py --config configs/v1_bc.yaml
```

### 多具身智能体生产调度的应用场景

#### 适用于 DT 的场景

**1. 实时流水线调度**
- 场景：生产流水线上多个机器人需要快速响应
- 特征：
  - 历史状态稳定（过去 K 步观测连贯）
  - 需要低延迟决策（< 100ms）
  - 动作模式相对确定（重复性任务）
- 调用方式：
  ```python
  request = {
      "trajectory": [obs_t-2, obs_t-1, obs_t],  # 3 步历史
      "backend": "dt",
      "return_logits": True  # 可选：返回动作置信度
  }
  ```

**2. 已知工作流的循环调度**
- 场景：重复的装配任务、转运任务等
- 特征：
  - 任务序列基本一致
  - 大量历史数据用于训练
  - 可预测的机器人运动轨迹
- 示例：汽车制造装配线、电子产品组装

**3. 高吞吐量低间延时间窗**
- 场景：高峰时期需要快速调度，不允许等待
- 特征：
  - 秒级 SLA 保障
  - CPU/GPU 资源充足
  - 可接受单一动作输出

**4. 实时容错与快速恢复**
- 场景：机器人故障时快速重新分配任务
- 特征：
  - 需要 < 50ms 决策时间
  - 历史记录清晰（能反映当前系统状态）
  - DT 可学习到故障模式的快速补偿

---

## Diffusion Policy

### 特点

- **当前状态**: 只需最后一个观测
- **多样性**: 采样生成多个候选方案
- **灵活**: 支持温度控制、引导采样（v2+）
- **适合**: 离线规划、探索不同策略、需要备选方案

### 输入要求

```python
trajectory = [obs_t]  # 只需当前观测
```

### 推理参数

- **num_candidates** (1-20): 采样候选数，越多越慢但覆盖更广
- **temperature** (>0): 采样温度，>1 更随机，<1 更确定
- **seed**: 固定种子可复现结果

### 训练

#### Step 1: 生成基线数据（如果没有专家数据）

```bash
# 使用启发式策略生成数据
python policy_service/heuristic_baseline.py \
  --num_episodes 100 \
  --num_steps 50 \
  --strategy earliest_deadline \
  --output_dir ./data/baseline
```

支持的启发式策略：
- `earliest_deadline`: 最早截止时间优先
- `nearest_distance`: 最近距离优先
- `min_completion_time`: 最小完成时间优先

#### Step 2: 训练 Diffusion 模型

```bash
python -m policy_service.diffusion_policy.train_imitation \
  --config policy_service/configs/diffusion_v1.yaml \
  --device cuda  # 或 cpu
```

配置文件 `configs/diffusion_v1.yaml`:
```yaml
model:
  max_robots: 10
  max_jobs: 50
  num_diffusion_steps: 10  # 扩散步数
  hidden_dim: 256
  num_layers: 4

training:
  batch_size: 64
  num_epochs: 100
  lr: 3.0e-4

data:
  train_path: ./data/baseline  # 训练数据目录
```

### 多具身智能体生产调度的应用场景

#### 适用于 Diffusion 的场景

**1. 复杂任务冲突解决**
- 场景：多种任务类型，存在资源竞争和优先级冲突
- 特征：
  - 任务动作分布多模态（同一状态可能有多个最优动作）
  - 需要生成多个可行方案供上层评估
  - 某些任务可能被多个机器人抢占
- 调用方式：
  ```python
  request = {
      "trajectory": [obs_t],
      "backend": "diffusion",
      "num_candidates": 5,    # 生成 5 个不同的分配方案
      "temperature": 1.2      # 增加多样性
  }
  ```
- 示例应用：
  - 订单冲突时的多方案规划
  - 优先级动态调整下的备选策略
  - A/B 测试不同调度策略的效果

**2. 离线规划与全局优化**
- 场景：非实时的批量任务规划，需要全局最优
- 特征：
  - 允许 200-500ms 规划时间
  - 生成多个候选方案，选择最优的
  - 可结合制约条件进行后处理
- 示例：
  - 每小时一次的批量规划
  - 班次内优化与负载均衡
  - 长期任务排序优化

**3. 探索与容错恢复**
- 场景：系统发生异常（机器人故障、任务插队）时的快速恢复
- 特征：
  - 需要快速生成多个替代方案
  - 动作空间需要被充分探索
  - 历史数据可能不完整或过时
- 调用方式：
  ```python
  request = {
      "trajectory": [obs_current_state],
      "backend": "diffusion",
      "num_candidates": 10,   # 生成 10 个恢复方案
      "temperature": 1.5,     # 高探索性
      "seed": None            # 随机采样
  }
  ```
- 场景示例：
  - 机器人动力失效 → 快速重新分配任务
  - 任务紧急插队 → 生成多个调度调整方案
  - 工作站故障 → 生成替代路线

**4. 策略对比与决策支持**
- 场景：需要比较多个决策方案的质量
- 特征：
  - Diffusion 生成多个候选
  - DT 快速筛选最优
  - 为管理者提供决策依据
- 工作流：
  ```python
  # Step 1: Diffusion 生成多个方案
  diffusion_response = diffusion_request({
      "backend": "diffusion",
      "num_candidates": 5
  })
  
  # Step 2: 评估每个候选方案
  for candidate in diffusion_response.candidates:
      score = evaluate_quality(candidate)
      candidate.quality_score = score
  
  # Step 3: DT 辅助快速决策
  dt_response = dt_request({
      "backend": "dt",
      "context": "select best from candidates"
  })
  ```

**5. 新场景适应与数据不足**
- 场景：生产流程升级、新机器人集成、新工作站启用
- 特征：
  - 历史数据不足以训练可靠的 DT
  - 需要快速生成合理的初始方案
  - 边采边学（在线学习）
- 示例：
  - 新产线启动：用 Diffusion 启动，逐步积累数据训练 DT
  - 新机器人集成：Diffusion 先生成可行方案，逐步调整
  - 季节性任务：Diffusion 快速适应新需求

---

## 切换策略

### 方式 1: 请求中指定

```python
import requests

# 使用 DT
response = requests.post("http://localhost:8000/policy/act", json={
    "trajectory": [...],
    "backend": "dt"
})

# 使用 Diffusion
response = requests.post("http://localhost:8000/policy/act", json={
    "trajectory": [...],
    "backend": "diffusion",
    "num_candidates": 5
})
```

### 方式 2: 环境变量设置默认

```bash
# 设置默认为 Diffusion
export DEFAULT_POLICY_BACKEND=diffusion

# 启动服务（请求中不指定 backend 时使用默认）
python -m uvicorn policy_service.unified_service:app --port 8000
```

### 方式 3: 使用专用端点

```python
# DT 端点
requests.post("http://localhost:8000/policy/dt/act", json={...})

# Diffusion 端点
requests.post("http://localhost:8000/policy/diffusion/act", json={...})
```

---

## 混合策略协同调用

在多具身智能体协同的生产调度中，DT 和 Diffusion 可以形成**互补的决策管道**：

### 场景 1: 分级决策（Diffusion → DT）

用 Diffusion 生成多个候选方案，再用 DT 快速筛选最优的。

```python
# 第一阶段：Diffusion 生成多个候选
diffusion_response = requests.post("/policy/diffusion/act", json={
    "trajectory": [obs],
    "num_candidates": 10,
    "temperature": 1.2
})

candidates = diffusion_response.candidates  # 10 个不同的分配方案

# 第二阶段：DT 快速筛选
# 模拟执行每个候选，或使用 DT 的 logits 评分
scores = []
for candidate in candidates:
    # 方案1：快速评估函数
    score = quick_evaluate(candidate)
    scores.append(score)

best_idx = np.argmax(scores)
final_action = candidates[best_idx]
```

**适用场景**：
- 高优先级、影响面大的任务分配
- 需要保证决策质量同时要求快速响应
- 可容忍 100-300ms 决策时间

### 场景 2: 冗余容错（DT Primary + Diffusion Backup）

主要用 DT 保证实时性，当 DT 输出置信度低时，调用 Diffusion 生成备选方案。

```python
# 第一阶段：尝试用 DT
dt_response = requests.post("/policy/dt/act", json={
    "trajectory": [obs_t-2, obs_t-1, obs_t],
    "return_logits": True
})

dt_confidence = np.max(dt_response.actions[0].logits)

# 第二阶段：如果置信度不足，使用 Diffusion
if dt_confidence < CONFIDENCE_THRESHOLD:  # 如 0.6
    diffusion_response = requests.post("/policy/diffusion/act", json={
        "trajectory": [obs],
        "num_candidates": 3,
        "temperature": 1.0
    })
    # 使用 Diffusion 的最高置信候选
    final_action = diffusion_response.actions
else:
    # 使用 DT 的输出
    final_action = dt_response.actions
```

**适用场景**：
- 需要 99.9% 的决策可靠性
- 实时系统（不能等待 Diffusion）
- 有备用时间进行二阶段验证

### 场景 3: 时间驱动切换（高峰 DT + 低谷 Diffusion）

根据系统负载动态选择策略。

```python
import time

def smart_dispatch(obs, system_load):
    """
    高峰期（负载 > 80%）：使用快速的 DT
    低谷期（负载 < 50%）：使用质量好的 Diffusion
    """
    if system_load > 0.8:
        # 高峰期：快速响应
        backend = "dt"
        config = {
            "trajectory": [obs],
            "backend": backend,
            "return_logits": False
        }
    elif system_load < 0.5:
        # 低谷期：质量优化
        backend = "diffusion"
        config = {
            "trajectory": [obs],
            "backend": backend,
            "num_candidates": 8,
            "temperature": 0.9
        }
    else:
        # 中等负载：均衡
        backend = "diffusion"
        config = {
            "trajectory": [obs],
            "backend": backend,
            "num_candidates": 3,
            "temperature": 1.0
        }
    
    response = requests.post("/policy/act", json=config)
    return response.actions, backend
```

**适用场景**：
- 24 小时连续运行的生产系统
- 分时段 SLA 要求不同
- 有预测性的负载波动

### 场景 4: 异常驱动切换（故障恢复）

机器人故障或任务异常时，快速切换到 Diffusion 进行恢复。

```python
def dispatch_with_fallback(obs, is_emergency=False):
    """
    正常情况：使用 DT 快速响应
    应急情况：使用 Diffusion 探索所有可能
    """
    try:
        if is_emergency:
            # 应急模式：生成 10 个恢复方案
            response = requests.post("/policy/diffusion/act", json={
                "trajectory": [obs],
                "num_candidates": 10,
                "temperature": 1.5  # 高探索性
            })
            # 选择满足约束的方案
            valid_candidates = [
                c for c in response.candidates 
                if is_feasible(c)
            ]
            return valid_candidates[0] if valid_candidates else None
        else:
            # 正常模式：DT 快速决策
            response = requests.post("/policy/dt/act", json={
                "trajectory": last_k_obs,
                "return_logits": True
            })
            return response.actions
    except Exception as e:
        logger.warning(f"Decision failed: {e}, fallback to Diffusion")
        # 异常时自动降级到 Diffusion
        response = requests.post("/policy/diffusion/act", json={
            "trajectory": [obs],
            "num_candidates": 5,
            "temperature": 1.2
        })
        return response.actions
```

**适用场景**：
- 需要高可用性（多级容错）
- 故障恢复时间要求严格
- 有自动降级机制

### 场景 5：学习反馈循环（Diffusion 数据生成 → DT 训练）

用 Diffusion 生成多样化的数据，用于持续改进 DT。

```python
def continuous_learning_loop():
    """
    1. Diffusion 生成多样化的训练数据
    2. 评估质量
    3. 高质量数据用于训练新的 DT
    4. 部署新 DT，迭代改进
    """
    
    for epoch in range(num_epochs):
        # 收集数据：使用 Diffusion 在各种系统状态下采样
        training_data = []
        
        for obs in diverse_observations:
            response = requests.post("/policy/diffusion/act", json={
                "trajectory": [obs],
                "num_candidates": 20,
                "temperature": 1.5  # 多样性
            })
            
            for candidate in response.candidates:
                # 评估这个候选方案的质量
                quality_score = evaluate_in_simulator(candidate)
                
                if quality_score > QUALITY_THRESHOLD:
                    training_data.append({
                        "obs": obs,
                        "action": candidate,
                        "return": quality_score
                    })
        
        # 训练新的 DT
        if len(training_data) > MIN_DATA_SIZE:
            train_dt(training_data)
            evaluate_and_deploy_if_better()
```

**适用场景**：
- 持续优化的学习系统
- 有充足的计算资源
- 能够离线评估和模拟

---

## 选择哪种策略？决策框架

| 因素 | 优先选 DT | 优先选 Diffusion | 混合 |
|------|----------|-----------------|------|
| **响应时间** | < 50ms | 100-500ms | 优先 DT，低置信时用 Diffusion |
| **决策质量** | 稳定可预测 | 需要多样性 | Diffusion 生成，DT 筛选 |
| **历史依赖** | 有完整历史 | 数据不足 | 历史充足用 DT，否则用 Diffusion |
| **任务多样性** | 低（重复性强） | 高（新任务多） | 新任务用 Diffusion，熟悉任务用 DT |
| **系统稳定性** | 高（少异常） | 低（频繁故障） | 正常用 DT，故障用 Diffusion |
| **计算资源** | GPU 不足 | GPU 充足 | 根据实时负载动态选择 |

---

## 数据格式

### 训练数据（共享格式）

两种策略使用相同的 episode 数据格式（JSONL 或 JSON）：

**JSONL 格式**（每行一个 step）:
```jsonl
{"obs": {...}, "action": [...], "reward": 1.5, "done": false}
{"obs": {...}, "action": [...], "reward": 2.0, "done": false}
...
```

**JSON 格式**（完整 episode）:
```json
{
  "episode_id": "ep_001",
  "steps": [
    {"obs": {...}, "action": [...], "reward": 1.5, "done": false},
    ...
  ],
  "total_reward": 50.0
}
```

### 观测格式

```json
{
  "t": 0,
  "robots": [
    {
      "robot_id": "r0",
      "position": {"x": 10.5, "y": 20.3},
      "status": "idle",
      "current_job_id": null,
      "battery_level": 95.0,
      "load_capacity": 0.0
    }
  ],
  "jobs": [
    {
      "job_id": "j0",
      "job_type": "transport",
      "source_station_id": "s1",
      "target_station_id": "s3",
      "deadline": 30.0,
      "priority": 80,
      "required_capacity": 0.0
    }
  ],
  "stations": [
    {
      "station_id": "s1",
      "station_type": "assembly",
      "position": {"x": 5.0, "y": 10.0},
      "is_available": true,
      "queued_jobs": []
    }
  ],
  "global_time": 0.0
}
```

### 动作格式

```json
[
  {
    "robot_id": "r0",
    "action_type": "assign_job",
    "assign_job_id": "j3"
  },
  {
    "robot_id": "r1",
    "action_type": "idle",
    "assign_job_id": null
  }
]
```

---

## 高级功能（v2+ 路线图）

### 扩展动作空间

当前 v1 只支持任务分配（`assign_job_id`），v2+ 将扩展到：

```json
{
  "robot_id": "r0",
  "action_type": "assign_job",
  "assign_job_id": "j3",
  "handoff_point": {"x": 15.0, "y": 25.0},  // 交接点
  "time_window": {"start": 5.0, "end": 10.0}  // 时间窗
}
```

### 约束引导采样（Diffusion）

```python
response = requests.post("/policy/diffusion/act", json={
    "trajectory": [...],
    "num_candidates": 10,
    "guidance_scale": 2.0,  // 增强约束满足
    "constraints": {
        "max_distance_per_robot": 50.0,
        "deadline_slack": 5.0
    }
})
```

### 可行性修复

自动修复不可行的动作（如任务冲突、容量超限）。

---

## 常见问题

### Q1: 如何选择策略？

- **实时在线**: 用 DT（快速、确定）
- **离线规划**: 用 Diffusion（多样性、备选）
- **混合**: 用 Diffusion 生成候选，用 DT 快速筛选

### Q2: Diffusion 推理太慢？

- 减少 `num_candidates`（1-3 即可）
- 减少 `num_diffusion_steps`（5-8 步）
- 使用 GPU（`export POLICY_DEVICE=cuda`）

### Q3: 如何提高 Diffusion 质量？

- 使用更多高质量训练数据
- 增加 `num_diffusion_steps`（10-20 步）
- 调整采样温度（0.8-1.2）

### Q4: 两种策略可以同时加载吗？

是的，统一服务会同时加载两种后端，根据请求动态路由。

---

## 学术参考文献

### Decision Transformer 相关论文

#### 1. Decision Transformer（核心论文）

**标题**: Decision Transformer: Reinforcement Learning via Sequence Modeling  
**作者**: Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch  
**会议**: NeurIPS 2021  
**发表**: arXiv:2106.01345 [cs.LG], 2021年6月2日  
**链接**: https://arxiv.org/abs/2106.01345

**核心思想**:
- 将强化学习抽象为序列建模问题
- 利用 Transformer 架构的简洁性和可扩展性
- 通过因果掩码 Transformer 输出最优动作
- 基于期望回报条件的自回归模型生成未来动作
- 在 Atari、OpenAI Gym 等任务上匹配或超越 SOTA 离线强化学习基线

**解决的问题**:
- 规避离线强化学习中价值函数/策略梯度不稳定、难优化的问题
- 在不与环境交互的条件下，从离线轨迹中直接学习策略
- 在多模态行为分布场景中稳定地产生高回报动作

**核心架构与实现方法**:
- 将轨迹重排为序列：$(R_t, s_t, a_t)$ 的 token 流
- 采用 **因果掩码 Transformer** 进行自回归建模
- 输入为期望回报 + 历史状态/动作，预测下一步动作
- 训练目标为最大化动作序列的对数似然（序列建模损失）

**BibTeX**:
```bibtex
@inproceedings{chen2021decision,
  title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author={Chen, Lili and Lu, Kevin and Rajeswaran, Aravind and Lee, Kimin and Grover, Aditya and Laskin, Michael and Abbeel, Pieter and Srinivas, Aravind and Mordatch, Igor},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

#### 2. Transformer 架构基础

**标题**: Attention Is All You Need  
**作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin  
**会议**: NeurIPS 2017  
**发表**: arXiv:1706.03762 [cs.CL], 2017年6月12日  
**链接**: https://arxiv.org/abs/1706.03762

**核心思想**:
- 提出完全基于注意力机制的 Transformer 架构
- 摒弃循环和卷积网络
- 在机器翻译任务上超越当时所有模型
- WMT 2014 英德翻译 BLEU=28.4（提升 2+ BLEU）
- 高度并行化，训练速度快

**解决的问题**:
- RNN/CNN 在长序列依赖建模上的效率与并行性瓶颈
- 序列到序列任务中编码器-解码器的训练效率问题

**核心架构与实现方法**:
- 多头自注意力（Multi-Head Self-Attention）替代循环结构
- 位置编码注入序列顺序信息
- 前馈网络 + 残差连接 + LayerNorm 组成基础模块
- 编码器-解码器堆叠，实现高并行的序列建模

**BibTeX**:
```bibtex
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5998--6008},
  year={2017}
}
```

---

### Diffusion Policy 相关论文

#### 1. Diffusion Policy（核心论文）

**标题**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion  
**作者**: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song  
**会议**: RSS 2023（扩展版发表于 arXiv）  
**发表**: arXiv:2303.04137 [cs.RO], 2023年3月7日  
**链接**: https://arxiv.org/abs/2303.04137  
**项目主页**: http://diffusion-policy.cs.columbia.edu/

**核心思想**:
- 将机器人策略表示为条件去噪扩散过程
- 学习动作分布的得分函数梯度
- 通过 Langevin 动力学迭代优化生成动作
- 优雅处理多模态动作分布
- 在 12 个机器人操作任务上平均提升 46.9%
- 包含循环视野控制、视觉条件、时间序列扩散 Transformer

**解决的问题**:
- 传统行为克隆在多模态动作分布上的模式坍塌
- 高维动作空间下策略学习不稳定、泛化差
- 复杂机器人任务需要多候选动作生成与规划能力

**核心架构与实现方法**:
- **条件扩散模型**：以观测为条件，对动作序列进行去噪生成
- 通过多步扩散/去噪迭代生成动作（采样式推理）
- 采用时间序列扩散 Transformer 作为去噪网络
- 使用 receding horizon 控制（滚动规划）进行在线执行

**BibTeX**:
```bibtex
@article{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  journal={arXiv preprint arXiv:2303.04137},
  year={2023}
}
```

#### 2. Discrete Diffusion（Masking Diffusion）

**标题**: Structured Denoising Diffusion Models in Discrete State-Spaces  
**作者**: Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg  
**会议**: NeurIPS 2021  
**发表**: arXiv:2107.03006 [cs.LG], 2021年7月7日  
**链接**: https://arxiv.org/abs/2107.03006

**核心思想**:
- 引入离散去噪扩散概率模型（D3PM）
- 扩展 DDPM 到离散状态空间（文本、分类等）
- 支持多种转移矩阵：高斯核、最近邻、**吸收状态（Masking）**
- Masking 方法连接扩散模型与自回归/掩码生成模型
- 在文本生成（LM1B）和图像生成（CIFAR-10）上取得强结果

**解决的问题**:
- 传统 DDPM 仅适用于连续数据（图像/音频），难以处理离散符号序列
- 离散数据的扩散噪声建模缺乏统一框架

**核心架构与实现方法**:
- 使用**离散状态转移矩阵**定义前向噪声过程
- 引入 **Masking/吸收状态** 使扩散过程更适配离散 token
- 反向过程通过分类分布预测干净 token
- 训练目标结合变分下界与交叉熵辅助损失

**我们的使用**:  
我们的 Diffusion Policy 实现采用 **Masking Diffusion** 方法，因为它更适合离散任务分配场景。通过逐步 unmask tokens 的方式生成动作序列。

**BibTeX**:
```bibtex
@inproceedings{austin2021structured,
  title={Structured Denoising Diffusion Models in Discrete State-Spaces},
  author={Austin, Jacob and Johnson, Daniel D and Ho, Jonathan and Tarlow, Daniel and van den Berg, Rianne},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

#### 3. DDPM（扩散模型基础）

**标题**: Denoising Diffusion Probabilistic Models  
**作者**: Jonathan Ho, Ajay Jain, Pieter Abbeel  
**会议**: NeurIPS 2020  
**发表**: arXiv:2006.11239 [cs.LG], 2020年6月19日  
**链接**: https://arxiv.org/abs/2006.11239  
**代码**: https://github.com/hojonathanho/diffusion

**核心思想**:
- 扩散概率模型的奠基性工作
- 基于非平衡热力学的潜变量模型
- 通过加权变分下界训练
- 连接扩散模型与去噪得分匹配和 Langevin 动力学
- CIFAR-10 上 FID=3.17（当时 SOTA）

**解决的问题**:
- 生成模型在高维数据上的稳定性与样本质量难以兼得
- 需要统一的概率框架来连接扩散过程与生成建模

**核心架构与实现方法**:
- 定义前向高斯噪声链（逐步加噪）
- 训练去噪网络预测噪声或数据重建
- 通过可计算的变分下界进行最大似然训练
- 反向采样采用逐步去噪（可视作 Langevin 动力学）

**BibTeX**:
```bibtex
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6840--6851},
  year={2020}
}
```

---

### 其他相关工作

#### 多智能体强化学习

- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017
- Rashid et al., "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning", ICML 2018

#### 调度问题的深度学习

- Zhang et al., "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning", NeurIPS 2020
- Park et al., "Learning to Schedule Job-Shop Problems: Representation and Policy Learning using Graph Neural Network and Reinforcement Learning", IJCAI 2021

---

**版本**: 2.0.0  
**更新日期**: 2026-01-30  
**维护**: Shannon Team
