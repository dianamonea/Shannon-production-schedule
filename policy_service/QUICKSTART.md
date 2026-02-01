# MADT Policy Service - 快速开始 (5 分钟)

## 🎯 你将学到

- ✅ 运行单元测试验证安装
- ✅ 启动推理服务 API
- ✅ 发送推理请求获得动作
- ✅ 了解项目结构

---

## 1️⃣ 验证安装

```bash
cd policy_service
python test_madt.py
```

**预期输出**:
```
============================================================
MADT Policy Service - Unit Tests
============================================================

=== Test 1: Schema Validation ===
✓ Created valid StepObservation
✓ Correctly caught validation error

...

✓ All tests passed!
```

✅ 如果看到 "All tests passed"，说明安装成功！

---

## 2️⃣ 启动推理服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**输出示例**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

服务启动成功！访问 http://localhost:8000/docs 查看 Swagger UI。

---

## 3️⃣ 测试 API (在新终端)

### 健康检查

```bash
curl http://localhost:8000/health
```

**响应**:
```json
{
  "status": "healthy",
  "service": "madt_policy_service",
  "version": "v1.0"
}
```

### 查询策略信息

```bash
curl http://localhost:8000/policy/info
```

**响应**:
```json
{
  "version": "v1.0",
  "device": "cpu",
  "model_config": {
    "max_robots": 10,
    "max_jobs": 50,
    "hidden_dim": 256,
    ...
  }
}
```

### 推理请求

创建文件 `test_request.json`:

```json
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
          "load_capacity": 0.0,
          "current_job_id": null
        },
        {
          "robot_id": "robot_1",
          "position": {"x": 30.0, "y": 40.0},
          "status": "idle",
          "battery_level": 75.0,
          "load_capacity": 0.0,
          "current_job_id": null
        }
      ],
      "jobs": [
        {
          "job_id": "job_0",
          "job_type": "assembly",
          "source_station_id": "station_0",
          "target_station_id": "station_1",
          "deadline": 100.0,
          "priority": 75,
          "required_capacity": 0.0
        },
        {
          "job_id": "job_1",
          "job_type": "packaging",
          "source_station_id": "station_1",
          "target_station_id": "station_2",
          "deadline": 150.0,
          "priority": 50,
          "required_capacity": 0.0
        }
      ],
      "stations": [
        {
          "station_id": "station_0",
          "station_type": "assembly",
          "position": {"x": 0.0, "y": 0.0},
          "is_available": true,
          "queued_jobs": []
        },
        {
          "station_id": "station_1",
          "station_type": "quality_check",
          "position": {"x": 50.0, "y": 50.0},
          "is_available": true,
          "queued_jobs": []
        },
        {
          "station_id": "station_2",
          "station_type": "storage",
          "position": {"x": 100.0, "y": 100.0},
          "is_available": true,
          "queued_jobs": []
        }
      ],
      "lanes": null
    },
    {
      "t": 1,
      "global_time": 1.0,
      "robots": [
        {
          "robot_id": "robot_0",
          "position": {"x": 11.0, "y": 21.0},
          "status": "working",
          "battery_level": 84.5,
          "load_capacity": 10.0,
          "current_job_id": "job_0"
        },
        {
          "robot_id": "robot_1",
          "position": {"x": 31.0, "y": 41.0},
          "status": "idle",
          "battery_level": 74.5,
          "load_capacity": 0.0,
          "current_job_id": null
        }
      ],
      "jobs": [
        {
          "job_id": "job_1",
          "job_type": "packaging",
          "source_station_id": "station_1",
          "target_station_id": "station_2",
          "deadline": 150.0,
          "priority": 50,
          "required_capacity": 0.0
        }
      ],
      "stations": [
        {
          "station_id": "station_0",
          "station_type": "assembly",
          "position": {"x": 0.0, "y": 0.0},
          "is_available": true,
          "queued_jobs": []
        },
        {
          "station_id": "station_1",
          "station_type": "quality_check",
          "position": {"x": 50.0, "y": 50.0},
          "is_available": false,
          "queued_jobs": ["job_0"]
        },
        {
          "station_id": "station_2",
          "station_type": "storage",
          "position": {"x": 100.0, "y": 100.0},
          "is_available": true,
          "queued_jobs": []
        }
      ],
      "lanes": null
    },
    {
      "t": 2,
      "global_time": 2.0,
      "robots": [
        {
          "robot_id": "robot_0",
          "position": {"x": 12.0, "y": 22.0},
          "status": "working",
          "battery_level": 83.5,
          "load_capacity": 10.0,
          "current_job_id": "job_0"
        },
        {
          "robot_id": "robot_1",
          "position": {"x": 32.0, "y": 42.0},
          "status": "idle",
          "battery_level": 73.5,
          "load_capacity": 0.0,
          "current_job_id": null
        }
      ],
      "jobs": [
        {
          "job_id": "job_1",
          "job_type": "packaging",
          "source_station_id": "station_1",
          "target_station_id": "station_2",
          "deadline": 150.0,
          "priority": 50,
          "required_capacity": 0.0
        }
      ],
      "stations": [
        {
          "station_id": "station_0",
          "station_type": "assembly",
          "position": {"x": 0.0, "y": 0.0},
          "is_available": true,
          "queued_jobs": []
        },
        {
          "station_id": "station_1",
          "station_type": "quality_check",
          "position": {"x": 50.0, "y": 50.0},
          "is_available": false,
          "queued_jobs": ["job_0"]
        },
        {
          "station_id": "station_2",
          "station_type": "storage",
          "position": {"x": 100.0, "y": 100.0},
          "is_available": true,
          "queued_jobs": []
        }
      ],
      "lanes": null
    },
    {
      "t": 3,
      "global_time": 3.0,
      "robots": [
        {
          "robot_id": "robot_0",
          "position": {"x": 13.0, "y": 23.0},
          "status": "working",
          "battery_level": 82.5,
          "load_capacity": 10.0,
          "current_job_id": "job_0"
        },
        {
          "robot_id": "robot_1",
          "position": {"x": 33.0, "y": 43.0},
          "status": "idle",
          "battery_level": 72.5,
          "load_capacity": 0.0,
          "current_job_id": null
        }
      ],
      "jobs": [
        {
          "job_id": "job_1",
          "job_type": "packaging",
          "source_station_id": "station_1",
          "target_station_id": "station_2",
          "deadline": 150.0,
          "priority": 50,
          "required_capacity": 0.0
        }
      ],
      "stations": [
        {
          "station_id": "station_0",
          "station_type": "assembly",
          "position": {"x": 0.0, "y": 0.0},
          "is_available": true,
          "queued_jobs": []
        },
        {
          "station_id": "station_1",
          "station_type": "quality_check",
          "position": {"x": 50.0, "y": 50.0},
          "is_available": false,
          "queued_jobs": ["job_0"]
        },
        {
          "station_id": "station_2",
          "station_type": "storage",
          "position": {"x": 100.0, "y": 100.0},
          "is_available": true,
          "queued_jobs": []
        }
      ],
      "lanes": null
    }
  ],
  "return_logits": true
}
```

发送推理请求:

```bash
curl -X POST http://localhost:8000/policy/act \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**响应示例**:
```json
{
  "actions": [
    {
      "robot_id": "robot_0",
      "action_type": "idle",
      "assign_job_id": null,
      "target_position": null
    },
    {
      "robot_id": "robot_1",
      "action_type": "assign_job",
      "assign_job_id": "job_1",
      "target_position": null
    }
  ],
  "action_distributions": [
    {
      "robot_id": "robot_0",
      "action_type": "idle",
      "assign_job_id": null,
      "logits": {
        "job_0": 1.234,
        "job_1": 0.567,
        "idle": 2.345
      },
      "confidence": 0.85
    },
    {
      "robot_id": "robot_1",
      "action_type": "assign_job",
      "assign_job_id": "job_1",
      "logits": {
        "job_0": 0.123,
        "job_1": 2.456,
        "idle": 0.789
      },
      "confidence": 0.91
    }
  ],
  "meta": {
    "policy_version": "v1.0",
    "model_device": "cpu",
    "num_robots": 2,
    "num_available_jobs": 1
  }
}
```

🎉 **推理成功！**

---

## 4️⃣ 项目结构

```
policy_service/
├── app.py                           # FastAPI 推理服务 ⭐
├── test_madt.py                     # 单元测试 (6 个)
├── start.py                         # 交互菜单
├── generate_data.py                 # 数据生成
│
├── common/                          # 共享代码
│   ├── schemas.py                   # Pydantic 数据模型
│   └── vectorizer.py                # 向量化
│
├── training/                        # 训练相关
│   ├── model.py                     # Decision Transformer
│   ├── dataset.py                   # 数据加载器
│   └── train.py                     # 训练脚本
│
├── configs/                         # 配置文件
│   └── v1_bc.yaml                   # v1 配置
│
├── data/                            # 数据目录
│   └── episodes/episodes.jsonl      # 示例数据 (20 episodes)
│
├── README.md                        # 详细文档
├── IMPLEMENTATION_SUMMARY.md        # 实现总结
└── QUICKSTART.md                    # 本文件
```

---

## 5️⃣ 下一步

### 生成训练数据

```bash
python generate_data.py 100 ./data/episodes
# 生成 100 个随机 episode 用于训练
```

### 启动训练

```bash
python -m training.train --config configs/v1_bc.yaml
# 执行行为克隆训练，保存最佳模型到 ./checkpoints/
```

### 使用训练后的模型

编辑 `app.py` 中的路径，使用 `./checkpoints/best_model.pt`，然后重启服务。

---

## 🆘 常见问题

**Q: 推理返回全是 idle？**  
A: 这是虚拟模型（随机）的行为。训练后会改进。

**Q: 如何修改 API 端口？**  
A: `uvicorn app:app --port 9000`

**Q: 如何使用 GPU？**  
A: 编辑 `configs/v1_bc.yaml`: `device: "cuda"`

**Q: 如何查看完整文档？**  
A: 打开 `README.md` 或访问 `http://localhost:8000/docs`

---

## 📚 深入学习

- 📖 [README.md](README.md) - 完整使用指南
- 📋 [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - 实现细节
- 🧪 查看单元测试: `cat test_madt.py`
- 🤖 查看模型代码: `cat training/model.py`

---

## 💬 反馈

如有问题或建议，欢迎反馈！

---

**Next**: 阅读 [README.md](README.md) 了解更多功能，或尝试 `python start.py` 使用交互菜单。

🚀 **Happy Scheduling!**
