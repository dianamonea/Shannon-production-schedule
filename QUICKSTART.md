# 快速开始：生产调度多智能体系统

## 📋 概述

这是一个完整的多智能体生产调度系统演示，展示了：
- ✅ 会话接入后端的完整流程
- ✅ 三个具身智能体的顺序执行和协同
- ✅ 所有协同过程的详细追踪

## 🚀 快速启动

### 前置条件
- Docker Desktop 已安装并运行
- Python 3.12+
- requests 库已安装

### 步骤 1：启动后端服务

```bash
cd c:\Users\Administrator\Documents\GitHub\Shannon\deploy\compose
docker compose up -d

# 验证服务已启动（等待30秒左右）
docker compose ps
```

应该看到所有容器都是 "running" 状态。

### 步骤 2：运行演示

```bash
cd c:\Users\Administrator\Documents\GitHub\Shannon

# 设置UTF-8编码
$env:PYTHONIOENCODING='utf-8'

# 运行完整演示
python production_scheduler_demo.py
```

### 步骤 3：验证会话接入

```bash
# 在演示完成后，验证会话是否接入后端
python verify_session_backend.py
```

## 📊 预期输出

### 演示脚本会显示：

1. **环境验证**
```
✓ Python 版本: 3.12.10
✓ requests 已安装
✓ 后端服务正常
```

2. **会话创建**
```
✓ 会话已在后端创建
  会话 ID: 7d4de500-e79d-4d89-9a60-e7f40df5acee
  工作流 ID: task-00000000-0000-0000-0000-000000000002-1769666258
```

3. **三智能体执行**
```
🚀 【机床智能体】MachineToolAgent 开始执行...
  └─ 初始化和各个执行步骤的日志
✓ 完成 (耗时: 0.07秒)

🚀 【AGV智能体】AGVCoordinator 开始执行...
  └─ 初始化和各个执行步骤的日志
✓ 完成 (耗时: 0.11秒)

🚀 【机器人智能体】RobotCellAgent 开始执行...
  └─ 初始化和各个执行步骤的日志
✓ 完成 (耗时: 0.05秒)
```

4. **协同过程时间线**
```
【完整协同过程时间线】
 1. [机床智能体] 初始化 MachineToolAgent
 2. [机床智能体] 向后端提交任务
 3. [机床智能体] 后端任务执行成功
 ... (共26条事件)
26. [机器人智能体] 执行完成，耗时 0.05 秒
```

5. **关键指标总结**
```
📈 关键指标总结:
  结构件数量: 5 件
  主轴利用率: 90.0%
  预计加工时长: 9.0 小时
  AGV排队时间: 12 分钟
  机器人抽检率: 10.0%
```

## 📁 生成的文件

### JSON 结果文件
- **schedule_result_XXXXXXX.json** (12-13KB)
  - 包含会话信息
  - 三个智能体的执行结果
  - 26条协同事件的完整时间线

### 文档
- **PRODUCTION_SCHEDULING_IMPLEMENTATION.md** - 详细的实现文档
- **SESSION_BACKEND_INTEGRATION_SUMMARY.md** - 完成总结
- **QUICKSTART.md** - 本文件（快速开始指南）

## 🔗 监控和调试

### 查看工作流执行
```
浏览器打开: http://localhost:8088 (Temporal UI)
```

### 查看性能指标
```
浏览器打开: http://localhost:3030 (Grafana)
```

### API 文档
```
浏览器打开: http://localhost:8080/api/docs
```

## 📝 查看结果

### 方式 1：查看最新的 JSON 结果
```bash
# PowerShell
$files = Get-ChildItem -Filter "schedule_result*.json" -File | Sort-Object LastWriteTime -Descending
Get-Content $files[0].FullName | ConvertFrom-Json | ConvertTo-Json -Depth 3
```

### 方式 2：使用验证脚本查看
```bash
python verify_session_backend.py
```

这会显示：
- 会话信息
- 三智能体协同情况
- 关键性能指标
- 完整的协同事件时间线

## 🎯 核心特点

### 1. 会话接入后端
```
会话ID通过 POST /api/v1/tasks API 接入后端
后端返回工作流ID用于追踪
```

### 2. 三智能体协同
```
机床智能体 → 输出排产方案
    ↓ (作为输入)
AGV智能体 → 输出物流方案
    ↓ (作为输入)
机器人智能体 → 输出执行计划
```

### 3. 完整追踪
```
26条协同事件按时间戳精确排序
每条事件包含智能体名称、时间、操作描述
```

## ❓ 常见问题

### Q: 脚本运行出错 "后端连接失败"
A: 确保 Docker 容器已启动：
```bash
docker compose ps
docker compose logs gateway
```

### Q: 输出乱码
A: 确保设置了正确的编码：
```bash
$env:PYTHONIOENCODING='utf-8'
```

### Q: 如何停止后端服务？
A: 
```bash
cd deploy/compose
docker compose down
```

### Q: 如何重新运行演示？
A: 只需要再次运行脚本，会自动生成新的会话和结果：
```bash
python production_scheduler_demo.py
```

## 📚 详细文档

- 📖 [实现详解](PRODUCTION_SCHEDULING_IMPLEMENTATION.md) - 详细的代码说明和架构设计
- 📋 [完成总结](SESSION_BACKEND_INTEGRATION_SUMMARY.md) - 整体的实现成果总结
- 🚀 [本文件](QUICKSTART.md) - 快速开始指南

## 🎓 学习要点

通过这个演示，你可以学习到：

1. **会话管理** - 如何在多智能体系统中维持会话连续性
2. **API 集成** - 如何与后端 API 交互（POST /api/v1/tasks）
3. **智能体设计** - 如何设计和实现多个有关联的智能体
4. **结果追踪** - 如何记录和追踪复杂的多步骤执行过程
5. **JSON 输出** - 如何组织和保存复杂的结构化数据

## 💡 后续扩展

可以基于这个系统进行的扩展：

1. **多轮协同** - 支持智能体之间的多轮对话和迭代优化
2. **动态调度** - 根据实时反馈动态调整计划
3. **异常处理** - 当执行失败时的自动重调度
4. **可视化** - 在桌面应用中实时展示协同过程
5. **真实集成** - 与实际的 CNC、AGV、机器人系统集成

## 📞 支持

如有问题，可以：
1. 查看详细的实现文档
2. 检查 Docker 日志：`docker compose logs`
3. 查看生成的 JSON 结果文件的内容

---

**开始体验吧！** 🎉

```bash
python production_scheduler_demo.py
```
