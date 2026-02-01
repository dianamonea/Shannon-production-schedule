# 生产扰动处理系统使用指南

## 概述

生产调度演示系统已集成**扰动处理机制**，能够模拟和应对真实制造环境中的8种常见生产扰动。三类具身智能体（机床、AGV、机器人）可根据扰动类型动态调整调度计划。

---

## 支持的扰动类型

系统实现了 **8 种扰动类型**（超过需求的6种）：

| 扰动类型 | 严重程度 | 描述 | 影响时长 |
|---------|---------|------|---------|
| `MACHINE_FAILURE` | 🔴 高 | CNC-2 主轴轴承过热，需紧急维护 | 120 分钟 |
| `MATERIAL_DELAY` | 🟡 中 | 钛合金原材料供应商延迟交货 | 120 分钟 |
| `QUALITY_ISSUE` | 🟡 中 | PART-003 检测发现尺寸偏差，需返工 | 45 分钟 |
| `URGENT_ORDER` | 🟢 紧急 | 新增紧急订单 PART-URGENT，4小时内完成 | 0 分钟 |
| `POWER_FLUCTUATION` | 🟢 低 | 车间电力限制，机床功率需降低20% | 60 分钟 |
| `TOOL_DAMAGE` | 🟡 中 | CNC-1 铣刀磨损严重，需立即更换 | 30 分钟 |
| `AGV_BREAKDOWN` | 🔴 高 | AGV-01 导航系统故障，无法正常运输 | 90 分钟 |
| `OPERATOR_SHORTAGE` | 🟡 中 | 夜班操作员请假，人手减少1人 | 480 分钟 |

---

## 智能体应对策略

### 1. 机床智能体 (MachineToolAgent)

**处理的扰动类型**：5 种
- `MACHINE_FAILURE` → 将故障机床的任务转移到其他机床
- `TOOL_DAMAGE` → 预留刀具更换时间（30分钟）
- `POWER_FLUCTUATION` → 降低功率至80%，延长加工时间25%
- `URGENT_ORDER` → 插入紧急订单至队列首位，重新排序
- `QUALITY_ISSUE` → 为不合格零件预留返工时间

**示例输出**：
```
⚠️  检测到扰动: machine_failure
📋 应对策略: 将 cnc_2 的任务转移到其他机床，预计延迟 120 分钟
```

### 2. AGV智能体 (AGVCoordinator)

**处理的扰动类型**：3 种
- `AGV_BREAKDOWN` → 将故障AGV的任务分配给备用车辆
- `MATERIAL_DELAY` → 优先配送已到货物料，预留等待时间
- `URGENT_ORDER` → 开辟专用物流通道，优先配送

**示例输出**：
```
⚠️  检测到扰动: agv_breakdown
📋 应对策略: 将 AGV-01 的任务分配给其他AGV，启用备用车辆
```

### 3. 机器人智能体 (RobotCellAgent)

**处理的扰动类型**：3 种
- `QUALITY_ISSUE` → 提高抽检比例至30%，增加检测点
- `OPERATOR_SHORTAGE` → 切换到全自动上下料模式
- `URGENT_ORDER` → 预留专用检测通道和返工工位

**示例输出**：
```
⚠️  检测到扰动: operator_shortage
📋 应对策略: 切换到全自动上下料模式，减少人工干预
```

---

## 运行演示

```bash
cd c:\Users\Administrator\Documents\GitHub\Shannon
python production_scheduler_demo.py
```

### 演示流程

1. **扰动生成阶段**：
   - 系统随机生成 6 个生产扰动
   - 显示扰动类型、严重程度、影响资源和时长

2. **智能体响应阶段**：
   - 每个智能体自动筛选相关扰动
   - 根据扰动类型执行应对策略
   - 记录扰动处理日志

3. **结果保存**：
   - 生成 JSON 文件（`schedule_result_<timestamp>.json`）
   - 包含完整的扰动列表和响应记录

---

## 输出示例

### 控制台输出

```
============================================================
【扰动】模拟生产过程中的随机扰动
============================================================

⚠️  共检测到 6 个生产扰动：

  1. 🟡 [operator_shortage] 夜班操作员请假，人手减少1人
     影响资源: operator_team | 影响时长: 480 分钟
  2. 🟡 [quality_issue] PART-003 检测发现尺寸偏差，需返工
     影响资源: PART-003 | 影响时长: 45 分钟
  3. 🔴 [machine_failure] CNC-2 主轴轴承过热，需紧急维护
     影响资源: cnc_2 | 影响时长: 120 分钟

💡 智能体将根据扰动类型动态调整生产计划...
```

### 扰动响应摘要

```
【扰动响应摘要】
  📊 共处理 9 个扰动

  1. 🟡 [【机床智能体】MachineToolAgent] quality_issue
     ➜ 应对措施: 为 PART-003 预留返工时间 45 分钟
  2. 🔴 [【机床智能体】MachineToolAgent] machine_failure
     ➜ 应对措施: 将 cnc_2 的任务转移到其他机床，预计延迟 120 分钟
  3. 🟢 [【AGV智能体】AGVCoordinator] urgent_order
     ➜ 应对措施: 为紧急订单开辟专用物流通道，优先配送
```

### JSON 输出结构

```json
{
  "disturbances_detected": [
    {
      "type": "machine_failure",
      "severity": "high",
      "description": "CNC-2 主轴轴承过热，需紧急维护",
      "affected_resource": "cnc_2",
      "impact_duration": 120
    }
  ],
  "disturbance_responses": [
    {
      "agent": "【机床智能体】MachineToolAgent",
      "disturbance_type": "machine_failure",
      "severity": "high",
      "description": "CNC-2 主轴轴承过热，需紧急维护",
      "affected_resource": "cnc_2",
      "response": "将 cnc_2 的任务转移到其他机床，预计延迟 120 分钟",
      "timestamp": "2026-01-29T14:27:43.763076"
    }
  ]
}
```

---

## 关键指标

演示系统会显示扰动处理对生产的影响：

```
📈 关键指标总结:
  结构件数量: 6 件          ← 紧急订单增加了1件
  主轴利用率: 95.0%         ← 机床故障后重新分配提高利用率
  预计加工时长: 10.8 小时   ← 扰动导致时间延长
  扰动处理数量: 9 个         ← 三个智能体共处理9个扰动实例
  三智能体协同: 机床 ↔ AGV ↔ 机器人
```

---

## 技术实现

### 数据结构

```python
class DisturbanceType(Enum):
    MACHINE_FAILURE = "machine_failure"
    MATERIAL_DELAY = "material_delay"
    QUALITY_ISSUE = "quality_issue"
    URGENT_ORDER = "urgent_order"
    POWER_FLUCTUATION = "power_fluctuation"
    TOOL_DAMAGE = "tool_damage"
    AGV_BREAKDOWN = "agv_breakdown"
    OPERATOR_SHORTAGE = "operator_shortage"

@dataclass
class Disturbance:
    type: DisturbanceType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_resource: str
    impact_duration: int  # 分钟
```

### 扰动生成器

```python
disturbance_gen = DisturbanceGenerator()
disturbances = disturbance_gen.generate_random_disturbances(num=6)
```

### 智能体处理流程

```python
# 1. 筛选相关扰动
relevant_disturbances = agent._filter_relevant_disturbances(disturbances)

# 2. 执行应对策略
for disturbance in relevant_disturbances:
    if disturbance.type == DisturbanceType.MACHINE_FAILURE:
        # 执行具体应对措施
        response = "将故障机床任务转移..."
        agent.log_disturbance(disturbance, response)

# 3. 更新调度计划
input_data = agent.handle_disturbances(relevant_disturbances, input_data)
```

---

## 扩展建议

1. **增加扰动类型**：
   - 环境因素（温度、湿度）
   - 供应链中断（多级供应商延迟）
   - 网络故障（MES系统中断）

2. **增强应对策略**：
   - 机器学习预测扰动发生概率
   - 基于历史数据优化应对措施
   - 多智能体协商机制

3. **可视化增强**：
   - Grafana 仪表盘显示扰动趋势
   - 实时扰动告警
   - 应对措施效果评估

---

## 参考文件

- **演示脚本**：[production_scheduler_demo.py](production_scheduler_demo.py)
- **快速开始**：[QUICKSTART.md](QUICKSTART.md)
- **实现细节**：[PRODUCTION_SCHEDULING_IMPLEMENTATION.md](PRODUCTION_SCHEDULING_IMPLEMENTATION.md)
- **会话集成**：[SESSION_BACKEND_INTEGRATION_SUMMARY.md](SESSION_BACKEND_INTEGRATION_SUMMARY.md)

---

## 常见问题

**Q: 为什么同一个扰动会被多个智能体处理？**  
A: 某些扰动（如紧急订单）需要多个智能体协同应对。机床调整排产序列，AGV开辟专用通道，机器人预留检测工位。

**Q: 扰动是如何生成的？**  
A: 系统从8个预定义模板中随机选择6个，每次运行扰动类型和顺序可能不同。

**Q: 如何查看扰动处理的详细过程？**  
A: 查看生成的 JSON 文件中的 `coordination_timeline` 字段，包含完整的44步协同过程。

---

**最后更新**: 2026-01-29  
**版本**: 1.0.0  
**作者**: Shannon 多智能体系统团队
