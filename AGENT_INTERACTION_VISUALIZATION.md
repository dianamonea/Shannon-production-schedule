# 多智能体交互可视化 - 使用指南

## 📊 概述

Shannon 生产调度系统现已支持**网页和桌面应用**中的多智能体交互可视化。通过实时展示生产扰动、智能体响应和协同过程，让用户直观理解三类具身智能体（机床、AGV、机器人）的协作机制。

---

## 🌐 网页版使用

### 快速启动

```bash
cd c:\Users\Administrator\Documents\GitHub\Shannon
python visualization-server.py localhost 8888
```

然后打开浏览器访问：**http://localhost:8888**

### 功能特性

#### 1. 智能体状态监控
- **机床智能体** (⚙️)：显示 CNC-1/2/3 的处理扰动数和利用率
- **AGV智能体** (📦)：显示 AGV-01/02 的任务分配和路由优化
- **机器人智能体** (⚡)：显示 ROBOT-01/02 的上下料和检测任务

每个智能体卡片实时显示：
- 处理的扰动数量
- 设备利用率
- 运行状态

#### 2. 扰动检测展示
显示所有检测到的生产扰动，包括：
- 扰动类型（6+种）
- 严重程度（🔴高、🟡中、🟢低、🟠紧急）
- 详细描述
- 影响资源和时长

```
扰动示例：
  🔴 [machine_failure] CNC-2 主轴轴承过热，需紧急维护
     📍 资源: cnc_2
     ⏱️ 影响: 120分钟
```

#### 3. 交互流程可视化
时间线展示智能体的动态响应过程：

```
⚙️ 机床智能体 14:27:43
   ⚠️ 扰动: CNC-2 主轴轴承过热，需紧急维护
   ✅ 应对: 将 cnc_2 的任务转移到其他机床，预计延迟 120 分钟

📦 AGV智能体 14:27:43
   ⚠️ 扰动: AGV-01 导航系统故障，无法正常运输
   ✅ 应对: 将 AGV-01 的任务分配给其他AGV，启用备用车辆

⚡ 机器人智能体 14:27:43
   ⚠️ 扰动: 夜班操作员请假，人手减少1人
   ✅ 应对: 切换到全自动上下料模式，减少人工干预
```

#### 4. 协同统计仪表板
实时显示：
- 机床处理的扰动数
- AGV处理的扰动数
- 机器人处理的扰动数
- 总扰动数

#### 5. 数据刷新
- **手动刷新**：点击"🔄 刷新"按钮
- **自动刷新**：启用自动刷新，每2秒更新一次

---

## 🖥️ 桌面应用版使用

### 访问方式

在 Shannon 桌面应用中：
1. 点击左侧菜单 "Agent Interaction" 🔄
2. 或直接导航到 `/agent-interaction` 页面

### 功能

与网页版相同，但集成在 Shannon 应用内部，支持：
- 与会话数据联动
- 本地数据源读取
- 深色/浅色主题切换
- 响应式设计（移动/桌面）

### 集成点

**文件位置**：
- `desktop/app/(app)/agent-interaction/page.tsx` - 主页面组件
- `desktop/app/api/latest-schedule/route.ts` - API 端点

**数据流**：
```
JSON 调度结果文件
    ↓
后端 API (/api/latest-schedule)
    ↓
桌面应用 (Agent Interaction 页面)
    ↓
实时可视化
```

---

## 📡 API 接口

### GET /api/latest-schedule

获取最新的生产调度结果和扰动数据。

**请求**：
```bash
curl http://localhost:8888/api/latest-schedule
```

**响应示例**：
```json
{
  "session_id": "0f90d3ed-f827-4bd0-91be-3d8242900bbe",
  "disturbances_detected": [
    {
      "type": "urgent_order",
      "severity": "critical",
      "description": "新增紧急订单 PART-URGENT，4小时内完成",
      "affected_resource": "new_urgent_part",
      "impact_duration": 0
    }
  ],
  "disturbance_responses": [
    {
      "agent": "【机床智能体】MachineToolAgent",
      "disturbance_type": "urgent_order",
      "severity": "critical",
      "description": "新增紧急订单 PART-URGENT，4小时内完成",
      "affected_resource": "new_urgent_part",
      "response": "将紧急订单插入队列首位，重新调整排产序列",
      "timestamp": "2026-01-29T14:27:43.885363"
    }
  ],
  "execution_summary": {
    "total_agents_executed": 3,
    "total_disturbances_handled": 9,
    "total_execution_time": 0.16
  }
}
```

---

## 🎯 使用场景

### 场景 1：监控生产过程

运行生产调度后，立即在网页版本中查看智能体的响应：

```bash
# 终端1：启动演示
python production_scheduler_demo.py

# 终端2：启动可视化服务
python visualization-server.py localhost 8888

# 浏览器：打开 http://localhost:8888
```

### 场景 2：分析扰动应对

检查特定扰动如何被智能体处理：
1. 在扰动列表中定位扰动
2. 在交互流程中查看所有智能体的响应
3. 分析响应策略的有效性

### 场景 3：调试智能体逻辑

验证智能体的扰动过滤和应对逻辑：
- 确认智能体只处理相关扰动
- 检查应对措施是否合理
- 跟踪执行时间和效率

### 场景 4：报告和演示

导出可视化数据用于报告：
- 截图展示交互流程
- 导出 JSON 数据进行分析
- 展示扰动处理能力

---

## 🔧 自定义配置

### 修改服务器端口

```bash
python visualization-server.py localhost 9999  # 使用端口 9999
```

### 修改数据源路径

编辑 `visualization-server.py`：

```python
# 修改这一行来改变查找调度结果的目录
schedule_files = sorted(
    glob.glob(str(current_dir / 'schedule_result_*.json')),
    ...
)
```

### 自定义 HTML 样式

编辑 `agent-interaction-visualization.html`，修改 Tailwind CSS 类或添加自定义 CSS。

---

## 📊 数据结构

### 扰动数据格式

```typescript
interface Disturbance {
  type: string;           // "machine_failure" | "agv_breakdown" | ...
  severity: string;       // "low" | "medium" | "high" | "critical"
  description: string;    // 详细描述
  affected_resource: string;  // 受影响的资源
  impact_duration: number;    // 影响时长（分钟）
}
```

### 应对数据格式

```typescript
interface DisturbanceResponse {
  agent: string;              // 智能体名称
  disturbance_type: string;   // 扰动类型
  severity: string;           // 严重程度
  description: string;        // 扰动描述
  affected_resource: string;  // 受影响资源
  response: string;           // 应对措施
  timestamp: string;          // ISO 时间戳
}
```

---

## 🚀 性能优化

### 大数据集处理

如果扰动和响应数据很大：

1. **分页显示**：修改 HTML 实现分页加载
2. **数据过滤**：按智能体或严重程度过滤
3. **虚拟滚动**：使用虚拟滚动库加载大列表

### 内存管理

```javascript
// 只保留最新的 100 条记录
const recentInteractions = data.disturbance_responses.slice(-100);
```

---

## 🐛 故障排除

### 问题：无法加载数据

**可能原因**：
- 调度结果文件不存在
- 服务器未启动
- 文件路径不正确

**解决方案**：
1. 确保运行了 `production_scheduler_demo.py`
2. 检查 `schedule_result_*.json` 文件是否存在
3. 查看浏览器控制台错误信息

### 问题：实时更新不工作

**可能原因**：
- 自动刷新未启用
- API 端点返回缓存数据

**解决方案**：
1. 手动点击"刷新"按钮
2. 清除浏览器缓存
3. 重启服务器

### 问题：样式显示异常

**可能原因**：
- Tailwind CSS 加载失败
- 浏览器兼容性问题

**解决方案**：
1. 检查网络连接
2. 尝试不同浏览器
3. 清除浏览器缓存并硬刷新 (Ctrl+Shift+R)

---

## 🔗 相关文档

- [生产调度演示](QUICKSTART.md)
- [扰动处理系统](DISTURBANCE_HANDLING_GUIDE.md)
- [后端集成](SESSION_BACKEND_INTEGRATION_SUMMARY.md)
- [技术架构](TECHNICAL_IMPLEMENTATION_DETAILS.md)

---

## 📝 更新日志

### v1.0.0 (2026-01-29)
- ✅ 网页版可视化页面
- ✅ 桌面应用集成页面
- ✅ 实时数据 API
- ✅ 响应式设计
- ✅ 自动刷新功能
- ✅ 扰动和应对展示
- ✅ 智能体统计仪表板

---

**最后更新**: 2026-01-29  
**维护者**: Shannon 多智能体系统团队
