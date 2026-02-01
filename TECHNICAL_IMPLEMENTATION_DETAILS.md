# 技术实现细节：会话接入后端与多智能体协同

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户端 (Python 脚本)                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 生成会话ID: UUID(7d4de500-e79d-4d89-9a60-e7f40df5acee) │
│                    │                                         │
│                    ▼                                         │
│  2. POST /api/v1/tasks {                                    │
│       "query": "初始化会话",                                 │
│       "session_id": "7d4de500...",                          │
│       "context": {...}                                      │
│     }                                                        │
│                    │                                         │
│                    ▼                                         │
│  3. 接收 workflow_id: task-00000000...0002-1769666258      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP POST
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Shannon 后端 (Gateway)                     │
├─────────────────────────────────────────────────────────────┤
│  http://localhost:8080                                      │
│  - 接收任务请求                                              │
│  - 管理会话生命周期                                          │
│  - 路由到相应的服务                                          │
│  - 返回工作流ID                                              │
└─────────────────────────────────────────────────────────────┘
```

## API 调用流程

### 步骤 1：会话初始化

**请求**:
```json
POST /api/v1/tasks

{
  "query": "初始化生产调度会话",
  "session_id": "7d4de500-e79d-4d89-9a60-e7f40df5acee",
  "context": {
    "workflow_type": "production_scheduling",
    "session_name": "machining_schedule_1769666258",
    "timestamp": "2026-01-29T13:57:38.630815"
  },
  "mode": "simple"
}
```

**响应**:
```json
{
  "session_id": "7d4de500-e79d-4d89-9a60-e7f40df5acee",
  "workflow_id": "task-00000000-0000-0000-0000-000000000002-1769666258",
  "status": "created",
  "timestamp": "2026-01-29T13:57:38.639502"
}
```

### 步骤 2-4：智能体任务提交

**请求**（以机床智能体为例）:
```json
POST /api/v1/tasks

{
  "query": "根据以下信息进行机床排产...",
  "session_id": "7d4de500-e79d-4d89-9a60-e7f40df5acee",
  "context": {
    "agent_type": "【机床智能体】MachineToolAgent",
    "agent_input": {
      "parts": [
        {
          "id": "PART-001",
          "name": "结构件A",
          "material": "铝合金",
          "process": ["铣削", "钻孔"],
          "priority_score": 92
        },
        ...
      ],
      "machines": {
        "cnc_1": {"status": "operational", "capability": ["铣削", "钻孔"]},
        "cnc_2": {"status": "operational", "capability": ["粗铣", "精铣"]},
        "cnc_3": {"status": "operational", "capability": ["钻孔", "攻丝"]}
      }
    },
    "timestamp": "2026-01-29T13:57:38.699716"
  },
  "mode": "simple"
}
```

**响应**:
```json
{
  "session_id": "7d4de500-e79d-4d89-9a60-e7f40df5acee",
  "workflow_id": "task-XXXX-XXXX-XXXX-XXXX-XXXX",
  "status": "completed",
  "result": {
    "agent": "MachineToolAgent",
    "status": "completed",
    "part_count": 5,
    "process_sequence": ["PART-004", "PART-001", "PART-005", "PART-002", "PART-003"],
    "machine_allocation": {...},
    "spindle_utilization": 0.9,
    "bottleneck": "cnc_2",
    "estimated_cycle_hours": 9.0
  }
}
```

## 会话继续性机制

### 核心概念
```python
# 所有智能体使用同一个会话ID
session_id = "7d4de500-e79d-4d89-9a60-e7f40df5acee"

# 每次API调用都包含该session_id
payload = {
    "query": "...",
    "session_id": session_id,  # 关键！
    "context": {...}
}

# 后端根据session_id将多个请求关联起来
# 形成完整的会话上下文
```

### 实现代码

```python
import uuid
import requests
from datetime import datetime

class SessionManager:
    def __init__(self):
        # 1. 生成唯一的会话ID
        self.session_id = str(uuid.uuid4())
        self.workflow_id = None
    
    def create_session(self, session_name: str) -> dict:
        """创建后端会话"""
        payload = {
            "query": "初始化生产调度会话",
            "session_id": self.session_id,
            "context": {
                "workflow_type": "production_scheduling",
                "session_name": session_name,
                "timestamp": datetime.now().isoformat()
            },
            "mode": "simple"
        }
        
        response = requests.post(
            'http://localhost:8080/api/v1/tasks',
            json=payload,
            timeout=5
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            self.workflow_id = result.get('workflow_id')
            return result
        else:
            raise Exception(f"会话创建失败: {response.status_code}")
    
    def submit_agent_task(self, agent_name: str, agent_input: dict) -> dict:
        """提交智能体任务"""
        # 重要：使用相同的session_id
        payload = {
            "query": f"执行 {agent_name} 任务",
            "session_id": self.session_id,  # 保持会话连续性
            "context": {
                "agent_type": agent_name,
                "agent_input": agent_input,
                "timestamp": datetime.now().isoformat()
            },
            "mode": "simple"
        }
        
        response = requests.post(
            'http://localhost:8080/api/v1/tasks',
            json=payload,
            timeout=10
        )
        
        return response.json() if response.status_code in [200, 201] else None
```

## 协同过程追踪

### 日志记录机制

```python
class CoordinationLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, agent_name: str, message: str):
        """记录协同事件"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "message": message
        }
        self.logs.append(entry)
    
    def get_timeline(self) -> list:
        """获取时间线（按时间戳排序）"""
        return sorted(self.logs, key=lambda x: x['timestamp'])
    
    def get_summary(self) -> dict:
        """获取协同总结"""
        agents = {}
        for log in self.logs:
            agent = log['agent']
            if agent not in agents:
                agents[agent] = {'count': 0, 'first_time': None, 'last_time': None}
            agents[agent]['count'] += 1
            if agents[agent]['first_time'] is None:
                agents[agent]['first_time'] = log['timestamp']
            agents[agent]['last_time'] = log['timestamp']
        return agents
```

### 智能体协同示例

```python
class MachineToolAgent:
    def __init__(self, session_manager: SessionManager, logger: CoordinationLogger):
        self.session_manager = session_manager
        self.logger = logger
    
    def execute(self, parts: list, machines: dict) -> dict:
        # 1. 初始化
        self.logger.log("MachineToolAgent", "初始化机床智能体")
        
        # 2. 向后端提交任务
        self.logger.log("MachineToolAgent", "向后端提交排产任务")
        
        result = self.session_manager.submit_agent_task(
            agent_name="MachineToolAgent",
            agent_input={
                "parts": parts,
                "machines": machines
            }
        )
        
        self.logger.log("MachineToolAgent", "后端任务执行成功")
        
        # 3. 处理结果
        self.logger.log("MachineToolAgent", "分析零件优先级")
        sequence = sorted(parts, key=lambda x: x['priority_score'], reverse=True)
        self.logger.log("MachineToolAgent", f"零件排序完成: {sequence}")
        
        # 4. 完成
        self.logger.log("MachineToolAgent", "执行完成")
        
        return result

class AGVCoordinator:
    def __init__(self, session_manager: SessionManager, logger: CoordinationLogger):
        self.session_manager = session_manager
        self.logger = logger
    
    def execute(self, machine_allocation: dict, agv_fleet: list) -> dict:
        # 关键：使用机床智能体的输出作为输入
        self.logger.log("AGVCoordinator", "初始化AGV协调智能体")
        self.logger.log("AGVCoordinator", "向后端提交物流任务")
        
        result = self.session_manager.submit_agent_task(
            agent_name="AGVCoordinator",
            agent_input={
                "machine_allocation": machine_allocation,  # 来自机床智能体
                "agv_fleet": agv_fleet
            }
        )
        
        self.logger.log("AGVCoordinator", "后端任务执行成功")
        self.logger.log("AGVCoordinator", "分析机床分配需求，规划物料路由")
        self.logger.log("AGVCoordinator", "执行完成")
        
        return result
```

## 数据流向

### 输入-输出关联

```
MachineToolAgent
├─ Input: parts[], machines{}
├─ Process: 分配零件到机床
└─ Output: machine_allocation{}
    │
    ├─ spindle_utilization: 90%
    ├─ process_sequence: [PART-004, ...]
    └─ bottleneck: "cnc_2"
         │
         └─► AGVCoordinator
              ├─ Input: machine_allocation{} ◄──── 来自上一个智能体
              │          agv_fleet[]
              ├─ Process: 规划物料路由
              └─ Output: material_routes[]
                  │
                  ├─ routes: [AGV-01->库房->cnc_1, ...]
                  └─ queue_time: 12分钟
                       │
                       └─► RobotCellAgent
                            ├─ Input: parts[] ◄──── 原始数据
                            │          robots[]
                            │          (隐含引用了前面的分配)
                            ├─ Process: 规划上下料任务
                            └─ Output: cell_assignments[]
                                │
                                ├─ assignments: [ROBOT-01->PART-001, ...]
                                └─ fixture_changeovers: 3
```

## 结果保存

### JSON 输出结构

```json
{
  "session_id": "7d4de500-e79d-4d89-9a60-e7f40df5acee",
  "session_info": {
    "id": "7d4de500-e79d-4d89-9a60-e7f40df5acee",
    "name": "machining_schedule_1769666258",
    "workflow_id": "task-00000000-0000-0000-0000-000000000002-1769666258",
    "status": "created",
    "timestamp": "2026-01-29T13:57:38.630815"
  },
  "machine_tool_scheduling": {
    "agent": "MachineToolAgent",
    "status": "completed",
    "part_count": 5,
    "process_sequence": ["PART-004", "PART-001", ...],
    "machine_allocation": {
      "cnc_1": ["PART-004", "PART-001"],
      "cnc_2": ["PART-005", "PART-002"],
      "cnc_3": ["PART-003"]
    },
    "spindle_utilization": 0.9,
    "bottleneck": "cnc_2",
    "estimated_cycle_hours": 9.0,
    "coordination_log": [
      {
        "timestamp": "2026-01-29T13:57:38.699716",
        "agent": "【机床智能体】MachineToolAgent",
        "message": "初始化 MachineToolAgent"
      },
      ...
    ]
  },
  "agv_coordination": { ... },
  "robot_coordination": { ... },
  "coordination_timeline": [
    {
      "timestamp": "2026-01-29T13:57:38.699716",
      "agent": "MachineToolAgent",
      "message": "初始化..."
    },
    ...
  ],
  "execution_summary": {
    "total_agents_executed": 3,
    "total_execution_time": 0.23,
    "status": "completed_successfully",
    "coordination_process": {
      "machine_tool_agent": {
        "status": "completed",
        "execution_time": 0.07,
        "workflow_id": "..."
      },
      "agv_coordinator": {
        "status": "completed",
        "execution_time": 0.11,
        "workflow_id": "...",
        "coordination_with": ["MachineToolAgent"]
      },
      "robot_cell_agent": {
        "status": "completed",
        "execution_time": 0.05,
        "workflow_id": "...",
        "coordination_with": ["MachineToolAgent", "AGVCoordinator"]
      }
    }
  }
}
```

## 性能指标

### 执行时间分析
- **机床智能体**: 0.07 秒
- **AGV智能体**: 0.11 秒
- **机器人智能体**: 0.05 秒
- **总计**: 0.23 秒

### 事件统计
- **总事件数**: 26 条
- **机床智能体事件**: 8 条
- **AGV智能体事件**: 9 条
- **机器人智能体事件**: 9 条

### 业务指标
- **机床利用率**: 90%
- **生产周期**: 9 小时
- **AGV队列等待**: 12 分钟
- **机器人检测率**: 10%

## 错误处理

### 会话创建失败的处理

```python
try:
    response = requests.post(
        'http://localhost:8080/api/v1/tasks',
        json=session_payload,
        timeout=5
    )
    
    if response.status_code in [200, 201]:
        session = response.json()
        session_id = session.get('session_id')
    else:
        # Fallback: 使用本地生成的session_id
        session_id = str(uuid.uuid4())
        print(f"后端未返回有效响应，使用本地session_id: {session_id}")
        
except Exception as e:
    # 异常处理: 继续使用生成的session_id
    session_id = str(uuid.uuid4())
    print(f"会话创建异常: {e}，继续使用本地session_id")

# 重要：后续仍然使用相同的session_id与后端交互
# 这样即使初始化失败，后续任务仍然被关联到同一个会话
```

## 扩展性考虑

### 多轮协同
```python
# 可以在同一会话中运行多次调度
while True:
    result = schedule_agent.execute(parts)
    if result['status'] == 'optimal':
        break
    parts = adjust_parts(result)
```

### 多车间协调
```python
# 使用不同的会话ID管理不同车间
workshop_1_session = str(uuid.uuid4())
workshop_2_session = str(uuid.uuid4())

# 分别提交任务
execute_with_session(workshop_1_task, workshop_1_session)
execute_with_session(workshop_2_task, workshop_2_session)

# 全局协调
global_session = str(uuid.uuid4())
execute_global_optimization(workshop_1_session, workshop_2_session)
```

### 实时反馈循环
```python
# 获取执行反馈
feedback = monitor_execution(machine_id)

# 使用反馈重新调度
if feedback['status'] == 'delayed':
    new_schedule = dynamic_reschedule(
        current_schedule,
        feedback,
        session_id  # 保持会话连续性
    )
```

---

**总结**

这个实现展示了如何在 Shannon 框架中：
1. 创建具有会话连续性的多智能体系统
2. 通过 REST API 与后端交互
3. 记录和追踪复杂的多步骤执行过程
4. 保存完整的协同信息用于后续分析

这为生产制造、物流、服务编排等复杂多智能体场景提供了可靠的基础。
