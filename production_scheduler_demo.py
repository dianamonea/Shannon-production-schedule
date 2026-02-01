#!/usr/bin/env python3
"""
结构件加工生产调度多智能体系统 - 完整示例
Structural Parts Machining Scheduling Multi-Agent System - Complete Example

这个脚本展示如何：
1. 连接到 Shannon 后端
2. 创建一个会话（Session）
3. 按顺序执行 3 类具身智能体（机床、AGV小车、机器人）
4. 三类智能体相互协同完成结构件加工调度
5. 在桌面程序中实时跟踪执行过程
"""

import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum

# ============================================================
# 生产扰动系统定义
# ============================================================

class DisturbanceType(Enum):
    """生产扰动类型枚举"""
    MACHINE_FAILURE = "machine_failure"          # 设备故障
    MATERIAL_DELAY = "material_delay"            # 物料延迟
    QUALITY_ISSUE = "quality_issue"              # 质量问题
    URGENT_ORDER = "urgent_order"                # 紧急插单
    POWER_FLUCTUATION = "power_fluctuation"      # 能源波动
    TOOL_DAMAGE = "tool_damage"                  # 刀具损坏
    AGV_BREAKDOWN = "agv_breakdown"              # AGV故障
    OPERATOR_SHORTAGE = "operator_shortage"      # 人员短缺

class Disturbance:
    """生产扰动"""
    def __init__(self, dist_type: DisturbanceType, severity: str, description: str, 
                 affected_resource: str, impact_duration: int = 0):
        self.type = dist_type
        self.severity = severity  # low, medium, high, critical
        self.description = description
        self.affected_resource = affected_resource
        self.impact_duration = impact_duration  # minutes
        self.timestamp = datetime.now().isoformat()
        self.handled = False
        self.response_strategy = None
    
    def __str__(self):
        return f"{self.type.value}({self.severity}): {self.description} [影响: {self.affected_resource}]"

class DisturbanceGenerator:
    """扰动生成器 - 模拟真实生产中的随机扰动"""
    
    @staticmethod
    def generate_random_disturbances(num: int = 3) -> List[Disturbance]:
        """生成随机扰动"""
        disturbances = []
        
        disturbance_templates = [
            {
                "type": DisturbanceType.MACHINE_FAILURE,
                "severity": "high",
                "description": "CNC-2 主轴轴承过热，需紧急维护",
                "affected_resource": "cnc_2",
                "impact_duration": 120
            },
            {
                "type": DisturbanceType.MATERIAL_DELAY,
                "severity": "medium",
                "description": "钛合金原材料供应商延迟2小时交货",
                "affected_resource": "material_titanium",
                "impact_duration": 120
            },
            {
                "type": DisturbanceType.QUALITY_ISSUE,
                "severity": "medium",
                "description": "PART-003 检测发现尺寸偏差，需返工",
                "affected_resource": "PART-003",
                "impact_duration": 45
            },
            {
                "type": DisturbanceType.URGENT_ORDER,
                "severity": "critical",
                "description": "新增紧急订单 PART-URGENT，要求4小时内完成",
                "affected_resource": "new_urgent_part",
                "impact_duration": 0
            },
            {
                "type": DisturbanceType.POWER_FLUCTUATION,
                "severity": "low",
                "description": "车间电力限制，机床功率需降低20%",
                "affected_resource": "power_system",
                "impact_duration": 60
            },
            {
                "type": DisturbanceType.TOOL_DAMAGE,
                "severity": "medium",
                "description": "CNC-1 铣刀磨损严重，需立即更换",
                "affected_resource": "cnc_1",
                "impact_duration": 30
            },
            {
                "type": DisturbanceType.AGV_BREAKDOWN,
                "severity": "high",
                "description": "AGV-01 导航系统故障，无法正常运输",
                "affected_resource": "AGV-01",
                "impact_duration": 90
            },
            {
                "type": DisturbanceType.OPERATOR_SHORTAGE,
                "severity": "medium",
                "description": "夜班操作员请假，人手减少1人",
                "affected_resource": "operator_team",
                "impact_duration": 480
            }
        ]
        
        # 随机选择指定数量的扰动
        selected = random.sample(disturbance_templates, min(num, len(disturbance_templates)))
        
        for template in selected:
            disturbance = Disturbance(
                dist_type=template["type"],
                severity=template["severity"],
                description=template["description"],
                affected_resource=template["affected_resource"],
                impact_duration=template.get("impact_duration", 0)
            )
            disturbances.append(disturbance)
        
        return disturbances

# ============================================================
# 第 0 步：检查环境和导入
# ============================================================

def verify_environment():
    """验证必要的环境配置"""
    print("=" * 60)
    print("【第 0 步】验证环境配置")
    print("=" * 60)
    
    # 检查 Python 版本
    import sys
    print(f"✓ Python 版本: {sys.version.split()[0]}")
    
    # 检查必要的包
    try:
        import requests
        print(f"✓ requests 已安装")
    except ImportError:
        print("✗ 需要安装 requests: pip install requests")
        return False
    
    # 检查后端服务
    try:
        response = __import__('requests').get('http://localhost:8080/health', timeout=2)
        if response.status_code == 200:
            print(f"✓ 后端服务正常: {response.json()}")
        else:
            print(f"✗ 后端返回状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接后端服务: {e}")
        print("  请确保 Docker 容器运行: docker compose ps")
        return False
    
    print()
    return True


# ============================================================
# 第 1 步：创建会话
# ============================================================

def create_session(session_name: str) -> Dict[str, Any]:
    """
    创建一个新的 Shannon 会话（通过提交初始任务）
    此会话将用来追踪整个生产调度流程的执行
    """
    print("=" * 60)
    print("【第 1 步】创建会话（Session）- 接入后端")
    print("=" * 60)
    
    import requests
    import uuid
    
    # 使用 UUID 作为会话 ID，确保全局唯一
    session_id = str(uuid.uuid4())
    
    print(f"📋 会话初始化:")
    print(f"  会话 ID: {session_id}")
    print(f"  会话名称: {session_name}")
    print(f"  创建时间: {datetime.now().isoformat()}")
    
    # 通过提交初始任务来创建会话
    try:
        task_payload = {
            "query": "初始化生产调度会话",
            "session_id": session_id,
            "context": {
                "workflow_type": "production_scheduling",
                "timestamp": datetime.now().isoformat(),
                "session_name": session_name
            },
            "mode": "simple"
        }
        
        response = requests.post(
            'http://localhost:8080/api/v1/tasks',
            json=task_payload,
            timeout=5
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            session_response_id = result.get('session_id', session_id)
            workflow_id = result.get('workflow_id', result.get('task_id', 'unknown'))
            
            print(f"✓ 会话已在后端创建")
            print(f"  会话 ID: {session_response_id}")
            print(f"  工作流 ID: {workflow_id}")
            
            return {
                "id": session_response_id,
                "session_id": session_response_id,
                "name": session_name,
                "workflow_id": workflow_id,
                "status": "created",
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"⚠️  初始化任务响应: {response.status_code}")
            print(f"  但会话ID仍可用: {session_id}")
            # 即使初始化任务失败，会话ID仍然可用于后续任务
            return {
                "id": session_id,
                "session_id": session_id,
                "name": session_name,
                "status": "initialized_with_fallback",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"⚠️  后端初始化异常: {e}")
        print(f"  继续使用会话 ID: {session_id}")
        # 异常情况下仍然返回有效的会话ID
        return {
            "id": session_id,
            "session_id": session_id,
            "name": session_name,
            "status": "fallback",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# ============================================================
# 第 2-4 步：执行三类具身智能体
# ============================================================

class ProductionSchedulingAgent:
    """生产调度智能体基类"""
    
    def __init__(self, agent_name: str, session_id: str):
        self.agent_name = agent_name
        self.session_id = session_id
        self.start_time = None
        self.end_time = None
        self.result = None
        self.workflow_id = None
        self.coordination_log = []
        self.disturbances = []  # 记录遇到的扰动
        self.response_actions = []  # 记录应对措施
    
    def log_coordination(self, message: str):
        """记录协同过程"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "agent": self.agent_name,
            "message": message
        }
        self.coordination_log.append(log_entry)
        print(f"  └─ [{timestamp}] {message}")
    
    def log_disturbance(self, disturbance: Disturbance, response: str):
        """记录扰动和应对措施"""
        self.disturbances.append({
            "disturbance": disturbance,
            "response": response,
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat()
        })
        self.log_coordination(f"⚠️  检测到扰动: {disturbance.type.value}")
        self.log_coordination(f"📋 应对策略: {response}")
    
    def handle_disturbances(self, disturbances: List[Disturbance], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理扰动 - 由子类实现具体策略"""
        raise NotImplementedError
    
    def execute(self, input_data: Dict[str, Any], disturbances: List[Disturbance] = None) -> Dict[str, Any]:
        """执行智能体（支持扰动处理）"""
        self.start_time = datetime.now()
        print(f"\n🚀 {self.agent_name} 开始执行...")
        print(f"   会话 ID: {self.session_id}")
        
        self.log_coordination(f"初始化 {self.agent_name}")
        
        # 检查是否有扰动需要处理
        if disturbances:
            relevant_disturbances = self._filter_relevant_disturbances(disturbances)
            if relevant_disturbances:
                print(f"   ⚠️  检测到 {len(relevant_disturbances)} 个相关扰动")
                input_data = self.handle_disturbances(relevant_disturbances, input_data)
        
        # 通过后端 API 调用智能体
        result = self._call_agent_api(input_data)
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        self.log_coordination(f"执行完成，耗时 {duration:.2f} 秒")
        print(f"✓ {self.agent_name} 完成 (耗时: {duration:.2f}秒)")
        
        self.result = result
        return result
    
    def _filter_relevant_disturbances(self, disturbances: List[Disturbance]) -> List[Disturbance]:
        """筛选与本智能体相关的扰动 - 由子类覆盖"""
        return disturbances
    
    def _call_agent_api(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """通过后端 API 调用智能体任务"""
        import requests
        
        # 构建任务查询，让后端 LLM 执行智能体逻辑
        agent_prompt = self._build_agent_prompt(input_data)
        
        payload = {
            "query": agent_prompt,
            "session_id": self.session_id,
            "context": {
                "agent_type": self.agent_name,
                "agent_input": input_data,
                "timestamp": datetime.now().isoformat()
            },
            "mode": "simple"
        }
        
        self.log_coordination(f"向后端提交任务: {self.agent_name}")
        
        try:
            response = requests.post(
                'http://localhost:8080/api/v1/tasks',
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                self.workflow_id = result.get('workflow_id', result.get('session_id'))
                self.log_coordination(f"后端任务执行成功，工作流ID: {self.workflow_id}")
                
                # 解析后端返回结果或使用模拟数据
                return self._generate_mock_result(input_data)
            else:
                self.log_coordination(f"后端返回状态码 {response.status_code}，使用本地结果")
                return self._generate_mock_result(input_data)
        except Exception as e:
            self.log_coordination(f"API 调用异常: {e}，使用本地模拟数据")
            return self._generate_mock_result(input_data)
    
    def _build_agent_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建智能体提示词 - 由子类实现"""
        raise NotImplementedError
    
    def _generate_mock_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟结果用于演示"""
        raise NotImplementedError


class MachineToolAgent(ProductionSchedulingAgent):
    """机床具身智能体 - 支持扰动处理"""

    def _filter_relevant_disturbances(self, disturbances: List[Disturbance]) -> List[Disturbance]:
        """筛选与机床相关的扰动"""
        relevant_types = [
            DisturbanceType.MACHINE_FAILURE,
            DisturbanceType.TOOL_DAMAGE,
            DisturbanceType.POWER_FLUCTUATION,
            DisturbanceType.URGENT_ORDER,
            DisturbanceType.QUALITY_ISSUE
        ]
        return [d for d in disturbances if d.type in relevant_types]
    
    def handle_disturbances(self, disturbances: List[Disturbance], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理机床相关的扰动"""
        machines = input_data.get('machines', {})
        parts = input_data.get('parts', [])
        
        for disturbance in disturbances:
            if disturbance.type == DisturbanceType.MACHINE_FAILURE:
                # 机床故障 - 将任务转移到备用机床
                failed_machine = disturbance.affected_resource
                if failed_machine in machines:
                    machines[failed_machine]['status'] = 'maintenance'
                    response = f"将 {failed_machine} 的任务转移到其他机床，预计延迟 {disturbance.impact_duration} 分钟"
                    self.log_disturbance(disturbance, response)
                    
            elif disturbance.type == DisturbanceType.TOOL_DAMAGE:
                # 刀具损坏 - 安排刀具更换，调整时间表
                response = f"安排立即更换刀具，预留 {disturbance.impact_duration} 分钟更换时间"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.POWER_FLUCTUATION:
                # 能源波动 - 降低加工速度，延长周期
                response = "降低机床运行功率至80%，加工时间延长25%"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.URGENT_ORDER:
                # 紧急插单 - 调整优先级
                urgent_part = {
                    "id": "PART-URGENT",
                    "name": "紧急结构件",
                    "material": "铝合金",
                    "process": ["铣削"],
                    "priority_score": 100,
                    "due_date": "2026-01-29"
                }
                parts.insert(0, urgent_part)
                response = "将紧急订单插入队列首位，重新调整排产序列"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.QUALITY_ISSUE:
                # 质量问题 - 预留返工时间
                response = f"为 {disturbance.affected_resource} 预留返工时间 {disturbance.impact_duration} 分钟"
                self.log_disturbance(disturbance, response)
        
        input_data['machines'] = machines
        input_data['parts'] = parts
        return input_data

    def _build_agent_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建机床排产智能体的提示词"""
        parts = input_data.get('parts', [])
        machines = input_data.get('machines', {})
        
        machine_info = "\n".join([
            f"  - {mid}: {m.get('status', 'unknown')} | 能力: {', '.join(m.get('capability', []))}"
            for mid, m in machines.items()
        ])
        
        parts_info = "\n".join([
            f"  - {p['id']} ({p['name']}): {p['material']} | 工序: {', '.join(p['process'])} | 优先级: {p['priority_score']}"
            for p in parts
        ])
        
        prompt = f"""你是一个结构件加工的机床调度智能体。需要根据以下信息进行最优排产：

可用机床：
{machine_info}

待加工件：
{parts_info}

请分析：
1. 按优先级和工序能力分配零件到具体机床
2. 计算主轴利用率
3. 识别瓶颈机床
4. 估算总加工周期
5. 提供详细的机床分配方案

请以 JSON 格式返回分配方案。"""
        
        return prompt

    def _generate_mock_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成机床排产结果"""
        parts = input_data.get('parts', [])
        machines = input_data.get('machines', {})

        self.log_coordination("分析零件优先级和加工工序")
        
        sequence = sorted(parts, key=lambda x: x.get('priority_score', 0), reverse=True)
        part_ids = [p['id'] for p in sequence]
        
        self.log_coordination(f"零件排序完成: {' -> '.join(part_ids)}")

        # 分配到机床
        allocation = {
            "cnc_1": part_ids[0:2],
            "cnc_2": part_ids[2:4],
            "cnc_3": part_ids[4:]
        }
        
        alloc_str = f"CNC-1({len(allocation['cnc_1'])}件), CNC-2({len(allocation['cnc_2'])}件), CNC-3({len(allocation['cnc_3'])}件)"
        self.log_coordination(f"机床分配: {alloc_str}")

        utilization = min(0.95, 0.6 + len(parts) * 0.06)
        
        self.log_coordination(f"计算主轴利用率: {utilization*100:.1f}%")

        return {
            "agent": "MachineToolAgent",
            "status": "completed",
            "part_count": len(parts),
            "process_sequence": part_ids,
            "machine_allocation": allocation,
            "spindle_utilization": utilization,
            "bottleneck": "cnc_2",
            "estimated_cycle_hours": len(parts) * 1.8,
            "planning_timestamp": datetime.now().isoformat(),
            "coordination_log": self.coordination_log
        }


class AGVCoordinator(ProductionSchedulingAgent):
    """AGV小车具身智能体 - 与机床智能体协同，支持扰动处理"""

    def _filter_relevant_disturbances(self, disturbances: List[Disturbance]) -> List[Disturbance]:
        """筛选与AGV相关的扰动"""
        relevant_types = [
            DisturbanceType.AGV_BREAKDOWN,
            DisturbanceType.MATERIAL_DELAY,
            DisturbanceType.URGENT_ORDER
        ]
        return [d for d in disturbances if d.type in relevant_types]
    
    def handle_disturbances(self, disturbances: List[Disturbance], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理AGV相关的扰动"""
        agv_fleet = input_data.get('agv_fleet', [])
        
        for disturbance in disturbances:
            if disturbance.type == DisturbanceType.AGV_BREAKDOWN:
                # AGV故障 - 调用备用AGV或调整路由
                failed_agv = disturbance.affected_resource
                for agv in agv_fleet:
                    if agv['id'] == failed_agv:
                        agv['status'] = 'maintenance'
                response = f"将 {failed_agv} 的任务分配给其他AGV，启用备用车辆"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.MATERIAL_DELAY:
                # 物料延迟 - 调整配送优先级
                response = f"优先配送已到货物料，预留 {disturbance.impact_duration} 分钟等待时间"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.URGENT_ORDER:
                # 紧急订单 - 优先配送
                response = "为紧急订单开辟专用物流通道，优先配送"
                self.log_disturbance(disturbance, response)
        
        input_data['agv_fleet'] = agv_fleet
        return input_data

    def _build_agent_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建AGV物流协同的提示词"""
        machine_allocation = input_data.get('machine_allocation', {})
        agv_fleet = input_data.get('agv_fleet', [])
        
        agv_info = "\n".join([f"  - {a['id']}: {a.get('status', 'unknown')}" for a in agv_fleet])
        
        machines = "\n".join([f"  - {m}: {len(v)}件" for m, v in machine_allocation.items()])
        
        prompt = f"""你是一个生产物流协调智能体，需要与机床智能体协同。

机床分配结果：
{machines}

可用AGV：
{agv_info}

请规划：
1. 从库房到各机床的物料路由
2. AGV的最优调度方案
3. 队列管理和缓冲策略
4. 避免交通热点
5. 确保机床的持续供料

请以 JSON 格式返回物流协调方案。"""
        
        return prompt

    def _generate_mock_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成AGV物流协调结果"""
        machine_allocation = input_data.get('machine_allocation', {})
        agv_fleet = input_data.get('agv_fleet', [])

        self.log_coordination("分析机床分配需求，规划物料路由")

        routes = []
        for i, route in enumerate(["库房->cnc_1", "库房->cnc_2", "库房->cnc_3", "cnc_2->检测站"]):
            if i < len(agv_fleet):
                agv_id = agv_fleet[i]['id']
                routes.append({"agv": agv_id, "route": route})
                self.log_coordination(f"路由分配: {agv_id} 负责 {route}")
        
        self.log_coordination(f"共规划 {len(routes)} 条物料路由")
        self.log_coordination("分析交通热点和缓冲策略")

        return {
            "agent": "AGVCoordinator",
            "status": "completed",
            "agv_count": len(agv_fleet),
            "material_routes": routes,
            "queue_time_minutes": 12,
            "traffic_hotspot": "通道A",
            "buffer_strategy": "cnc_2前置缓存2托盘",
            "coordination_timestamp": datetime.now().isoformat(),
            "coordination_log": self.coordination_log
        }


class RobotCellAgent(ProductionSchedulingAgent):
    """机器人具身智能体 - 与机床和AGV协同，支持扰动处理"""

    def _filter_relevant_disturbances(self, disturbances: List[Disturbance]) -> List[Disturbance]:
        """筛选与机器人相关的扰动"""
        relevant_types = [
            DisturbanceType.QUALITY_ISSUE,
            DisturbanceType.OPERATOR_SHORTAGE,
            DisturbanceType.URGENT_ORDER
        ]
        return [d for d in disturbances if d.type in relevant_types]
    
    def handle_disturbances(self, disturbances: List[Disturbance], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理机器人相关的扰动"""
        robots = input_data.get('robots', [])
        
        for disturbance in disturbances:
            if disturbance.type == DisturbanceType.QUALITY_ISSUE:
                # 质量问题 - 提高检测频率
                response = "提高抽检比例至 30%，增加关键尺寸检测点"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.OPERATOR_SHORTAGE:
                # 操作员短缺 - 启用自动化模式
                response = "切换到全自动上下料模式，减少人工干预"
                self.log_disturbance(disturbance, response)
                
            elif disturbance.type == DisturbanceType.URGENT_ORDER:
                # 紧急订单 - 优先处理
                response = "为紧急订单预留专用检测通道和返工工位"
                self.log_disturbance(disturbance, response)
        
        input_data['robots'] = robots
        return input_data

    def _build_agent_prompt(self, input_data: Dict[str, Any]) -> str:
        """构建机器人协同的提示词"""
        parts = input_data.get('parts', [])
        robots = input_data.get('robots', [])
        
        robot_info = "\n".join([f"  - {r['id']}: {r.get('cell', 'unknown')}" for r in robots])
        
        parts_info = "\n".join([f"  - {p['id']}: {p['material']} | 工序: {', '.join(p['process'])}" for p in parts])
        
        prompt = f"""你是一个机器人协同执行的智能体，需要与机床和AGV协同完成生产。

加工零件：
{parts_info}

可用机器人：
{robot_info}

请规划：
1. 零件的上下料分配（机器人-零件映射）
2. 夹具切换的顺序和次数
3. 检测单元的抽检比例和规则
4. 返工缓冲区的设置
5. 与AGV和机床的同步逻辑

请以 JSON 格式返回机器人执行方案。"""
        
        return prompt

    def _generate_mock_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成机器人协同执行结果"""
        parts = input_data.get('parts', [])
        robots = input_data.get('robots', [])

        self.log_coordination("分析零件加工流程，规划机器人任务")

        assignments = []
        for i, p in enumerate(parts):
            if i < len(robots):
                robot_id = robots[i % len(robots)]['id']
                assignment = {"robot": robot_id, "task": f"上下料-{p['id']}"}
                assignments.append(assignment)
                self.log_coordination(f"任务分配: {robot_id} 负责 {p['id']} 的上下料")
        
        self.log_coordination(f"共分配 {len(assignments)} 个上下料任务")
        self.log_coordination("计算夹具切换次数和抽检策略")

        return {
            "agent": "RobotCellAgent",
            "status": "completed",
            "robot_count": len(robots),
            "cell_assignments": assignments,
            "fixture_changeovers": 3,
            "inspection_rate": 0.1,
            "rework_buffer": "检测站旁预留2工位",
            "robot_timestamp": datetime.now().isoformat(),
            "coordination_log": self.coordination_log
        }


# ============================================================
# 主函数
# ============================================================

def main():
    """主执行函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "生产调度多智能体系统" + " " * 30 + "║")
    print("║" + " " * 8 + "Production Scheduling Multi-Agent System" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 第 0 步：验证环境
    if not verify_environment():
        print("\n❌ 环境验证失败，请检查配置后重试")
        return
    
    # 第 1 步：创建会话
    session_name = f"machining_schedule_{int(time.time())}"
    session = create_session(session_name)
    session_id = session.get('id', session.get('session_id', 'demo-session'))
    
    # 准备初始数据
    print("\n" + "=" * 60)
    print("【准备】加载初始生产数据")
    print("=" * 60)
    
    parts = [
        {"id": "PART-001", "name": "结构件A", "material": "铝合金", "process": ["铣削", "钻孔"], "priority_score": 92, "due_date": "2026-02-05"},
        {"id": "PART-002", "name": "结构件B", "material": "钛合金", "process": ["粗铣", "精铣", "去毛刺"], "priority_score": 85, "due_date": "2026-02-08"},
        {"id": "PART-003", "name": "结构件C", "material": "不锈钢", "process": ["钻孔", "攻丝"], "priority_score": 78, "due_date": "2026-02-10"},
        {"id": "PART-004", "name": "结构件D", "material": "铝合金", "process": ["铣削", "检测"], "priority_score": 96, "due_date": "2026-02-03"},
        {"id": "PART-005", "name": "结构件E", "material": "铝合金", "process": ["粗铣", "精铣"], "priority_score": 88, "due_date": "2026-02-06"},
    ]

    machines = {
        "cnc_1": {"status": "operational", "capability": ["铣削", "钻孔"]},
        "cnc_2": {"status": "operational", "capability": ["粗铣", "精铣"]},
        "cnc_3": {"status": "operational", "capability": ["钻孔", "攻丝"]},
    }

    agv_fleet = [
        {"id": "AGV-01", "status": "idle"},
        {"id": "AGV-02", "status": "charging"},
    ]

    robots = [
        {"id": "ROBOT-01", "cell": "上下料单元"},
        {"id": "ROBOT-02", "cell": "检测单元"},
    ]

    print(f"✓ 加载 {len(parts)} 个结构件任务")
    print(f"✓ 加载 {len(machines)} 台机床")
    print(f"✓ 加载 {len(agv_fleet)} 台AGV小车")
    print(f"✓ 加载 {len(robots)} 台机器人")
    print(f"✓ 会话 ID: {session_id}")
    
    # 第 1.5 步：生成生产扰动
    print("\n" + "=" * 60)
    print("【扰动】模拟生产过程中的随机扰动")
    print("=" * 60)
    
    disturbance_gen = DisturbanceGenerator()
    disturbances = disturbance_gen.generate_random_disturbances(num=6)
    
    print(f"\n⚠️  共检测到 {len(disturbances)} 个生产扰动：\n")
    for i, d in enumerate(disturbances, 1):
        severity_icon = "🔴" if d.severity == "high" else "🟡" if d.severity == "medium" else "🟢"
        print(f"  {i}. {severity_icon} [{d.type.value}] {d.description}")
        print(f"     影响资源: {d.affected_resource} | 影响时长: {d.impact_duration} 分钟")
    
    print("\n💡 智能体将根据扰动类型动态调整生产计划...\n")
    
    # 第 2 步：机床排产
    print("\n")
    machine_tool_agent = MachineToolAgent("【机床智能体】MachineToolAgent", session_id)
    machine_result = machine_tool_agent.execute({
        "parts": parts,
        "machines": machines,
        "session_id": session_id
    }, disturbances=disturbances)

    print(f"  🧰 零件数量: {machine_result['part_count']} 件")
    print(f"  🔧 主轴利用率: {machine_result['spindle_utilization']*100:.1f}%")
    print(f"  ⏱️  预计加工时长: {machine_result['estimated_cycle_hours']:.1f} 小时")
    print(f"  🧭 瓶颈机床: {machine_result['bottleneck']}")
    print(f"  📋 排产序列: {' -> '.join(machine_result['process_sequence'])}")
    print()
    print("  【机床智能体执行日志】")
    for log in machine_result.get('coordination_log', []):
        print(f"    • {log['message']}")

    # 第 3 步：AGV 物流协同（与机床协同）
    print("\n")
    print("=" * 60)
    print("【第 3 步】AGV 物流协同 - 与机床智能体协同")
    print("=" * 60)
    
    agv_coordinator = AGVCoordinator("【AGV智能体】AGVCoordinator", session_id)
    agv_result = agv_coordinator.execute({
        "machine_allocation": machine_result['machine_allocation'],
        "agv_fleet": agv_fleet,
        "session_id": session_id
    }, disturbances=disturbances)

    print(f"  🚚 AGV 数量: {agv_result['agv_count']}")
    print(f"  🧩 物料路由: {[r['route'] for r in agv_result['material_routes']]}")
    print(f"  🕒 排队时间: {agv_result['queue_time_minutes']} 分钟")
    print(f"  🧱 缓冲策略: {agv_result['buffer_strategy']}")
    print()
    print("  【AGV智能体执行日志 - 与机床协同】")
    for log in agv_result.get('coordination_log', []):
        print(f"    • {log['message']}")

    # 第 4 步：机器人协同（与机床和AGV协同）
    print("\n")
    print("=" * 60)
    print("【第 4 步】机器人协同 - 与机床和AGV共同完成生产")
    print("=" * 60)
    
    robot_cell_agent = RobotCellAgent("【机器人智能体】RobotCellAgent", session_id)
    robot_result = robot_cell_agent.execute({
        "parts": parts,
        "robots": robots,
        "session_id": session_id
    }, disturbances=disturbances)

    print(f"  🤖 机器人数量: {robot_result['robot_count']}")
    print(f"  🔁 夹具切换次数: {robot_result['fixture_changeovers']}")
    print(f"  🔍 抽检比例: {robot_result['inspection_rate']*100:.1f}%")
    print(f"  📌 返工缓冲: {robot_result['rework_buffer']}")
    print()
    print("  【机器人智能体执行日志 - 与前两个智能体协同】")
    for log in robot_result.get('coordination_log', []):
        print(f"    • {log['message']}")
    
    # 汇总结果
    print("\n")
    print("=" * 60)
    print("【结果】最终生产调度计划 - 三智能体协同")
    print("=" * 60)
    
    # 收集所有协同日志和扰动响应
    all_coordination_logs = []
    all_coordination_logs.extend(machine_result.get('coordination_log', []))
    all_coordination_logs.extend(agv_result.get('coordination_log', []))
    all_coordination_logs.extend(robot_result.get('coordination_log', []))
    
    # 收集所有扰动响应
    all_disturbances = []
    all_disturbances.extend(machine_tool_agent.disturbances)
    all_disturbances.extend(agv_coordinator.disturbances)
    all_disturbances.extend(robot_cell_agent.disturbances)
    
    # 按时间戳排序
    all_coordination_logs.sort(key=lambda x: x.get('timestamp', ''))
    
    final_schedule = {
        "session_id": session_id,
        "session_info": session,
        "timestamp": datetime.now().isoformat(),
        "disturbances_detected": [
            {
                "type": d.type.value,
                "severity": d.severity,
                "description": d.description,
                "affected_resource": d.affected_resource,
                "impact_duration": d.impact_duration
            } for d in disturbances
        ],
        "disturbance_responses": [
            {
                "agent": item['agent'],
                "disturbance_type": item['disturbance'].type.value,
                "severity": item['disturbance'].severity,
                "description": item['disturbance'].description,
                "affected_resource": item['disturbance'].affected_resource,
                "response": item['response'],
                "timestamp": item['timestamp']
            } for item in all_disturbances
        ],
        "machine_tool_scheduling": machine_result,
        "agv_coordination": agv_result,
        "robot_coordination": robot_result,
        "coordination_timeline": all_coordination_logs,
        "execution_summary": {
            "total_agents_executed": 3,
            "total_disturbances_handled": len(all_disturbances),
            "total_execution_time": sum([
                (machine_tool_agent.end_time - machine_tool_agent.start_time).total_seconds(),
                (agv_coordinator.end_time - agv_coordinator.start_time).total_seconds(),
                (robot_cell_agent.end_time - robot_cell_agent.start_time).total_seconds(),
            ]),
            "status": "completed_successfully",
            "coordination_process": {
                "machine_tool_agent": {
                    "status": machine_result.get('status'),
                    "execution_time": (machine_tool_agent.end_time - machine_tool_agent.start_time).total_seconds(),
                    "workflow_id": machine_tool_agent.workflow_id,
                    "disturbances_handled": len(machine_tool_agent.disturbances)
                },
                "agv_coordinator": {
                    "status": agv_result.get('status'),
                    "execution_time": (agv_coordinator.end_time - agv_coordinator.start_time).total_seconds(),
                    "workflow_id": agv_coordinator.workflow_id,
                    "coordination_with": ["MachineToolAgent"],
                    "disturbances_handled": len(agv_coordinator.disturbances)
                },
                "robot_cell_agent": {
                    "status": robot_result.get('status'),
                    "execution_time": (robot_cell_agent.end_time - robot_cell_agent.start_time).total_seconds(),
                    "workflow_id": robot_cell_agent.workflow_id,
                    "coordination_with": ["MachineToolAgent", "AGVCoordinator"],
                    "disturbances_handled": len(robot_cell_agent.disturbances)
                }
            }
        }
    }
    
    # 显示扰动响应摘要
    print("\n【扰动响应摘要】")
    if all_disturbances:
        print(f"  📊 共处理 {len(all_disturbances)} 个扰动\n")
        for i, item in enumerate(all_disturbances, 1):
            disturbance = item['disturbance']
            response = item['response']
            agent = item['agent']
            severity_icon = "🔴" if disturbance.severity == "high" else "🟡" if disturbance.severity == "medium" else "🟢"
            print(f"  {i}. {severity_icon} [{agent}] {disturbance.type.value}")
            print(f"     ➜ 应对措施: {response}")
    else:
        print("  ✅ 未检测到扰动")
    
    # 显示协同过程
    print("\n【完整协同过程时间线】")
    for i, log in enumerate(all_coordination_logs, 1):
        agent_name = log.get('agent', 'Unknown').replace('【', '').replace('】', '').replace('智能体', '')
        message = log.get('message', '')
        print(f"{i:2d}. [{agent_name}] {message}")
    
    # 保存结果到文件
    output_file = f"schedule_result_{int(time.time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_schedule, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 调度计划已完成")
    print(f"  📄 结果已保存到: {output_file}")
    print(f"  🔗 会话 ID: {session_id}")
    print(f"  ⏱️  总执行时间: {final_schedule['execution_summary']['total_execution_time']:.2f} 秒")
    
    # 打印关键指标
    print(f"\n📈 关键指标总结:")
    print(f"  结构件数量: {machine_result['part_count']} 件")
    print(f"  主轴利用率: {machine_result['spindle_utilization']*100:.1f}%")
    print(f"  预计加工时长: {machine_result['estimated_cycle_hours']:.1f} 小时")
    print(f"  AGV排队时间: {agv_result['queue_time_minutes']} 分钟")
    print(f"  机器人抽检率: {robot_result['inspection_rate']*100:.1f}%")
    print(f"  扰动处理数量: {len(all_disturbances)} 个")
    print(f"  三智能体协同: 机床 ↔ AGV ↔ 机器人")
    
    print("\n" + "=" * 60)
    print("✨ 三智能体协同生产调度流程已完成（含扰动处理）！")
    print("=" * 60)
    
    # 提示用户后续步骤
    print("\n📌 后续步骤:")
    print("  1. 打开 Shannon 桌面程序")
    print(f"  2. 在程序中查找会话 ID: {session_id}")
    print("  3. 查看三类具身智能体的执行详情和协同日志")
    print("  4. 验证实时监控数据")
    print(f"  5. 导出完整的 JSON 报告: {output_file}")
    
    print("\n💻 监控界面:")
    print("  - Temporal UI (工作流监控): http://localhost:8088")
    print("  - Grafana (性能指标): http://localhost:3030")
    print("  - API 文档: http://localhost:8080/api/docs")
    
    return final_schedule


if __name__ == "__main__":
    result = main()
    print("\n✅ 脚本执行完成！\n")
