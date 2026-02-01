"""
扩散式 MARL 与现有生产调度系统的集成示例
Integration Example: Diffusion Policy MARL with Existing System

这个文件展示如何将扩散式多智能体策略集成到 production_scheduler_demo.py 中。
"""

import sys
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

# 导入现有系统
# from production_scheduler_demo import ProductionSchedulingAgent, State

# 导入扩散式 MARL
from diffusion_marl import (
    DiffusionConfig,
    DiffusionScheduler,
    DiffusionMachineToolAgent,
    DiffusionAGVCoordinator,
    DiffusionRobotCellAgent,
    DiffusionMARL
)


# ============================================================
# 集成方案 2: 混合式智能体
# ============================================================

class HybridProductionScheduler:
    """混合式生产调度系统 - 结合传统方法和扩散式策略"""
    
    def __init__(self, use_diffusion_ratio: float = 0.5):
        """
        初始化混合调度器
        
        Args:
            use_diffusion_ratio: 使用扩散模型的概率 (0.0-1.0)
        """
        # 配置扩散模型
        self.diffusion_config = DiffusionConfig(
            scheduler=DiffusionScheduler.DDPM,
            num_steps=30,  # 减少步数以加快速度
            communication_rounds=2,
            guidance_scale=7.5
        )
        
        # 创建扩散式智能体
        self.diffusion_machine = DiffusionMachineToolAgent(
            agent_id="diffusion_machine",
            machine_ids=["cnc_1", "cnc_2", "cnc_3"],
            config=self.diffusion_config
        )
        
        self.diffusion_agv = DiffusionAGVCoordinator(
            agent_id="diffusion_agv",
            agv_ids=["AGV-01", "AGV-02", "AGV-03"],
            config=self.diffusion_config
        )
        
        self.diffusion_robot = DiffusionRobotCellAgent(
            agent_id="diffusion_robot",
            robot_ids=["ROBOT-01", "ROBOT-02"],
            config=self.diffusion_config
        )
        
        # 初始化 MARL 框架
        self.marl = DiffusionMARL(self.diffusion_config)
        self.marl.register_agent("machine_agent", self.diffusion_machine)
        self.marl.register_agent("agv_agent", self.diffusion_agv)
        self.marl.register_agent("robot_agent", self.diffusion_robot)
        self.marl.initialize_coordinator()
        
        # 混合权重
        self.diffusion_ratio = use_diffusion_ratio
        self.traditional_ratio = 1.0 - use_diffusion_ratio
        
        # 统计信息
        self.stats = {
            'total_schedules': 0,
            'diffusion_used': 0,
            'traditional_used': 0,
            'average_quality': 0.0,
            'average_coordination': 0.0
        }
        
        print(f"✓ 初始化混合调度器")
        print(f"  扩散模型权重: {self.diffusion_ratio:.1%}")
        print(f"  传统方法权重: {self.traditional_ratio:.1%}")
    
    def schedule_machine_work(self, parts: List[Dict], current_time: float) -> Dict:
        """调度机床工作"""
        
        # 决定使用哪种方法
        use_diffusion = np.random.random() < self.diffusion_ratio
        
        if use_diffusion:
            # 使用扩散模型
            schedule = self.diffusion_machine.schedule_parts(parts, current_time)
            method = "Diffusion"
            self.stats['diffusion_used'] += 1
        else:
            # 使用传统方法（简单贪心）
            schedule = self._traditional_machine_schedule(parts, current_time)
            method = "Traditional"
            self.stats['traditional_used'] += 1
        
        self.stats['total_schedules'] += 1
        
        return {
            'schedule': schedule,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
    
    def dispatch_agvs(self, 
                     transport_requests: List[Dict], 
                     current_time: float) -> Dict:
        """调度 AGV"""
        
        use_diffusion = np.random.random() < self.diffusion_ratio
        
        if use_diffusion:
            # 使用扩散模型
            dispatch = self.diffusion_agv.dispatch_agvs(transport_requests, current_time)
            method = "Diffusion"
        else:
            # 使用传统方法
            dispatch = self._traditional_agv_dispatch(transport_requests, current_time)
            method = "Traditional"
        
        return {
            'dispatch': dispatch,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
    
    def assign_robot_tasks(self, tasks: List[Dict]) -> Dict:
        """分配机器人任务"""
        
        use_diffusion = np.random.random() < self.diffusion_ratio
        
        if use_diffusion:
            # 使用扩散模型
            assignment = self.diffusion_robot.assign_robot_tasks(tasks)
            method = "Diffusion"
        else:
            # 使用传统方法
            assignment = self._traditional_robot_assignment(tasks)
            method = "Traditional"
        
        return {
            'assignment': assignment,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_disturbances(self, disturbances: List[Dict], state: Dict) -> Dict:
        """使用 MARL 协调处理扰动"""
        
        environment_state = {
            'active_disturbances': len(disturbances),
            'disturbance_types': [d.get('type') for d in disturbances],
            'current_utilization': state.get('average_utilization', 0.7)
        }
        
        # 训练 MARL 来处理扰动
        marl_result = self.marl.train_episode(environment_state)
        
        responses = []
        for disturbance in disturbances:
            response = self._generate_disturbance_response(
                disturbance,
                marl_result,
                state
            )
            responses.append(response)
        
        return {
            'responses': responses,
            'coordination_quality': marl_result['coordination_quality'],
            'strategy': 'MARL-coordinated'
        }
    
    def update_diffusion_ratio(self, new_ratio: float):
        """动态调整扩散模型的使用比例"""
        self.diffusion_ratio = np.clip(new_ratio, 0.0, 1.0)
        self.traditional_ratio = 1.0 - self.diffusion_ratio
        
        print(f"📊 已更新权重比例")
        print(f"  扩散模型: {self.diffusion_ratio:.1%}")
        print(f"  传统方法: {self.traditional_ratio:.1%}")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = self.stats['total_schedules']
        
        if total > 0:
            diffusion_pct = self.stats['diffusion_used'] / total * 100
            traditional_pct = self.stats['traditional_used'] / total * 100
        else:
            diffusion_pct = traditional_pct = 0
        
        return {
            'total_schedules': total,
            'diffusion_usage': f"{diffusion_pct:.1f}%",
            'traditional_usage': f"{traditional_pct:.1f}%",
            'average_quality': self.stats['average_quality'],
            'average_coordination': self.stats['average_coordination']
        }
    
    # ========================================================
    # 传统方法实现（用于对比）
    # ========================================================
    
    def _traditional_machine_schedule(self, parts: List[Dict], current_time: float) -> List[Dict]:
        """传统的机床调度方法（FIFO + 负载均衡）"""
        
        schedule = []
        
        # 按优先级排序
        sorted_parts = sorted(
            parts,
            key=lambda p: p.get('priority', 50),
            reverse=True
        )
        
        # 为每个零件分配最空闲的机床
        machine_loads = {'cnc_1': 0, 'cnc_2': 0, 'cnc_3': 0}
        
        for part in sorted_parts[:5]:  # 最多调度 5 个
            # 选择负载最低的机床
            lightest_machine = min(machine_loads, key=machine_loads.get)
            
            schedule_entry = {
                'part_id': part.get('part_id', 'unknown'),
                'machine': lightest_machine,
                'start_time': current_time,
                'process': part.get('process', []),
                'priority': part.get('priority', 0)
            }
            
            schedule.append(schedule_entry)
            
            # 更新机床负载
            process_time = len(part.get('process', [])) * 10
            machine_loads[lightest_machine] += process_time
        
        return schedule
    
    def _traditional_agv_dispatch(self, 
                                 requests: List[Dict], 
                                 current_time: float) -> List[Dict]:
        """传统的 AGV 调度方法"""
        
        dispatch = []
        available_agvs = ["AGV-01", "AGV-02", "AGV-03"]
        
        for i, request in enumerate(requests[:len(available_agvs)]):
            dispatch.append({
                'agv_id': available_agvs[i],
                'source': request.get('source', 'warehouse'),
                'destination': request.get('destination', 'machine'),
                'priority': request.get('priority', 1)
            })
        
        return dispatch
    
    def _traditional_robot_assignment(self, tasks: List[Dict]) -> List[Dict]:
        """传统的机器人任务分配方法"""
        
        assignment = []
        robots = ["ROBOT-01", "ROBOT-02"]
        
        for i, task in enumerate(tasks):
            assignment.append({
                'robot_id': robots[i % len(robots)],
                'task_id': task.get('task_id', f'task_{i}'),
                'task_type': task.get('type', 'assembly'),
                'sequence': list(range(min(3, len(task.get('steps', [])))))
            })
        
        return assignment
    
    def _generate_disturbance_response(self, 
                                       disturbance: Dict, 
                                       marl_result: Dict, 
                                       state: Dict) -> Dict:
        """生成扰动应对策略"""
        
        disturbance_type = disturbance.get('type', 'unknown')
        
        # 基于 MARL 协调结果的应对
        if disturbance_type == 'MACHINE_FAILURE':
            response_type = 'REROUTE_TO_BACKUP'
            actions = ['activate_cnc_4', 'notify_planning']
        elif disturbance_type == 'MATERIAL_DELAY':
            response_type = 'REPRIORITIZE_QUEUE'
            actions = ['advance_high_priority_jobs', 'adjust_schedule']
        elif disturbance_type == 'URGENT_ORDER':
            response_type = 'INSERT_JOB'
            actions = ['pause_non_critical', 'expedite_transport']
        else:
            response_type = 'STANDARD_MITIGATION'
            actions = ['monitor', 'prepare_contingency']
        
        return {
            'disturbance_id': disturbance.get('id', 'unknown'),
            'disturbance_type': disturbance_type,
            'response_type': response_type,
            'actions': actions,
            'confidence': min(1.0, marl_result['coordination_quality'] * 1.2)
        }


# ============================================================
# 使用示例
# ============================================================

def example_hybrid_scheduling():
    """混合调度的使用示例"""
    
    print("="*60)
    print("扩散式 MARL 与传统方法的混合调度演示")
    print("="*60)
    
    # 初始化混合调度器（50% 使用扩散模型）
    scheduler = HybridProductionScheduler(use_diffusion_ratio=0.5)
    
    # 模拟零件和任务
    parts = [
        {"part_id": "P001", "priority": 92, "process": ["铣削", "钻孔", "攻丝"]},
        {"part_id": "P002", "priority": 85, "process": ["粗铣", "精铣"]},
        {"part_id": "P003", "priority": 78, "process": ["钻孔"]},
    ]
    
    transport_requests = [
        {"source": "warehouse_A", "destination": "cnc_1", "priority": 1},
        {"source": "cnc_1", "destination": "cnc_2", "priority": 2},
    ]
    
    tasks = [
        {"task_id": "T001", "type": "assembly", "steps": ["step1", "step2", "step3"]},
        {"task_id": "T002", "type": "quality_check", "steps": ["check1", "check2"]},
    ]
    
    disturbances = [
        {"id": "D001", "type": "MACHINE_FAILURE", "severity": "high"},
        {"id": "D002", "type": "URGENT_ORDER", "severity": "medium"},
    ]
    
    system_state = {
        'average_utilization': 0.75,
        'queue_length': 8,
        'active_jobs': 12
    }
    
    # 第 1 步：机床调度
    print("\n📋 第 1 步: 机床调度")
    print("-" * 60)
    
    for epoch in range(3):
        result = scheduler.schedule_machine_work(parts, current_time=epoch*10)
        print(f"Epoch {epoch+1}: 使用方法 = {result['method']}")
        if result['schedule']:
            for s in result['schedule']:
                print(f"  - {s['part_id']} → {s['machine']}")
    
    # 第 2 步：AGV 调度
    print("\n🚚 第 2 步: AGV 调度")
    print("-" * 60)
    
    for epoch in range(2):
        result = scheduler.dispatch_agvs(transport_requests, current_time=epoch*10)
        print(f"Epoch {epoch+1}: 使用方法 = {result['method']}")
        if result['dispatch']:
            for d in result['dispatch']:
                print(f"  - {d['agv_id']}: {d['source']} → {d['destination']}")
    
    # 第 3 步：机器人任务分配
    print("\n🤖 第 3 步: 机器人任务分配")
    print("-" * 60)
    
    result = scheduler.assign_robot_tasks(tasks)
    print(f"使用方法 = {result['method']}")
    if result['assignment']:
        for a in result['assignment']:
            print(f"  - {a['robot_id']}: {a['task_id']} ({a['task_type']})")
    
    # 第 4 步：扰动处理
    print("\n⚠️  第 4 步: 扰动协调处理")
    print("-" * 60)
    
    result = scheduler.handle_disturbances(disturbances, system_state)
    print(f"协调质量: {result['coordination_quality']:.2%}")
    for response in result['responses']:
        print(f"\n  扰动: {response['disturbance_type']}")
        print(f"  应对: {response['response_type']}")
        print(f"  措施: {', '.join(response['actions'])}")
    
    # 第 5 步：动态调整权重
    print("\n📊 第 5 步: 性能反馈与权重调整")
    print("-" * 60)
    
    stats_before = scheduler.get_statistics()
    print(f"\n调整前统计:")
    for key, value in stats_before.items():
        print(f"  {key}: {value}")
    
    # 假设扩散模型表现更好，增加其权重
    scheduler.update_diffusion_ratio(0.7)
    
    # 继续调度
    for epoch in range(3):
        scheduler.schedule_machine_work(parts, current_time=100+epoch*10)
    
    stats_after = scheduler.get_statistics()
    print(f"\n调整后统计:")
    for key, value in stats_after.items():
        print(f"  {key}: {value}")
    
    # 打印最终总结
    print("\n" + "="*60)
    print("✓ 演示完成!")
    print("="*60)


# ============================================================
# 集成到现有系统的示例
# ============================================================

def integrate_with_existing_system():
    """展示如何集成到现有的 production_scheduler_demo.py"""
    
    print("""
    # 在 production_scheduler_demo.py 中添加以下代码：
    
    from hybrid_diffusion_scheduler import HybridProductionScheduler
    
    # 初始化混合调度器（替换现有的智能体）
    hybrid_scheduler = HybridProductionScheduler(use_diffusion_ratio=0.5)
    
    # 在 main 函数中使用
    def main():
        # ... 现有代码 ...
        
        # 替换机床调度
        machine_schedule = hybrid_scheduler.schedule_machine_work(
            parts=current_parts,
            current_time=current_time
        )
        
        # 替换 AGV 调度
        agv_dispatch = hybrid_scheduler.dispatch_agvs(
            transport_requests=requests,
            current_time=current_time
        )
        
        # 替换机器人任务分配
        robot_assignment = hybrid_scheduler.assign_robot_tasks(tasks)
        
        # 协调处理扰动
        disturbance_responses = hybrid_scheduler.handle_disturbances(
            disturbances=detected_disturbances,
            state=current_state
        )
        
        # ... 继续现有流程 ...
    """)


if __name__ == '__main__':
    # 运行演示
    example_hybrid_scheduling()
    
    # 显示集成说明
    print("\n\n" + "="*60)
    print("集成到现有系统的说明:")
    print("="*60)
    integrate_with_existing_system()
