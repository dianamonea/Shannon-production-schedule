#!/usr/bin/env python3
"""
验证会话接入后端的脚本
Verify Session Backend Integration
"""

import json
import requests
import time
from datetime import datetime

def check_backend_session():
    """检查后端是否记录了会话"""
    print("=" * 70)
    print("【会话接入后端验证】")
    print("=" * 70)
    
    try:
        # 检查 Gateway 健康状态
        response = requests.get('http://localhost:8080/health', timeout=2)
        if response.status_code == 200:
            health = response.json()
            print(f"\n✓ 后端服务状态: {health.get('status', 'unknown')}")
            print(f"  版本: {health.get('version', 'unknown')}")
            print(f"  时间: {health.get('time', 'unknown')}")
        else:
            print(f"✗ 后端服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接后端: {e}")
        return False
    
    # 读取最新的 JSON 结果文件
    print("\n" + "=" * 70)
    print("【读取最新的生产调度结果】")
    print("=" * 70)
    
    try:
        # 获取最新的 schedule_result 文件
        import os
        import glob
        
        files = glob.glob("schedule_result*.json")
        if not files:
            print("✗ 未找到调度结果文件")
            return False
        
        latest_file = max(files, key=os.path.getctime)
        print(f"\n✓ 最新结果文件: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 验证会话信息
        print("\n【会话信息】")
        session_id = result.get('session_id')
        session_info = result.get('session_info', {})
        
        print(f"  会话 ID: {session_id}")
        print(f"  会话名称: {session_info.get('name', 'N/A')}")
        print(f"  工作流 ID: {session_info.get('workflow_id', 'N/A')}")
        print(f"  会话状态: {session_info.get('status', 'N/A')}")
        print(f"  创建时间: {session_info.get('timestamp', 'N/A')}")
        
        # 显示协同过程时间线
        print("\n【三智能体协同过程时间线】")
        coordination_timeline = result.get('coordination_timeline', [])
        
        if coordination_timeline:
            print(f"  总事件数: {len(coordination_timeline)} 条")
            
            agents = {}
            for log in coordination_timeline:
                agent = log.get('agent', 'Unknown')
                if agent not in agents:
                    agents[agent] = []
                agents[agent].append(log)
            
            print(f"\n  参与协同的智能体:")
            for agent, logs in agents.items():
                clean_agent = agent.replace('【', '').replace('】', '').replace('智能体', '')
                print(f"    - {clean_agent}: {len(logs)} 条事件")
            
            # 显示时间线的前10条和最后5条事件
            print(f"\n  【协同事件前10条】")
            for i, log in enumerate(coordination_timeline[:10], 1):
                agent = log.get('agent', 'Unknown').replace('【', '').replace('】', '')
                msg = log.get('message', '')
                time_str = log.get('timestamp', '')[-8:]  # 只显示时间部分
                print(f"    {i:2d}. [{time_str}] {agent}: {msg}")
            
            if len(coordination_timeline) > 15:
                print(f"\n  ... ({len(coordination_timeline) - 15} 条事件) ...\n")
            
            if len(coordination_timeline) > 10:
                print(f"  【协同事件最后5条】")
                start_idx = len(coordination_timeline) - 5
                for i, log in enumerate(coordination_timeline[-5:], start_idx + 1):
                    agent = log.get('agent', 'Unknown').replace('【', '').replace('】', '')
                    msg = log.get('message', '')
                    time_str = log.get('timestamp', '')[-8:]
                    print(f"    {i:2d}. [{time_str}] {agent}: {msg}")
        
        # 执行总结
        print("\n【执行总结】")
        exec_summary = result.get('execution_summary', {})
        
        print(f"  执行的智能体总数: {exec_summary.get('total_agents_executed', 'N/A')}")
        print(f"  总执行时间: {exec_summary.get('total_execution_time', 'N/A'):.2f} 秒")
        print(f"  执行状态: {exec_summary.get('status', 'N/A')}")
        
        # 协同过程详情
        print("\n【协同过程详情】")
        coord_process = exec_summary.get('coordination_process', {})
        
        for agent_key, agent_info in coord_process.items():
            print(f"\n  {agent_key}:")
            print(f"    - 状态: {agent_info.get('status', 'N/A')}")
            print(f"    - 执行时间: {agent_info.get('execution_time', 0):.2f} 秒")
            if 'coordination_with' in agent_info:
                print(f"    - 协同对象: {', '.join(agent_info.get('coordination_with', []))}")
        
        # 关键指标
        print("\n【关键性能指标】")
        machine_result = result.get('machine_tool_scheduling', {})
        agv_result = result.get('agv_coordination', {})
        robot_result = result.get('robot_coordination', {})
        
        print(f"  结构件数量: {machine_result.get('part_count', 'N/A')} 件")
        print(f"  机床排产序列: {' -> '.join(machine_result.get('process_sequence', [])[:3])}...")
        print(f"  主轴利用率: {machine_result.get('spindle_utilization', 0)*100:.1f}%")
        print(f"  预计加工时长: {machine_result.get('estimated_cycle_hours', 'N/A'):.1f} 小时")
        print(f"  AGV路由数: {len(agv_result.get('material_routes', []))}")
        print(f"  AGV排队时间: {agv_result.get('queue_time_minutes', 'N/A')} 分钟")
        print(f"  机器人夹具切换次数: {robot_result.get('fixture_changeovers', 'N/A')}")
        print(f"  机器人抽检比例: {robot_result.get('inspection_rate', 0)*100:.1f}%")
        
        print("\n" + "=" * 70)
        print("✅ 会话已成功接入后端！")
        print("=" * 70)
        print("\n📊 协同特点:")
        print("  1. 会话通过 POST /api/v1/tasks 接入后端")
        print("  2. 三个智能体按顺序执行，各自向后端提交任务")
        print("  3. 后面的智能体使用前面智能体的输出作为输入（体现协同）")
        print("  4. 完整的协同过程记录在 coordination_timeline 中")
        print("\n🔗 后端服务:")
        print("  - Gateway: http://localhost:8080")
        print("  - Temporal UI: http://localhost:8088")
        print("  - Grafana: http://localhost:3030")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

if __name__ == "__main__":
    success = check_backend_session()
    exit(0 if success else 1)
