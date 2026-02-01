#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时可视化生产调度进度 - 终端版本
"""

import requests
import time
import json
from datetime import datetime

def get_session_tasks(session_id, api_url="http://localhost:8080"):
    """获取Session中的所有任务"""
    try:
        response = requests.get(
            f"{api_url}/api/v1/sessions/{session_id}",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def format_status(status):
    """格式化状态显示"""
    status_map = {
        'PENDING': '⏳ 等待中',
        'RUNNING': '🔄 运行中',
        'COMPLETED': '✅ 已完成',
        'FAILED': '❌ 失败',
        'CANCELLED': '⚠️ 已取消'
    }
    return status_map.get(status, status)

def visualize_progress(session_id, refresh_interval=3):
    """可视化显示进度"""
    print(f"\n{'='*80}")
    print(f"🎯 实时监控 Shannon 多Agent生产调度系统")
    print(f"{'='*80}\n")
    print(f"📌 Session ID: {session_id}")
    print(f"🔄 刷新间隔: {refresh_interval}秒")
    print(f"💡 按 Ctrl+C 退出\n")
    
    agent_names = {
        1: "订单分析师",
        2: "设备规划师",
        3: "物流协调员",
        4: "质量检查员",
        5: "成本分析师",
        6: "总调度师"
    }
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # 清屏效果（可选）
            if iteration > 1:
                print("\n" + "─"*80 + "\n")
            
            print(f"🕐 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 获取session数据
            data = get_session_tasks(session_id)
            
            if data:
                tasks = data.get("tasks", [])
                total_tasks = len(tasks)
                completed = sum(1 for t in tasks if t.get("status") == "COMPLETED")
                running = sum(1 for t in tasks if t.get("status") == "RUNNING")
                failed = sum(1 for t in tasks if t.get("status") == "FAILED")
                
                # 显示总体进度
                print(f"📊 总体进度: {completed}/{total_tasks} 完成")
                progress_bar_length = 50
                if total_tasks > 0:
                    progress = int((completed / total_tasks) * progress_bar_length)
                    bar = "█" * progress + "░" * (progress_bar_length - progress)
                    percentage = (completed / total_tasks) * 100
                    print(f"[{bar}] {percentage:.1f}%")
                print()
                
                # 显示各Agent状态
                print(f"{'Agent':<15} {'状态':<15} {'任务ID':<45} {'更新时间'}")
                print("─" * 100)
                
                for i, task in enumerate(tasks, 1):
                    task_id = task.get("id", "N/A")
                    status = task.get("status", "UNKNOWN")
                    updated_at = task.get("updated_at", "")
                    
                    # 尝试从task_id或其他字段获取agent名称
                    agent_name = agent_names.get(i, f"Agent {i}")
                    
                    # 截取task_id显示
                    display_task_id = task_id if len(task_id) <= 42 else task_id[:39] + "..."
                    
                    # 格式化更新时间
                    try:
                        if updated_at:
                            time_str = updated_at[:19].replace('T', ' ')
                        else:
                            time_str = ""
                    except:
                        time_str = ""
                    
                    print(f"{agent_name:<15} {format_status(status):<15} {display_task_id:<45} {time_str}")
                
                print()
                
                # 显示详细信息
                if running > 0:
                    print(f"🔄 正在运行: {running} 个任务")
                if failed > 0:
                    print(f"❌ 失败: {failed} 个任务")
                
                # 完成提示
                if completed == total_tasks and total_tasks > 0:
                    print("\n" + "="*80)
                    print("🎉 所有任务已完成！")
                    print("="*80)
                    
                    # 显示在桌面应用查看的提示
                    print(f"\n💡 在Shannon桌面应用中查看完整结果：")
                    print(f"   1. 打开 Shannon 桌面应用或访问 http://localhost:3000")
                    print(f"   2. 搜索 Session ID: {session_id}")
                    print(f"   3. 查看所有Agent的详细执行结果和日志\n")
                    break
                    
            else:
                print(f"⚠️ 无法获取Session数据，Session ID可能不存在或服务未响应")
                print(f"   请确认：")
                print(f"   - Session ID正确: {session_id}")
                print(f"   - Shannon服务运行中: http://localhost:8080/health")
            
            # 等待下次刷新
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        print(f"💡 Session ID: {session_id}")
        print(f"   可在桌面应用中继续查看结果\n")

def main():
    """主函数"""
    # 自动获取最新的session ID或使用指定的
    import sys
    
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        # 使用默认的session ID（从main.py的输出获取）
        session_id = "production-scheduler-1769613594"
        print(f"📝 使用默认Session ID: {session_id}")
        print(f"   如需指定，请运行: python visualize_progress.py <session_id>\n")
    
    visualize_progress(session_id, refresh_interval=3)

if __name__ == "__main__":
    main()
