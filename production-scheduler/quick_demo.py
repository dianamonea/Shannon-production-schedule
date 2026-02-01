#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示脚本 - 在Shannon桌面应用上快速显示结果
"""

import json
import time
import requests

def quick_demo():
    """快速演示 - 直接查询已有的任务"""
    session_id = "production-scheduler-1769502931"
    api_url = "http://localhost:8080"
    
    print("\n" + "="*70)
    print("🔍 查询当前执行的任务...")
    print("="*70 + "\n")
    
    # 查询所有任务
    try:
        response = requests.get(
            f"{api_url}/api/v1/sessions/{session_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Session ID: {session_id}")
            print(f"📊 任务列表：")
            
            # 显示所有任务
            tasks = data.get("tasks", [])
            for i, task in enumerate(tasks, 1):
                task_id = task.get("id", "N/A")
                status = task.get("status", "UNKNOWN")
                print(f"   {i}. {task_id}")
                print(f"      状态: {status}")
                print()
        else:
            print(f"❌ 查询失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        
    # 提示用户在桌面应用上查看
    print("\n" + "="*70)
    print("💡 在Shannon桌面应用上查看结果：")
    print("="*70)
    print(f"\n1. 打开Shannon桌面应用")
    print(f"2. 搜索Session ID：{session_id}")
    print(f"3. 查看所有Agent的执行状态和结果")
    print(f"\n或点击 'My Agents' 查看所有正在运行的任务\n")

if __name__ == "__main__":
    quick_demo()
