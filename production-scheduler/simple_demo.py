#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单演示 - 创建单个可在Web UI查看的任务
"""

import json
import requests
import time

def create_simple_task():
    """创建一个简单的任务用于Web UI可视化"""
    api_url = "http://localhost:8080"
    
    # 创建一个生产调度查询
    query = """
作为生产调度总监，请分析以下生产场景并给出建议：

📦 订单信息：
- 订单A：100件产品，紧急程度高，交期3天
- 订单B：200件产品，紧急程度中，交期7天  
- 订单C：150件产品，紧急程度低，交期14天

🏭 设备状况：
- 生产线1：可用，效率100%
- 生产线2：可用，效率80%
- 生产线3：维修中，预计2天后可用

📊 库存情况：
- 原材料A：充足
- 原材料B：仅够150件产品
- 原材料C：需要采购

请分析：
1. 订单优先级排序
2. 设备分配方案
3. 物流和采购建议
4. 质量控制要点
5. 成本预估
6. 最终调度方案
"""
    
    print("\n" + "="*70)
    print("🚀 创建生产调度分析任务")
    print("="*70 + "\n")
    
    try:
        response = requests.post(
            f"{api_url}/api/v1/tasks",
            headers={"Content-Type": "application/json"},
            json={
                "query": query,
                "session_id": f"production-demo-{int(time.time())}"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get("task_id")
            session_id = data.get("session_id", f"production-demo-{int(time.time())}")
            
            print(f"✅ 任务创建成功！\n")
            print(f"📋 Task ID: {task_id}")
            print(f"📌 Session ID: {session_id}\n")
            print("="*70)
            print("🌐 在Web UI中查看实时进度：")
            print("="*70)
            print(f"\n1️⃣ 打开浏览器访问: http://localhost:3000")
            print(f"\n2️⃣ 点击左侧菜单 '我的代理人们' 或 'Sessions'")
            print(f"\n3️⃣ 查找 Session ID: {session_id}")
            print(f"\n   或直接访问:")
            print(f"   http://localhost:3000/run-detail?session_id={session_id}")
            print(f"\n4️⃣ 查看AI实时分析生产调度方案！\n")
            
            print("="*70)
            print("📺 其他可视化方式：")
            print("="*70)
            print(f"\n- Temporal UI: http://localhost:8088")
            print(f"  搜索: {task_id}")
            print(f"\n- Grafana: http://localhost:3030")
            print(f"  查看性能指标\n")
            
            return task_id, session_id
        else:
            print(f"❌ 创建失败: HTTP {response.status_code}")
            print(f"响应: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None, None

if __name__ == "__main__":
    create_simple_task()
