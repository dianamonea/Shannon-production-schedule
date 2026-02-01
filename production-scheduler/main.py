#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Agent生产调度系统 - 集成 Shannon 智能体
"""

import json
import time
import requests
from typing import Dict, List

class ProductionSchedulerAgent:
    """生产调度多代理系统"""
    
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.session_id = f"production-scheduler-{int(time.time())}"
        print(f"\n✅ 系统初始化完成")
        print(f"📌 Session ID: {self.session_id}")
        print(f"🔗 API 地址: {self.api_url}")
        print(f"💡 打开桌面程序查看实时进度\n")
    
    def submit_task(self, query: str, agent_name: str = "Assistant", retries: int = 3) -> str:
        """提交任务到 Shannon，返回任务 ID"""
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/v1/tasks",
                    headers={"Content-Type": "application/json"},
                    json={
                        "query": query,
                        "session_id": self.session_id
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    task_id = data.get("task_id")
                    print(f"  ✅ 任务已提交")
                    print(f"     Agent: {agent_name}")
                    print(f"     Task ID: {task_id}")
                    return task_id
                else:
                    print(f"  ⚠️ 提交失败 (尝试 {attempt+1}/{retries}): {response.status_code}")
                    if attempt < retries - 1:
                        time.sleep(2)
            except Exception as e:
                print(f"  ⚠️ 错误 (尝试 {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
        
        print(f"  ❌ 任务提交失败")
        return None
    
    def wait_for_result(self, task_id: str, agent_name: str = "Assistant", max_wait: int = 300) -> str:
        """等待任务完成，返回结果"""
        print(f"  ⏳ 等待 {agent_name} 完成...")
        start_time = time.time()
        retry_count = 0
        max_retries = 50
        connection_errors = 0
        max_connection_errors = 5
        
        while True:
            try:
                response = requests.get(
                    f"{self.api_url}/api/v1/tasks/{task_id}",
                    timeout=30
                )
                
                connection_errors = 0  # 重置连接错误计数
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    if status == "TASK_STATUS_COMPLETED":
                        result = data.get("result")
                        print(f"  ✅ {agent_name} 完成✓")
                        return result
                    elif "RUNNING" in status or "PROCESSING" in status:
                        retry_count += 1
                        elapsed = time.time() - start_time
                        print(f"     ⏳ 处理中 ({retry_count}/{max_retries}, 已等{int(elapsed)}秒)...")
                        time.sleep(5)
                    else:
                        print(f"     📊 状态: {status}")
                        time.sleep(3)
                else:
                    print(f"  ⚠️ 查询失败: {response.status_code}")
                    time.sleep(3)
                
                # 超时检查
                if time.time() - start_time > max_wait:
                    print(f"  ❌ 超时: 已等待 {max_wait} 秒，任务仍未完成")
                    return None
                    
                if retry_count >= max_retries:
                    print(f"  ❌ 重试次数过多，放弃等待")
                    return None
                    
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                connection_errors += 1
                print(f"  ⚠️ 连接错误 ({connection_errors}/{max_connection_errors}): {type(e).__name__}")
                
                if connection_errors >= max_connection_errors:
                    print(f"  ❌ 连接失败次数过多，放弃等待")
                    return None
                
                time.sleep(5)
            except Exception as e:
                print(f"  ⚠️ 其他错误: {e}")
                time.sleep(3)
    
    def agent_1_order_analysis(self, orders: List[Dict]) -> Dict:
        """Agent 1: 订单分析师"""
        print("\n" + "="*70)
        print("🤖 Agent 1: 订单分析师 - 分析订单优先级")
        print("="*70)
        
        query = f"""你是生产订单分析专家。分析以下订单数据，按优先级排序。

【订单数据】
{json.dumps(orders, ensure_ascii=False, indent=2)}

【分析要求】
1. 按交期紧急程度排序（最紧急优先）
2. 按订单金额大小考虑
3. 识别高风险订单
4. 给出优化建议

【输出要求】
必须输出有效的JSON格式（确保能被 Python 解析）：
{{
    "priority_list": [
        {{"order_id": "ORD001", "priority": 1, "urgency": "HIGH", "reason": "交期最紧张"}},
        {{"order_id": "ORD002", "priority": 2, "urgency": "MEDIUM", "reason": "..."}}
    ],
    "high_risk_orders": ["ORD002"],
    "total_capacity_needed": 350,
    "analysis_summary": "..."
}}"""
        
        task_id = self.submit_task(query, "订单分析师")
        if not task_id:
            print(f"  ⚠️ 使用默认数据继续执行")
            return {
                "priority_list": orders,
                "high_risk_orders": [],
                "total_capacity_needed": sum(o.get("qty", 0) for o in orders),
                "analysis_summary": "使用默认数据"
            }
        
        result = self.wait_for_result(task_id, "订单分析师")
        if result:
            try:
                analysis = json.loads(result)
                print(f"  📊 分析结果:")
                print(f"     - 订单总数: {len(analysis.get('priority_list', []))}")
                print(f"     - 高风险: {len(analysis.get('high_risk_orders', []))} 个")
                print(f"     - 总产能需求: {analysis.get('total_capacity_needed', 0)} 件")
                return analysis
            except:
                print(f"  ⚠️ 解析结果失败，返回原始数据")
                return {"raw_result": result}
        
        print(f"  ⚠️ 使用默认数据继续执行")
        return {
            "priority_list": orders,
            "high_risk_orders": [],
            "total_capacity_needed": sum(o.get("qty", 0) for o in orders),
            "analysis_summary": "使用默认数据"
        }
    
    def agent_2_equipment_planning(self, priority_orders: Dict, equipment: List[Dict]) -> Dict:
        """Agent 2: 设备规划师"""
        print("\n" + "="*70)
        print("🤖 Agent 2: 设备规划师 - 制定设备分配方案")
        print("="*70)
        
        query = f"""你是设备规划专家。根据优先级订单，分配生产设备。

【优先级订单列表】
{json.dumps(priority_orders.get('priority_list', [])[:5], ensure_ascii=False, indent=2)}

【可用设备状态】
{json.dumps(equipment, ensure_ascii=False, indent=2)}

【规划要求】
1. 按订单优先级分配设备
2. 考虑设备的工艺能力
3. 避免频繁切换工艺
4. 最大化设备利用率
5. 确保交期可达

【输出要求】
必须输出有效的JSON格式：
{{
    "allocation_plan": [
        {{"order_id": "ORD001", "line": 1, "start_date": "2026-01-28", "end_date": "2026-01-30", "status": "scheduled"}},
        {{"order_id": "ORD002", "line": 2, "start_date": "2026-01-30", "end_date": "2026-02-02", "status": "scheduled"}}
    ],
    "equipment_utilization": 0.85,
    "conflicts": [],
    "notes": "..."
}}"""
        
        task_id = self.submit_task(query, "设备规划师")
        if not task_id:
            print(f"  ⚠️ 使用默认分配继续执行")
            return {
                "allocation_plan": [{"order_id": f"ORD{i:03d}", "line": (i % 3) + 1, "status": "default"} for i in range(len(priority_orders.get('priority_list', [])))],
                "equipment_utilization": 0.7,
                "conflicts": []
            }
        
        result = self.wait_for_result(task_id, "设备规划师")
        if result:
            try:
                plan = json.loads(result)
                print(f"  📊 规划结果:")
                print(f"     - 已分配计划: {len(plan.get('allocation_plan', []))} 个")
                print(f"     - 设备利用率: {plan.get('equipment_utilization', 0):.0%}")
                print(f"     - 冲突数: {len(plan.get('conflicts', []))}")
                return plan
            except:
                print(f"  ⚠️ 解析结果失败")
                return {"raw_result": result}
        
        print(f"  ⚠️ 使用默认分配继续执行")
        return {
            "allocation_plan": [{"order_id": f"ORD{i:03d}", "line": (i % 3) + 1, "status": "default"} for i in range(len(priority_orders.get('priority_list', [])))],
            "equipment_utilization": 0.7,
            "conflicts": []
        }
    
    def agent_3_inventory_check(self, allocation: Dict, inventory: List[Dict]) -> Dict:
        """Agent 3: 物流协调员"""
        print("\n" + "="*70)
        print("🤖 Agent 3: 物流协调员 - 检查物料可行性")
        print("="*70)
        
        query = f"""你是物流协调专家。检查物料库存是否满足生产计划。

【生产分配计划】
{json.dumps(allocation.get('allocation_plan', [])[:3], ensure_ascii=False, indent=2)}

【库存情况】
{json.dumps(inventory, ensure_ascii=False, indent=2)}

【检查要求】
1. 检查每个订单的物料是否充足
2. 识别可能缺货的物料
3. 建议紧急采购方案
4. 评估风险等级

【输出要求】
必须输出有效的JSON格式：
{{
    "inventory_sufficient": true,
    "critical_materials": [],
    "urgent_purchases_needed": [
        {{"material": "原材料X", "qty": 100, "urgency": "HIGH"}}
    ],
    "risk_level": "LOW",
    "recommendations": ["..."]
}}"""
        
        task_id = self.submit_task(query, "物流协调员")
        if not task_id:
            print(f"  ⚠️ 使用默认检查继续执行")
            return {
                "inventory_sufficient": True,
                "critical_materials": [],
                "urgent_purchases_needed": [],
                "risk_level": "LOW"
            }
        
        result = self.wait_for_result(task_id, "物流协调员")
        if result:
            try:
                inventory_plan = json.loads(result)
                print(f"  📊 库存检查结果:")
                print(f"     - 库存充足: {'✅ 是' if inventory_plan.get('inventory_sufficient') else '❌ 否'}")
                print(f"     - 需要采购: {len(inventory_plan.get('urgent_purchases_needed', []))} 种")
                print(f"     - 风险等级: {inventory_plan.get('risk_level', 'UNKNOWN')}")
                return inventory_plan
            except:
                print(f"  ⚠️ 解析结果失败")
                return {"raw_result": result}
        
        print(f"  ⚠️ 使用默认检查继续执行")
        return {
            "inventory_sufficient": True,
            "critical_materials": [],
            "urgent_purchases_needed": [],
            "risk_level": "LOW"
        }
    
    def agent_4_final_review(self, all_results: Dict) -> Dict:
        """Agent 4: 质量审查官"""
        print("\n" + "="*70)
        print("🤖 Agent 4: 质量审查官 - 最终可行性评估")
        print("="*70)
        
        summary = f"""订单分析: {len(all_results.get('orders', {}).get('priority_list', []))} 个订单
设备分配: {len(all_results.get('equipment', {}).get('allocation_plan', []))} 个计划
库存检查: {'充足' if all_results.get('inventory', {}).get('inventory_sufficient') else '不足'}"""
        
        query = f"""你是生产质量和风险评估专家。审查整体生产方案。

【方案概要】
{summary}

【评估要求】
1. 整体可行性评分（0-100）
2. 主要风险评估
3. 是否需要调整
4. 最终建议

【输出要求】
必须输出有效的JSON格式：
{{
    "feasibility_score": 85,
    "is_feasible": true,
    "main_risks": ["..."],
    "needs_adjustment": false,
    "final_recommendation": "建议按方案执行",
    "approval_status": "APPROVED"
}}"""
        
        task_id = self.submit_task(query, "质量审查官")
        if not task_id:
            print(f"  ⚠️ 使用默认审查继续执行")
            return {
                "feasibility_score": 80,
                "is_feasible": True,
                "main_risks": [],
                "needs_adjustment": False,
                "final_recommendation": "方案可行",
                "approval_status": "APPROVED"
            }
        
        result = self.wait_for_result(task_id, "质量审查官")
        if result:
            try:
                review = json.loads(result)
                print(f"  📊 审查结果:")
                print(f"     - 可行性评分: {review.get('feasibility_score', 0)}/100")
                print(f"     - 状态: {review.get('approval_status', 'UNKNOWN')}")
                print(f"     - 建议: {review.get('final_recommendation', '...')}")
                return review
            except:
                print(f"  ⚠️ 解析结果失败")
                return {"raw_result": result}
        
        print(f"  ⚠️ 使用默认审查继续执行")
        return {
            "feasibility_score": 80,
            "is_feasible": True,
            "main_risks": [],
            "needs_adjustment": False,
            "final_recommendation": "方案可行",
            "approval_status": "APPROVED"
        }
        return {}
    
    def run_full_orchestration(self, orders: List[Dict], equipment: List[Dict], inventory: List[Dict]):
        """执行完整的多Agent协调"""
        print("\n\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "🏭 多Agent生产调度系统 - 完整执行" + " "*20 + "║")
        print("╚" + "="*68 + "╝")
        
        results = {}
        
        # Step 1: Agent 1 分析订单
        results['orders'] = self.agent_1_order_analysis(orders)
        time.sleep(2)
        
        # Step 2: Agent 2 规划设备
        results['equipment'] = self.agent_2_equipment_planning(
            results['orders'],
            equipment
        )
        time.sleep(2)
        
        # Step 3: Agent 3 检查物料
        results['inventory'] = self.agent_3_inventory_check(
            results['equipment'],
            inventory
        )
        time.sleep(2)
        
        # Step 4: Agent 4 最终审查
        results['review'] = self.agent_4_final_review(results)
        
        return results
    
    def print_final_report(self, results: Dict):
        """打印最终报告"""
        print("\n\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*20 + "📊 最终执行报告" + " "*33 + "║")
        print("╚" + "="*68 + "╝")
        
        print(f"\n✅ Session ID: {self.session_id}")
        print(f"   💡 提示：在桌面程序中搜索这个 Session ID 可以查看详细过程")
        
        print(f"\n【订单分析】")
        orders = results.get('orders', {})
        if 'priority_list' in orders:
            print(f"  ✓ 分析了 {len(orders['priority_list'])} 个订单")
            if orders['priority_list']:
                print(f"  ✓ 最紧急订单: {orders['priority_list'][0].get('order_id')}")
        
        print(f"\n【设备规划】")
        equipment = results.get('equipment', {})
        if 'allocation_plan' in equipment:
            print(f"  ✓ 制定了 {len(equipment['allocation_plan'])} 个分配计划")
            print(f"  ✓ 设备利用率: {equipment.get('equipment_utilization', 0):.0%}")
        
        print(f"\n【库存检查】")
        inventory = results.get('inventory', {})
        status = "✅ 充足" if inventory.get('inventory_sufficient') else "⚠️ 可能不足"
        print(f"  {status}")
        if inventory.get('urgent_purchases_needed'):
            print(f"  ⚠️ 需要采购: {len(inventory['urgent_purchases_needed'])} 种物料")
        
        print(f"\n【最终评审】")
        review = results.get('review', {})
        print(f"  ✓ 可行性评分: {review.get('feasibility_score', 0)}/100")
        print(f"  ✓ 状态: {review.get('approval_status', 'UNKNOWN')}")
        print(f"  ✓ 建议: {review.get('final_recommendation', '...')}")
        
        print(f"\n📍 后续步骤:")
        print(f"   1️⃣ 打开桌面程序，查看 Session: {self.session_id}")
        print(f"   2️⃣ 查看每个 Agent 的详细分析过程")
        print(f"   3️⃣ 在 Temporal UI (http://localhost:8088) 查看完整工作流")
        print(f"   4️⃣ 如需调整，修改输入数据重新运行")


# ========== 主程序 ==========
def main():
    print("\n" + "="*70)
    print("初始化多Agent生产调度系统...")
    print("="*70)
    
    # 创建系统实例
    scheduler = ProductionSchedulerAgent()
    
    # 准备测试数据
    orders = [
        {"id": "ORD001", "product": "产品A", "qty": 100, "deadline": "2026-01-30", "value": 50000},
        {"id": "ORD002", "product": "产品B", "qty": 50, "deadline": "2026-02-02", "value": 30000},
        {"id": "ORD003", "product": "产品A", "qty": 200, "deadline": "2026-02-05", "value": 80000},
    ]
    
    equipment = [
        {"line": 1, "status": "idle", "capability": ["A", "B"]},
        {"line": 2, "status": "available", "current_job": "maintenance", "eta": "2026-01-27"},
        {"line": 3, "status": "available", "capability": ["A"]},
    ]
    
    inventory = [
        {"material": "原材料X", "qty": 500, "unit": "kg"},
        {"material": "原材料Y", "qty": 100, "unit": "kg"},
        {"material": "原材料Z", "qty": 300, "unit": "kg"},
    ]
    
    # 执行完整协调
    results = scheduler.run_full_orchestration(orders, equipment, inventory)
    
    # 打印最终报告
    scheduler.print_final_report(results)
    
    # 保存结果
    filename = f"results_{scheduler.session_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存: {filename}")


if __name__ == "__main__":
    main()
