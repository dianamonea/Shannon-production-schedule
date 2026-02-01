"""
策略对比演示脚本
演示 DT 和 Diffusion 两种策略的推理对比
"""

import requests
import json
import numpy as np
from typing import Dict, List
import time


def create_sample_observation(num_robots: int = 5, num_jobs: int = 10) -> Dict:
    """创建示例观测状态"""
    np.random.seed(42)
    
    robots = [
        {
            'robot_id': f"r{i}",
            'position': {'x': float(np.random.uniform(0, 100)), 'y': float(np.random.uniform(0, 100))},
            'status': 'idle',
            'current_job_id': None,
            'battery_level': float(np.random.uniform(80, 100)),
            'load_capacity': 0.0,
        }
        for i in range(num_robots)
    ]
    
    jobs = [
        {
            'job_id': f"j{i}",
            'job_type': 'transport',
            'source_station_id': f"s{np.random.randint(0, 5)}",
            'target_station_id': f"s{np.random.randint(0, 5)}",
            'deadline': float(np.random.uniform(10, 50)),
            'priority': int(np.random.randint(0, 100)),
            'required_capacity': 0.0,
        }
        for i in range(num_jobs)
    ]
    
    stations = [
        {
            'station_id': f"s{i}",
            'station_type': np.random.choice(['assembly', 'quality_check', 'packaging', 'storage']),
            'position': {'x': float(np.random.uniform(0, 100)), 'y': float(np.random.uniform(0, 100))},
            'is_available': True,
            'queued_jobs': [],
        }
        for i in range(5)
    ]
    
    return {
        't': 0,
        'robots': robots,
        'jobs': jobs,
        'stations': stations,
        'lanes': None,
        'global_time': 0.0,
    }


def test_dt_policy(base_url: str = "http://localhost:8000"):
    """测试 DT 策略"""
    print("\n" + "="*60)
    print("测试 Decision Transformer 策略")
    print("="*60)
    
    # 创建观测序列（DT 需要历史）
    obs = create_sample_observation(num_robots=5, num_jobs=10)
    trajectory = [obs] * 3  # 简化：重复3次作为历史
    
    request_data = {
        'trajectory': trajectory,
        'backend': 'dt',
        'return_logits': False,
    }
    
    print(f"\n请求: POST {base_url}/policy/act")
    print(f"  - Backend: DT")
    print(f"  - Trajectory length: {len(trajectory)}")
    print(f"  - Num robots: {len(obs['robots'])}")
    print(f"  - Num jobs: {len(obs['jobs'])}")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/policy/act",
            json=request_data,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"\n✓ 响应成功 ({elapsed:.2f}ms)")
        print(f"\n元信息:")
        for key, value in result['meta'].items():
            print(f"  {key}: {value}")
        
        print(f"\n动作 ({len(result['actions'])} 个):")
        for action in result['actions'][:5]:  # 只显示前5个
            print(f"  - {action['robot_id']}: {action['action_type']}", end="")
            if action['assign_job_id']:
                print(f" -> {action['assign_job_id']}")
            else:
                print()
        
        if len(result['actions']) > 5:
            print(f"  ... (共 {len(result['actions'])} 个动作)")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return None


def test_diffusion_policy(base_url: str = "http://localhost:8000", num_candidates: int = 5):
    """测试 Diffusion 策略"""
    print("\n" + "="*60)
    print("测试 Diffusion 策略")
    print("="*60)
    
    # 创建观测（Diffusion 只需当前状态）
    obs = create_sample_observation(num_robots=5, num_jobs=10)
    
    request_data = {
        'trajectory': [obs],  # 只需最后一个状态
        'backend': 'diffusion',
        'num_candidates': num_candidates,
        'temperature': 1.0,
        'seed': 42,  # 固定种子用于可复现
    }
    
    print(f"\n请求: POST {base_url}/policy/act")
    print(f"  - Backend: Diffusion")
    print(f"  - Num candidates: {num_candidates}")
    print(f"  - Temperature: 1.0")
    print(f"  - Num robots: {len(obs['robots'])}")
    print(f"  - Num jobs: {len(obs['jobs'])}")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/policy/act",
            json=request_data,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"\n✓ 响应成功 ({elapsed:.2f}ms)")
        print(f"\n元信息:")
        for key, value in result['meta'].items():
            print(f"  {key}: {value}")
        
        print(f"\n主推荐动作:")
        for action in result['actions'][:5]:
            print(f"  - {action['robot_id']}: {action['action_type']}", end="")
            if action['assign_job_id']:
                print(f" -> {action['assign_job_id']}")
            else:
                print()
        
        if result.get('candidates'):
            print(f"\n候选方案 ({len(result['candidates'])} 个):")
            for cand in result['candidates'][:3]:  # 只显示前3个
                print(f"  Rank {cand['rank']}: Score = {cand['score']:.4f}")
                for action in cand['actions'][:3]:
                    print(f"    - {action['robot_id']}: {action.get('assign_job_id', 'idle')}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        return None


def compare_policies(base_url: str = "http://localhost:8000"):
    """对比两种策略"""
    print("\n" + "="*60)
    print("策略对比：DT vs Diffusion")
    print("="*60)
    
    # 使用相同的观测
    obs = create_sample_observation(num_robots=5, num_jobs=10)
    
    # DT 推理
    dt_request = {
        'trajectory': [obs] * 3,
        'backend': 'dt',
    }
    
    print("\n[1/2] DT 推理...")
    dt_start = time.time()
    dt_response = requests.post(f"{base_url}/policy/act", json=dt_request)
    dt_time = (time.time() - dt_start) * 1000
    dt_result = dt_response.json() if dt_response.ok else None
    
    # Diffusion 推理
    diffusion_request = {
        'trajectory': [obs],
        'backend': 'diffusion',
        'num_candidates': 3,
        'seed': 42,
    }
    
    print("[2/2] Diffusion 推理...")
    diff_start = time.time()
    diff_response = requests.post(f"{base_url}/policy/act", json=diffusion_request)
    diff_time = (time.time() - diff_start) * 1000
    diff_result = diff_response.json() if diff_response.ok else None
    
    # 对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print(f"\n推理时间:")
    print(f"  DT:        {dt_time:.2f}ms")
    print(f"  Diffusion: {diff_time:.2f}ms")
    
    if dt_result and diff_result:
        print(f"\n动作数量:")
        print(f"  DT:        {len(dt_result['actions'])}")
        print(f"  Diffusion: {len(diff_result['actions'])}")
        
        print(f"\n特性对比:")
        print(f"  DT:")
        print(f"    - 确定性输出")
        print(f"    - 需要历史序列")
        print(f"    - 单一方案")
        
        print(f"  Diffusion:")
        print(f"    - 随机采样")
        print(f"    - 只需当前状态")
        print(f"    - 多候选方案 ({diff_result['meta'].get('num_candidates', 0)} 个)")
        
        # 动作对比
        print(f"\n动作对比（前3个机器人）:")
        print(f"  {'Robot':<10} {'DT':<15} {'Diffusion':<15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15}")
        
        for i in range(min(3, len(dt_result['actions']), len(diff_result['actions']))):
            dt_action = dt_result['actions'][i]
            diff_action = diff_result['actions'][i]
            
            dt_job = dt_action.get('assign_job_id', 'idle')
            diff_job = diff_action.get('assign_job_id', 'idle')
            
            print(f"  {dt_action['robot_id']:<10} {dt_job:<15} {diff_job:<15}")


def save_results_to_file(dt_result: Dict, diff_result: Dict, output_file: str = "demo_results.json"):
    """保存对比结果到文件"""
    results = {
        'dt': dt_result,
        'diffusion': diff_result,
        'comparison': {
            'dt_inference_time_ms': dt_result['meta']['inference_time_ms'],
            'diffusion_inference_time_ms': diff_result['meta']['inference_time_ms'],
            'dt_num_actions': len(dt_result['actions']),
            'diffusion_num_actions': len(diff_result['actions']),
            'diffusion_num_candidates': diff_result['meta'].get('num_candidates', 1),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Policy Demo')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000', help='Service base URL')
    parser.add_argument('--mode', type=str, default='compare', choices=['dt', 'diffusion', 'compare'],
                        help='Test mode')
    parser.add_argument('--num_candidates', type=int, default=5, help='Diffusion candidates')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # 检查服务健康
    try:
        health_response = requests.get(f"{args.base_url}/health", timeout=5)
        health_response.raise_for_status()
        health = health_response.json()
        
        print("\n服务状态:")
        print(f"  Status: {health['status']}")
        print(f"  Available backends:")
        for backend, info in health['backends'].items():
            print(f"    - {backend}: {info['name']} v{info['version']}")
    
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 服务不可用: {e}")
        print(f"\n请先启动服务:")
        print(f"  python -m uvicorn policy_service.unified_service:app --port 8000")
        return
    
    # 执行测试
    if args.mode == 'dt':
        dt_result = test_dt_policy(args.base_url)
    
    elif args.mode == 'diffusion':
        diff_result = test_diffusion_policy(args.base_url, args.num_candidates)
    
    elif args.mode == 'compare':
        compare_policies(args.base_url)


if __name__ == '__main__':
    main()
