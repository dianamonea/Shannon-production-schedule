#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化启动脚本 - 等待Docker就绪后自动运行演示
"""

import subprocess
import time
import sys

def run_command(cmd, shell=True):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def wait_for_docker(max_retries=120):
    """等待Docker服务启动"""
    print("\n⏳ 等待Docker服务启动...\n")
    
    for attempt in range(max_retries):
        success, stdout, _ = run_command("docker ps")
        if success and "shannon" in stdout:
            containers = [line for line in stdout.split('\n') if 'shannon' in line]
            running = len(containers)
            print(f"✅ Docker容器已启动: {running} 个服务在线 (尝试 {attempt+1}/{max_retries})")
            return True
        
        if attempt % 10 == 0:
            print(f"⏳ 等待中... ({attempt+1}/{max_retries})")
        
        time.sleep(1)
    
    return False

def wait_for_api(max_retries=60):
    """等待API服务启动"""
    print("\n⏳ 等待API服务启动...\n")
    
    import requests
    
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API已就绪！ (尝试 {attempt+1}/{max_retries})")
                return True
        except:
            pass
        
        if attempt % 10 == 0:
            print(f"⏳ 等待API就绪... ({attempt+1}/{max_retries})")
        
        time.sleep(1)
    
    return False

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 Shannon 生产调度系统 - 自动启动脚本")
    print("="*80)
    
    # 等待Docker
    if not wait_for_docker():
        print("\n❌ Docker服务启动超时")
        sys.exit(1)
    
    # 等待API
    if not wait_for_api():
        print("\n❌ API服务启动超时")
        sys.exit(1)
    
    # 运行演示
    print("\n" + "="*80)
    print("🎯 启动生产调度演示")
    print("="*80)
    
    import os
    os.chdir(r"c:\Users\Administrator\Documents\GitHub\Shannon\production-scheduler")
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 运行演示脚本
    success, stdout, stderr = run_command("python run_demo.py")
    
    if success:
        print(stdout)
        print("\n✅ 演示已启动！\n")
        print("现在请打开以下链接查看实时进度：")
        print("  - Temporal UI: http://localhost:8088")
        print("  - Shannon Web: http://localhost:3000\n")
    else:
        print(f"❌ 运行失败: {stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
