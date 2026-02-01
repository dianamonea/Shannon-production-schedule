"""
MADT Policy Service 启动脚本
提供多种运行方式
"""

import sys
import subprocess
from pathlib import Path


def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("MADT Policy Service - Quick Start")
    print("="*60)
    print("\n选择操作:")
    print("1. 运行单元测试 (test)")
    print("2. 生成合成数据 (data)")
    print("3. 启动训练 (train)")
    print("4. 启动推理服务 (serve)")
    print("5. 查看文档 (help)")
    print("6. 退出 (exit)")
    print()


def run_tests():
    """运行测试"""
    print("\n▶ 运行单元测试...")
    result = subprocess.run(
        [sys.executable, "test_madt.py"],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def generate_data():
    """生成数据"""
    print("\n▶ 生成合成数据...")
    num_episodes = input("输入 episode 数量 (默认 50): ").strip() or "50"
    
    try:
        num = int(num_episodes)
        result = subprocess.run(
            [sys.executable, "generate_data.py", str(num), "./data/episodes"],
            cwd=Path(__file__).parent,
        )
        return result.returncode == 0
    except ValueError:
        print("✗ 无效的数字")
        return False


def train():
    """启动训练"""
    print("\n▶ 启动训练 (BC)...")
    print("配置: configs/v1_bc.yaml")
    print("数据: data/episodes/episodes.jsonl")
    print()
    
    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--config", "configs/v1_bc.yaml"],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def serve():
    """启动服务"""
    print("\n▶ 启动 FastAPI 推理服务...")
    print("地址: http://localhost:8000")
    print("文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务\n")
    
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def show_help():
    """显示帮助文档"""
    help_text = """
╔════════════════════════════════════════════════════════════╗
║  MADT Policy Service - Multi-Agent Decision Transformer   ║
╚════════════════════════════════════════════════════════════╝

【快速开始】
  1. python start.py  → 交互菜单
  2. python test_madt.py  → 单元测试
  3. python generate_data.py 50 ./data/episodes  → 生成数据
  4. python -m training.train --config configs/v1_bc.yaml  → 训练
  5. uvicorn app:app --host 0.0.0.0 --port 8000  → 推理服务

【API 端点】
  POST /policy/act           → 推理 (单个请求)
  POST /policy/act_batch     → 批量推理
  GET  /policy/info          → 策略信息
  GET  /health               → 健康检查

【项目结构】
  app.py                     → FastAPI 推理服务
  test_madt.py              → 单元测试 (6 个测试)
  generate_data.py          → 数据生成脚本
  common/schemas.py         → Pydantic 数据结构
  common/vectorizer.py      → 向量化 (状态/动作)
  training/model.py         → Decision Transformer
  training/dataset.py       → 数据加载器
  training/train.py         → 训练脚本 (BC)

【模型配置】
  configs/v1_bc.yaml        → v1 行为克隆配置
  - hidden_dim: 256
  - num_layers: 4
  - num_heads: 8
  - sequence_length: 4 (K 步)

【下一步】
  □ 运行测试: python test_madt.py
  □ 生成数据: python generate_data.py 100 ./data/episodes
  □ 启动训练: python -m training.train --config configs/v1_bc.yaml
  □ 启动服务: uvicorn app:app --port 8000
  □ 测试推理: curl http://localhost:8000/health

【文档】
  README.md                  → 详细文档
  https://arxiv.org/abs/2106.01021  → Decision Transformer 论文

═══════════════════════════════════════════════════════════════
"""
    print(help_text)


def main():
    """主菜单"""
    print("\n✓ MADT Policy Service 初始化完成")
    
    while True:
        print_menu()
        choice = input("请输入选项 (1-6): ").strip()
        
        try:
            if choice == "1":
                success = run_tests()
                if not success:
                    print("✗ 测试失败")
            elif choice == "2":
                success = generate_data()
                if success:
                    print("✓ 数据生成完成")
                else:
                    print("✗ 数据生成失败")
            elif choice == "3":
                success = train()
                if success:
                    print("✓ 训练完成")
                else:
                    print("✗ 训练失败或被中断")
            elif choice == "4":
                success = serve()
                if success:
                    print("✓ 服务已停止")
                else:
                    print("✗ 服务异常停止")
            elif choice == "5":
                show_help()
            elif choice == "6":
                print("\n👋 再见！")
                break
            else:
                print("✗ 无效的选项")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  被中断")
        except Exception as e:
            print(f"\n✗ 错误: {e}")


if __name__ == '__main__':
    main()
