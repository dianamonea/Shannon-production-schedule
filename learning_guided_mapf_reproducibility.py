"""
Learning-Guided MAPF 复现性代码包
Reproducibility Package for Top-tier Conference Paper

包含所有可复现实验所需的脚本和配置

作者：Shannon Research Team
日期：2026-02-01
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import time


@dataclass
class ExperimentConfig:
    """实验配置"""
    seed: int = 42
    num_trials: int = 25
    timeout: int = 300
    
    # 模型配置
    gnn_layers: int = 3
    gnn_hidden: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 256
    dropout: float = 0.1
    
    # 训练配置
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # 数据配置
    train_maps: int = 100
    val_maps: int = 20
    test_maps: int = 50
    agents_range: tuple = (10, 200)
    
    def get_hash(self) -> str:
        """获取配置的唯一哈希值"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ReproducibilityManager:
    """复现性管理器"""
    
    def __init__(self, output_dir: str = './reproducibility'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_requirements(self) -> str:
        """生成requirements.txt"""
        requirements = """# Learning-Guided MAPF Requirements
# Python 3.8+

# Core Dependencies
torch>=1.12.0
torch-geometric>=2.2.0
numpy>=1.22.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0

# Data Processing
pandas>=1.5.0
networkx>=2.8.0
pyyaml>=6.0

# Visualization
plotly>=5.11.0

# Experiment Tracking
wandb>=0.13.0
tensorboard>=2.11.0

# Development Tools
pytest>=7.2.0
black>=22.12.0
isort>=5.11.0
mypy>=0.991

# Documentation
sphinx>=5.3.0
sphinx-rtd-theme>=1.1.0
"""
        return requirements
    
    def generate_environment_yml(self) -> str:
        """生成conda环境配置"""
        yml = """# Learning-Guided MAPF Conda Environment
name: lg-mapf
channels:
  - pytorch
  - pyg
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=1.12.0
  - pyg>=2.2.0
  - numpy>=1.22.0
  - scipy>=1.9.0
  - matplotlib>=3.6.0
  - seaborn>=0.12.0
  - pandas>=1.5.0
  - networkx>=2.8.0
  - pyyaml>=6.0
  - pytest>=7.2.0
  - pip
  - pip:
    - wandb>=0.13.0
    - plotly>=5.11.0
    - tensorboard>=2.11.0
"""
        return yml
    
    def generate_run_all_script(self) -> str:
        """生成运行所有实验的脚本"""
        script = '''#!/bin/bash
# Learning-Guided MAPF - Run All Experiments
# This script reproduces all results from the paper

set -e  # Exit on error

echo "====================================="
echo "Learning-Guided MAPF Experiments"
echo "====================================="

# Check environment
echo "Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Set random seeds
export PYTHONHASHSEED=42

# Create output directories
mkdir -p results/main
mkdir -p results/ablation
mkdir -p results/generalization
mkdir -p results/scalability
mkdir -p figures
mkdir -p latex_tables

# Step 1: Generate training data
echo ""
echo "[1/8] Generating training data..."
python learning_guided_mapf_training.py generate --seed 42

# Step 2: Train model
echo ""
echo "[2/8] Training model..."
python learning_guided_mapf_training.py train --seed 42 --epochs 100

# Step 3: Run main comparison experiments
echo ""
echo "[3/8] Running main comparison experiments..."
python learning_guided_mapf_comparison.py --output results/main

# Step 4: Run ablation experiments
echo ""
echo "[4/8] Running ablation experiments..."
python learning_guided_mapf_experiments.py ablation --output results/ablation

# Step 5: Run generalization experiments
echo ""
echo "[5/8] Running generalization experiments..."
python learning_guided_mapf_experiments.py generalization --output results/generalization

# Step 6: Run scalability experiments
echo ""
echo "[6/8] Running scalability experiments..."
python learning_guided_mapf_experiments.py scalability --output results/scalability

# Step 7: Generate figures
echo ""
echo "[7/8] Generating paper figures..."
python learning_guided_mapf_paper_figures.py --output figures

# Step 8: Generate LaTeX tables
echo ""
echo "[8/8] Generating LaTeX tables..."
python learning_guided_mapf_latex_tables.py --output latex_tables

echo ""
echo "====================================="
echo "All experiments completed!"
echo "Results saved in: ./results/"
echo "Figures saved in: ./figures/"
echo "Tables saved in: ./latex_tables/"
echo "====================================="
'''
        return script
    
    def generate_run_all_ps1(self) -> str:
        """生成Windows PowerShell脚本"""
        script = '''# Learning-Guided MAPF - Run All Experiments (Windows)
# This script reproduces all results from the paper

$ErrorActionPreference = "Stop"

Write-Host "====================================="
Write-Host "Learning-Guided MAPF Experiments"
Write-Host "====================================="

# Check environment
Write-Host "Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Set random seeds
$env:PYTHONHASHSEED = "42"

# Create output directories
New-Item -ItemType Directory -Force -Path "results/main" | Out-Null
New-Item -ItemType Directory -Force -Path "results/ablation" | Out-Null
New-Item -ItemType Directory -Force -Path "results/generalization" | Out-Null
New-Item -ItemType Directory -Force -Path "results/scalability" | Out-Null
New-Item -ItemType Directory -Force -Path "figures" | Out-Null
New-Item -ItemType Directory -Force -Path "latex_tables" | Out-Null

# Step 1: Generate training data
Write-Host ""
Write-Host "[1/8] Generating training data..."
python learning_guided_mapf_training.py generate --seed 42

# Step 2: Train model
Write-Host ""
Write-Host "[2/8] Training model..."
python learning_guided_mapf_training.py train --seed 42 --epochs 100

# Step 3: Run main comparison experiments
Write-Host ""
Write-Host "[3/8] Running main comparison experiments..."
python learning_guided_mapf_comparison.py --output results/main

# Step 4: Run ablation experiments
Write-Host ""
Write-Host "[4/8] Running ablation experiments..."
python learning_guided_mapf_experiments.py ablation --output results/ablation

# Step 5: Run generalization experiments
Write-Host ""
Write-Host "[5/8] Running generalization experiments..."
python learning_guided_mapf_experiments.py generalization --output results/generalization

# Step 6: Run scalability experiments
Write-Host ""
Write-Host "[6/8] Running scalability experiments..."
python learning_guided_mapf_experiments.py scalability --output results/scalability

# Step 7: Generate figures
Write-Host ""
Write-Host "[7/8] Generating paper figures..."
python learning_guided_mapf_paper_figures.py --output figures

# Step 8: Generate LaTeX tables
Write-Host ""
Write-Host "[8/8] Generating LaTeX tables..."
python learning_guided_mapf_latex_tables.py --output latex_tables

Write-Host ""
Write-Host "====================================="
Write-Host "All experiments completed!"
Write-Host "Results saved in: ./results/"
Write-Host "Figures saved in: ./figures/"
Write-Host "Tables saved in: ./latex_tables/"
Write-Host "====================================="
'''
        return script
    
    def generate_config_files(self):
        """生成配置文件"""
        # 默认配置
        default_config = {
            "experiment": {
                "seed": 42,
                "num_trials": 25,
                "timeout": 300
            },
            "model": {
                "gnn": {
                    "layers": 3,
                    "hidden_dim": 128,
                    "aggregation": "mean"
                },
                "transformer": {
                    "heads": 4,
                    "layers": 2,
                    "ff_dim": 256,
                    "dropout": 0.1
                }
            },
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "num_epochs": 100,
                "weight_decay": 0.00001,
                "early_stopping_patience": 10
            },
            "data": {
                "train_maps": 100,
                "val_maps": 20,
                "test_maps": 50,
                "agents_min": 10,
                "agents_max": 200,
                "map_types": ["random", "room", "maze", "warehouse", "open"]
            }
        }
        
        return json.dumps(default_config, indent=2)
    
    def generate_readme(self) -> str:
        """生成复现性说明文档"""
        readme = """# Reproducibility Package

This package contains all code and configurations needed to reproduce the experiments
in our paper "Learning-Guided Conflict-Based Search for Multi-Agent Path Finding".

## Quick Start

### 1. Environment Setup

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate lg-mapf
```

Using pip:
```bash
pip install -r requirements.txt
```

### 2. Run All Experiments

Linux/macOS:
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Windows:
```powershell
.\run_all_experiments.ps1
```

### 3. Individual Experiments

Train model:
```bash
python learning_guided_mapf_training.py train --config config/default.json
```

Run comparison:
```bash
python learning_guided_mapf_comparison.py --agents 100 --trials 25
```

Run ablation:
```bash
python learning_guided_mapf_experiments.py ablation
```

## File Structure

```
reproducibility/
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
├── run_all_experiments.sh    # Linux/macOS script
├── run_all_experiments.ps1   # Windows script
├── config/
│   └── default.json          # Default experiment config
├── results/                  # Experiment results (generated)
├── figures/                  # Paper figures (generated)
└── latex_tables/             # LaTeX tables (generated)
```

## Expected Results

| Experiment | Expected Runtime | Output |
|------------|-----------------|--------|
| Data Generation | ~10 min | data/train, data/val, data/test |
| Model Training | ~4 hours | models/best_model.pt |
| Main Comparison | ~2 hours | results/main/*.json |
| Ablation Study | ~3 hours | results/ablation/*.json |
| Generalization | ~2 hours | results/generalization/*.json |
| Scalability | ~1 hour | results/scalability/*.json |

Total time: ~12 hours on a single NVIDIA RTX 3090

## Hardware Requirements

- GPU: NVIDIA GPU with 12GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 32GB+
- Storage: 10GB for data and results

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```json
"training": {"batch_size": 16}
```

### Slow Training
Enable mixed precision:
```bash
python learning_guided_mapf_training.py train --fp16
```

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{lg-cbs2026,
  title={Learning-Guided Conflict-Based Search for Multi-Agent Path Finding},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

## Contact

For questions, please open an issue on GitHub.
"""
        return readme
    
    def generate_all(self):
        """生成所有复现性文件"""
        # requirements.txt
        with open(self.output_dir / 'requirements.txt', 'w') as f:
            f.write(self.generate_requirements())
        print("✓ 生成 requirements.txt")
        
        # environment.yml
        with open(self.output_dir / 'environment.yml', 'w') as f:
            f.write(self.generate_environment_yml())
        print("✓ 生成 environment.yml")
        
        # run scripts
        with open(self.output_dir / 'run_all_experiments.sh', 'w') as f:
            f.write(self.generate_run_all_script())
        print("✓ 生成 run_all_experiments.sh")
        
        with open(self.output_dir / 'run_all_experiments.ps1', 'w') as f:
            f.write(self.generate_run_all_ps1())
        print("✓ 生成 run_all_experiments.ps1")
        
        # config
        config_dir = self.output_dir / 'config'
        config_dir.mkdir(exist_ok=True)
        with open(config_dir / 'default.json', 'w') as f:
            f.write(self.generate_config_files())
        print("✓ 生成 config/default.json")
        
        # README
        with open(self.output_dir / 'README.md', 'w') as f:
            f.write(self.generate_readme())
        print("✓ 生成 README.md")
        
        print(f"\n✅ 复现性包生成完成！保存在 {self.output_dir}")


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        self.logs = []
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        entry = {
            "timestamp": timestamp,
            "elapsed": f"{elapsed:.2f}s",
            "level": level,
            "message": message
        }
        self.logs.append(entry)
        print(f"[{timestamp}] [{level}] {message}")
    
    def save(self, filename: str = "experiment.log"):
        """保存日志"""
        with open(self.log_dir / filename, 'w') as f:
            for entry in self.logs:
                f.write(f"[{entry['timestamp']}] [{entry['level']}] [{entry['elapsed']}] {entry['message']}\n")


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, state: dict, name: str, epoch: int):
        """保存检查点"""
        import torch
        
        checkpoint = {
            'epoch': epoch,
            'state': state,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        path = self.checkpoint_dir / f'{name}_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"✓ 保存检查点: {path}")
        
        # 同时保存最新检查点
        latest_path = self.checkpoint_dir / f'{name}_latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, name: str, epoch: Optional[int] = None):
        """加载检查点"""
        import torch
        
        if epoch is None:
            path = self.checkpoint_dir / f'{name}_latest.pt'
        else:
            path = self.checkpoint_dir / f'{name}_epoch_{epoch}.pt'
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path)
        print(f"✓ 加载检查点: {path} (epoch {checkpoint['epoch']})")
        return checkpoint


def main():
    manager = ReproducibilityManager(output_dir='./reproducibility')
    manager.generate_all()


if __name__ == '__main__':
    main()
