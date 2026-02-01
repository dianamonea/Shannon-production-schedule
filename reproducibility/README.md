# Reproducibility Package

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
.un_all_experiments.ps1
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
©À©¤©¤ requirements.txt          # Python dependencies
©À©¤©¤ environment.yml           # Conda environment
©À©¤©¤ run_all_experiments.sh    # Linux/macOS script
©À©¤©¤ run_all_experiments.ps1   # Windows script
©À©¤©¤ config/
©¦   ©¸©¤©¤ default.json          # Default experiment config
©À©¤©¤ results/                  # Experiment results (generated)
©À©¤©¤ figures/                  # Paper figures (generated)
©¸©¤©¤ latex_tables/             # LaTeX tables (generated)
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
