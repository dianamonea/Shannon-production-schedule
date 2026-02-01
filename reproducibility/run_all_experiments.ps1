# Learning-Guided MAPF - Run All Experiments (Windows)
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
