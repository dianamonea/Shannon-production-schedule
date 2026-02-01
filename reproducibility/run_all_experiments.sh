#!/bin/bash
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
