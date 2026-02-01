# Learning-Guided MAPF 项目索引
# Project Index for Top-tier Conference Paper (NeurIPS/CoRL/ICML 2026)

## 📋 项目概述 (Project Overview)

本项目实现了 **Learning-Guided Conflict-Based Search for Large-Scale Multi-Agent Path Finding**，
使用 GNN + Transformer 学习冲突模式来加速 CBS 搜索。

---

## ✨ 论文贡献小节 (Contributions)

**Contribution 1 — 冲突图学习表示**
- 将冲突建模为图结构，设计 `ConflictGraphEncoder` 对冲突对进行表征学习，输出可解性/难度类别。
- 关键位置：[learning_guided_mapf.py](learning_guided_mapf.py#L145-L240)

**Contribution 2 — 三头冲突优先级预测**
- 设计 `ConflictPriorityTransformer` 同时预测优先级、解决难度与影响范围，用于冲突排序与搜索指导。
- 关键位置：[learning_guided_mapf.py](learning_guided_mapf.py#L247-L330)

**Contribution 3 — 学习引导CBS集成**
- 在CBS搜索中引入学习模型驱动的冲突选择策略与启发式流程，实现稳定加速并保持最优性。
- 关键位置：[learning_guided_mapf.py](learning_guided_mapf.py#L350-L608)

**Contribution 4 — 论文级可视化与复现流水线**
- 提供论文级图表、表格、案例研究与可视化仪表板，覆盖顶会投稿常规实验需求。
- 关键位置：[learning_guided_mapf_paper_figures.py](learning_guided_mapf_paper_figures.py), [generate_experiment_dashboard.py](generate_experiment_dashboard.py)

**论文贡献图示（Figure 0）**
- 输出：paper_figures/figure0_contributions.pdf


## 📁 文件结构 (File Structure)

### 🧠 核心算法实现 (Core Algorithm)

| 文件 | 描述 | 行数 |
|------|------|------|
| `learning_guided_mapf.py` | 主算法：GNN编码器 + Transformer排序器 + CBS集成 | ~500 |
| `learning_guided_mapf_training.py` | 模型训练框架和数据生成 | ~600 |

### 📊 实验代码 (Experiments)

| 文件 | 描述 | 功能 |
|------|------|------|
| `learning_guided_mapf_experiments.py` | 完整实验套件 | 消融实验、统计检验、泛化测试、可扩展性分析 |
| `learning_guided_mapf_comparison.py` | 方法对比框架 | 与SOTA方法对比 |
| `learning_guided_mapf_tests.py` | 单元测试 | 代码正确性验证 |

### 📈 可视化代码 (Visualization)

| 文件 | 描述 | 输出 |
|------|------|------|
| `learning_guided_mapf_visualization.py` | 训练可视化 | 训练曲线、注意力图、路径图 |
| `learning_guided_mapf_paper_figures.py` | 论文图表 | 6张出版级PDF图 |
| `reports/mapf_comparison_visualization.html` | 交互式仪表板 | HTML可视化 |

### 📝 论文材料 (Paper Materials)

| 文件 | 描述 | 格式 |
|------|------|------|
| `learning_guided_mapf_latex_tables.py` | LaTeX表格生成 | 7个表格 |
| `learning_guided_mapf_case_studies.py` | 案例研究 | 5个定性分析案例 |

### 📦 数据与复现 (Data & Reproducibility)

| 文件 | 描述 |
|------|------|
| `learning_guided_mapf_dataset.py` | 数据集生成（MovingAI格式） |
| `learning_guided_mapf_reproducibility.py` | 复现性包（requirements、脚本、配置） |

---

## 🚀 快速开始 (Quick Start)

### 1. 环境配置
```bash
# 创建conda环境
conda create -n lg-mapf python=3.9
conda activate lg-mapf

# 安装依赖
pip install torch torch-geometric numpy scipy matplotlib seaborn
```

### 2. 生成数据
```bash
python learning_guided_mapf_dataset.py
```

### 3. 训练模型
```bash
python learning_guided_mapf_training.py
```

### 4. 运行实验
```bash
python learning_guided_mapf_experiments.py
```

### 5. 生成论文材料
```bash
# 生成图表
python learning_guided_mapf_paper_figures.py

# 生成表格
python learning_guided_mapf_latex_tables.py

# 生成案例研究
python learning_guided_mapf_case_studies.py
```

---

## 📊 论文图表清单 (Paper Figures)

| 图号 | 内容 | 文件 |
|------|------|------|
| Figure 0 | Contributions Overview | `paper_figures/figure0_contributions.pdf` |
| Figure 1 | Method Overview | `paper_figures/figure1_method_overview.pdf` |
| Figure 2 | Architecture | `paper_figures/figure2_architecture.pdf` |
| Figure 3 | Main Results | `paper_figures/figure3_main_results.pdf` |
| Figure 4 | Ablation Study | `paper_figures/figure4_ablation.pdf` |
| Figure 5 | Generalization | `paper_figures/figure5_generalization.pdf` |
| Figure 6 | Qualitative | `paper_figures/figure6_qualitative.pdf` |

## 📋 论文表格清单 (Paper Tables)

| 表号 | 内容 | 文件 |
|------|------|------|
| Table 1 | Main Comparison | `latex_tables/table1_main_comparison.tex` |
| Table 2 | Ablation Study | `latex_tables/table2_ablation.tex` |
| Table 3 | Map Types | `latex_tables/table3_map_types.tex` |
| Table 4 | Statistical Tests | `latex_tables/table4_statistical_tests.tex` |
| Table 5 | Hyperparameters | `latex_tables/table5_hyperparameters.tex` |
| Table 6 | Computational Cost | `latex_tables/table6_computational.tex` |
| Table 7 | Generalization | `latex_tables/table7_generalization.tex` |

---

## ✅ 顶会论文检查清单 (Checklist for Top-tier Venues)

### 实验 (Experiments)
- [x] 与SOTA方法对比（10种方法）
- [x] 消融实验（9种变体）
- [x] 统计显著性检验（t-test, Wilcoxon, Cohen's d）
- [x] 泛化性实验（跨地图、跨规模）
- [x] 可扩展性测试
- [x] 计算资源分析
- [x] 失败案例分析
- [x] 标准Benchmark测试（MovingAI格式）

### 可视化 (Visualization)
- [x] 训练曲线
- [x] 注意力权重可视化
- [x] 路径可视化
- [x] 搜索树对比
- [x] 冲突热力图

### 复现性 (Reproducibility)
- [x] 随机种子固定
- [x] requirements.txt
- [x] 配置文件
- [x] 运行脚本
- [x] 详细README

### 代码质量 (Code Quality)
- [x] 单元测试
- [x] 类型注解
- [x] 详细注释
- [x] 模块化设计

---

## 📚 参考文献 (Key References)

1. **EECBS** - Li et al., AAAI 2021
2. **MAPF-LNS2** - Li et al., AAAI 2022
3. **LaCAM** - Okumura, AAAI 2023
4. **LaCAM*** - Okumura, AAAI 2024
5. **Learning to Resolve Conflicts** - Huang et al., AAAI 2023
6. **MAGAT** - Li et al., RA-L 2022
7. **SCRIMP** - Wang et al., ICRA 2024
8. **MAPF-GPT** - Andreychuk et al., arXiv 2024

---

## 📞 联系方式 (Contact)

如有问题，请创建 GitHub Issue。

---

**Last Updated:** 2025-02-01

**Target Venues:** NeurIPS 2026 / CoRL 2026 / ICML 2026
