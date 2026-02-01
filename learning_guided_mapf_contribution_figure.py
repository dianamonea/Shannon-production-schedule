"""
Learning-Guided MAPF 论文贡献图示生成器
Generates a contribution diagram (Figure 0)

输出: paper_figures/figure0_contributions.(png|pdf)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, xy, text, width=0.32, height=0.18, color="#667eea"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=2, edgecolor=color, facecolor="#f5f7ff"
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=11, color="#2c3e50"
    )


def add_arrow(ax, start, end, color="#667eea"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="->", mutation_scale=14,
        linewidth=2, color=color
    )
    ax.add_patch(arrow)


def generate(output_dir="paper_figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Boxes
    add_box(ax, (0.05, 0.4), "Conflict Graph\nEncoding (GNN)")
    add_box(ax, (0.36, 0.4), "Priority/Difficulty/\nScope Ranking (Transformer)")
    add_box(ax, (0.70, 0.4), "Learning-Guided\nCBS Search")

    # Arrows
    add_arrow(ax, (0.37, 0.49), (0.36, 0.49))
    add_arrow(ax, (0.68, 0.49), (0.70, 0.49))

    # Feedback arrow
    feedback = FancyArrowPatch(
        (0.86, 0.58), (0.14, 0.58),
        arrowstyle="->", mutation_scale=14,
        linewidth=1.8, color="#10b981"
    )
    ax.add_patch(feedback)
    ax.text(0.5, 0.62, "Search Feedback (Learning Signal)", ha="center", va="center",
            fontsize=10, color="#10b981")

    # Title
    ax.text(0.5, 0.9, "Figure 0: Core Contributions Overview",
            ha="center", va="center", fontsize=12, fontweight="bold")

    png_path = output_dir / "figure0_contributions.png"
    pdf_path = output_dir / "figure0_contributions.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("✓ 贡献图示生成完成:", png_path, pdf_path)


def main():
    generate()


if __name__ == "__main__":
    main()
