import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1) 手动录入三组结果（你贴的）
# =========================
results = {
    "w/o gene": {
        "mIoU_matched": 0.8459795168198145,
        "f1": 0.8064335958685112,
        "PQ": 0.6822263037801086,
        "recall": 0.6774974498469908,
        "precision": 0.9959811252849202,
    },
    "w/o image": {
        "mIoU_matched": 0.8918475175000541,
        "f1": 0.8457947488371699,
        "PQ": 0.7543199470650117,
        "recall": 0.7338498632714869,
        "precision": 0.9980405094675384,
    },
    "w/o MoE": {
        "mIoU_matched": 0.8076635851822795,
        "f1": 0.7102480451556107,
        "PQ": 0.5736414825190862,
        "recall": 0.5538792059056327,
        "precision": 0.9896386515816209,
    },
    "CellWeave": {
            "mIoU_matched": 0.9231470763018222,
            "f1": 0.8929924919556668,
            "PQ": 0.8243634081083523,
            "recall": 0.8074417702490827,
            "precision": 0.9988203067202527,
        },
}

# 你要求 IoU 必选：这里用 matched mIoU（实例 IoU）
metrics_to_plot = [
    ("mIoU_matched", "mIoU (matched)"),
    ("f1", "F1"),
    ("PQ", "PQ"),
]

colors = [
    "#D77A7A",  # dusty rose
    "#6BA6A9",  # soft cyan-teal
    "#9BB7A0",  # muted green
    "#8C88C6",  # muted lavender
]



methods = list(results.keys())
x = np.arange(len(methods))

# =========================
# 2) 画一个“共同图”：1 行 3 列小面板（Nature 论文常用）
# =========================
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), dpi=200)

for ax, (k, title) in zip(axes, metrics_to_plot):
    vals = [results[m][k] for m in methods]

    bars = ax.bar(x, vals, width=0.65, color=colors)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)

    # y 轴范围：给一点 headroom，方便看差异和标注
    vmin, vmax = min(vals), max(vals)
    pad = max(0.01, (vmax - vmin) * 0.35)
    ax.set_ylim(max(0, vmin - pad), min(1.0, vmax + pad))

    # Nature 常见的“干净轴”风格：去掉上/右边框，y 轴细一点
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=9, width=0.8, length=3)
    ax.tick_params(axis="x", width=0.8, length=0)

    # 数值标注（保留 3 位小数）
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# 总标题（可选）
fig.suptitle("Segmentation performance @ IoU≥0.3 (matched 1-to-1)", fontsize=12, y=1.05)

plt.tight_layout()
plt.savefig('<OUTPUT_FIGURE_FILE>', format='pdf', bbox_inches='tight', dpi=600)
plt.show()

# 如果你想保存（可取消注释）
# fig.savefig("<OUTPUT_FIGURE_FILE>", bbox_inches="tight", dpi=600)
