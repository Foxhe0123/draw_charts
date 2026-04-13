import matplotlib as mpl

# ===== 关键：避免导出 PDF 时出现 Type 3 字体 =====
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"

# ===== 尽量使用 Times 风格字体，贴近 IEEE 模板 =====
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif"
]

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# Step 1: Prepare data
# ==========================================
data = {
    "Method": ["MUSE", "Human-written", "LLM-generated"],
    "Topic Relevance": [7.767, 7.2, 7.4],
    "Structural \nCoherence": [6.833, 6.3, 6.433],
    "Content Richness": [7.267, 7.033, 6.67],
    "Literary Quality": [6.8, 6.7, 6.2],
    "Reliability": [7.267, 7.233, 6.733]
}
df = pd.DataFrame(data)

# ==========================================
# Step 2: Canvas setup
# ==========================================
categories = list(df.columns[1:])
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(polar=True))
plt.subplots_adjust(bottom=0.18)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# ==========================================
# Step 3: Draw radar chart
# ==========================================
styles = [
    {"color": "#DC143C", "ls": "-",  "marker": None},
    {"color": "#4682B4", "ls": "-.", "marker": None},
    {"color": "#FF8C00", "ls": "--", "marker": None},
]

for i, row in df.iterrows():
    values = row[1:].tolist()
    values += values[:1]
    style = styles[i % len(styles)]

    ax.plot(
        angles,
        values,
        linewidth=2.0,
        linestyle=style["ls"],
        color=style["color"],
        label=row["Method"]
    )
    ax.fill(
        angles,
        values,
        color=style["color"],
        alpha=0.12
    )

# ==========================================
# Step 4: Beautify
# ==========================================
ax.set_xticks(angles[:-1])
ax.set_xticklabels(
    categories,
    fontsize=14,
    fontweight="bold",
    color="#333333"
)
ax.tick_params(axis="x", pad=12)

ax.set_ylim(5.5, 8.2)
ax.set_yticks([6.0, 6.5, 7.0, 7.5, 8.0])
ax.set_yticklabels(
    ["6.0", "6.5", "7.0", "7.5", "8.0"],
    fontsize=10,
    color="dimgray"
)
ax.set_rlabel_position(25)

ax.grid(color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
ax.spines["polar"].set_visible(False)

# 如果你想加标题，可以取消下面这行注释
# ax.set_title("Human Evaluation", fontsize=16, fontweight="bold", pad=18)

# ==========================================
# Step 5: Legend
# ==========================================
legend = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=False,
    fontsize=14
)
plt.setp(legend.get_texts(), fontweight="bold")

# ==========================================
# Step 6: Save
# ==========================================
# 优先保存为 PDF（矢量图）
plt.savefig("human_evaluation.pdf", format="pdf", bbox_inches="tight")

# 保险起见，也可以同时导出高分辨率 PNG
# plt.savefig("human_evaluation.png", dpi=600, bbox_inches="tight")

plt.show()