import pandas as pd
import numpy as np
import matplotlib as mpl

# 避免 PDF 中出现 Type 3 字体
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns

from Draw_chat_2 import rounds


def render_chat():
    sheet1_df = pd.read_excel("converted_table.xlsx", sheet_name=0)
    sheet2_df = pd.read_excel("converted_table.xlsx", sheet_name=1)

    categories = sheet1_df.columns[1:]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors_rgba = [to_rgba('#FFA500', 0.8), to_rgba('green', 0.8), to_rgba('blue', 0.8)]
    fill_rgba = [to_rgba('#FFA500', 0.2), to_rgba('green', 0.2), to_rgba('blue', 0.2)]
    line_styles = ['solid', 'dashdot', 'dashed']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.set_facecolor('#f2f2f2')
    for i, row in sheet1_df.iterrows():
        values = row[1:].tolist() + [row[1]]
        ax.plot(angles, values, color=colors_rgba[i], linewidth=2.5,
                linestyle=line_styles[i], label=row['Method'])
        ax.fill(angles, values, color=fill_rgba[i])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9, fontweight='bold')
    ax.set_ylim(6, 8)
    ax.set_title("A. Human Evaluation", size=14, fontweight='bold', pad=20)
    ax.yaxis.set_ticks([6.0, 6.5, 7.0, 7.5, 8.0])
    ax.yaxis.grid(True, color='lightgray', linestyle='dashed', linewidth=0.8)
    ax.xaxis.grid(True, color='white', linewidth=1.2)
    ax.tick_params(colors='dimgray')

    ax = axes[1]
    ax.set_facecolor('#f2f2f2')
    for i, row in sheet2_df.iterrows():
        values = row[1:].tolist() + [row[1]]
        ax.plot(angles, values, color=colors_rgba[i], linewidth=2.5,
                linestyle=line_styles[i], label=row['Method'])
        ax.fill(angles, values, color=fill_rgba[i])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9, fontweight='bold')
    ax.set_ylim(6, 9)
    ax.set_title("2. Automatic Evaluation", size=14, fontweight='bold', pad=20)
    ax.yaxis.set_ticks([6.0, 6.75, 7.5, 8.25, 9.0])
    ax.yaxis.grid(True, color='lightgray', linestyle='dashed', linewidth=0.8)
    ax.xaxis.grid(True, color='white', linewidth=1.2)
    ax.tick_params(colors='dimgray')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.04),
               ncol=3,
               frameon=False,
               fontsize=11)

    fig.subplots_adjust(top=0.88, bottom=0.1, wspace=0.3)
    plt.savefig("evaluation_radar.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def round_chat1():
    methods = ["MUSE \n(one-round)", "MUSE \n(two-round)", "MUSEN \n(three-round)",
               "Ordinary \nWriter", "Expert Writer"]
    times = [78, 105, 149, 124, 172]
    colors_a = ['#CB9475', '#CC7C71', '#8D2F25', '#8CBF87', '#3E608D']

    rounds_local = [1, 2, 3, 4, 5, 6]
    new_issues = [32, 19, 16, 8, 6, 6]
    new_suggestions = [8, 5, 3, 2, 0.8, 0]
    cumulative_suggestions = np.cumsum(new_suggestions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=200)

    bars = ax1.bar(methods, times, color=colors_a, width=0.5, edgecolor='black', linewidth=0.9)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 4, f"{height} min",
                 ha='center', va='bottom', fontsize=16)

    ax1.set_title("a. Average Writing Time", fontsize=20, pad=10)
    ax1.set_ylabel("Time (minutes)", fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_ylim(0, max(times) + 30)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    ax2.plot(rounds_local, new_issues, marker='o', label='New Issues (per round)', color='#D55E00')
    ax2.plot(rounds_local, new_suggestions, marker='s', label='New Suggestions (per round)', color='#0072B2')
    ax2.plot(rounds_local, cumulative_suggestions, linestyle='--', marker='s',
             label='Cumulative Suggestions', color='#009E73')

    ax2.set_title("b. Suggestions and Issues Across Rounds", fontsize=20, pad=10)
    ax2.set_xlabel("Revision Round", fontsize=16)
    ax2.set_ylabel("Count", fontsize=16)
    ax2.set_xticks(rounds_local)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend(fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("writing_analysis.pdf", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == '__main__':
    round_chat1()