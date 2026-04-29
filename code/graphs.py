import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


NORMAL_COLOR = "#4C78A8"
ATTACK_COLOR = "#F58518"


def make_plots(data, out_folder, target):
    label_counts = data[target].value_counts().sort_index()
    ax = label_counts.plot(kind="bar", color=[NORMAL_COLOR, ATTACK_COLOR])
    ax.set_title("Normal Traffic vs Attack Traffic")
    ax.set_xlabel("Traffic Type")
    ax.set_ylabel("Rows")
    ax.set_xticklabels(["Normal", "Attack"], rotation=0)
    add_labels(ax)
    save_plot(out_folder, "label_counts.png")

    attack_counts = data["attack_cat"].fillna("Missing").value_counts().head(10)
    ax = attack_counts.plot(kind="bar", color=ATTACK_COLOR)
    ax.set_title("Attack Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=35)
    save_plot(out_folder, "attack_categories.png")

    proto = pd.crosstab(data["proto"], data[target])
    proto = proto.sort_values(1, ascending=False).head(8)
    ax = proto.plot(kind="bar", color=[NORMAL_COLOR, ATTACK_COLOR])
    ax.set_title("Protocols Used by Normal and Attack Traffic")
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Rows")
    ax.legend(["Normal", "Attack"])
    ax.tick_params(axis="x", rotation=0)
    save_plot(out_folder, "protocol_counts.png")

    make_box(data, target, "dur", "Connection Duration by Traffic Type", out_folder, "duration_by_label.png")

    data = data.copy()
    data["total_bytes"] = data["sbytes"] + data["dbytes"]
    make_box(data, target, "total_bytes", "Total Bytes by Traffic Type", out_folder, "bytes_by_label.png")

    make_heatmap(data, out_folder, target)


def add_labels(ax):
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_box(data, target, column, title, out_folder, file_name):
    normal = data.loc[data[target] == 0, column].dropna()
    attack = data.loc[data[target] == 1, column].dropna()

    box = plt.boxplot(
        [normal, attack],
        tick_labels=["Normal", "Attack"],
        showfliers=False,
        patch_artist=True,
    )
    box["boxes"][0].set_facecolor(NORMAL_COLOR)
    box["boxes"][1].set_facecolor(ATTACK_COLOR)
    plt.yscale("log")
    plt.title(title)
    plt.ylabel(column + " (log scale)")
    save_plot(out_folder, file_name)


def make_heatmap(data, out_folder, target):
    numbers = data.select_dtypes(include="number").drop(columns=["id"], errors="ignore")
    corr = numbers.corr(numeric_only=True)
    top_cols = corr[target].abs().sort_values(ascending=False).head(12).index
    small_corr = corr.loc[top_cols, top_cols]

    plt.figure(figsize=(9, 7))
    plt.imshow(small_corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(top_cols)), top_cols, rotation=45, ha="right")
    plt.yticks(range(len(top_cols)), top_cols)
    plt.title("Strongest Correlations")
    save_plot(out_folder, "correlation_heatmap.png")


def save_plot(out_folder, file_name):
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, file_name), dpi=140)
    plt.close()
