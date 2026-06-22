from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


results_dir = Path("ana/results_20260331")
file_paths = sorted(results_dir.glob("s*/summary_thresholds.csv"))

# ターゲット条件
target_layer = "conv1"
target_ecc = "ecc12"
max_r = 150

# CSVの component 列と direction 列に対応する表示順
conditions_order = [
    ("component0", "plus"),
    ("component1", "plus"),
    ("component2", "plus"),
    ("component0", "minus"),
    ("component1", "minus"),
    ("component2", "minus"),
]
labels = ["comp0+", "comp1+", "comp2+", "comp0-", "comp1-", "comp2-"]

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 多角形を閉じる

# プロット初期化
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 同心円状のグリッド
ax.set_rgrids([])
r_range = np.linspace(0, max_r, 6)
for i, r in enumerate(r_range):
    ax.plot(
        angles,
        [r] * (num_vars + 1),
        color="gray",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )
    if i in (2, 4):
        ax.text(
            np.pi / 2,
            r * 0.9,
            f"{r:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
            zorder=100,
        )

# 被験者ごとの閾値をプロット
required_columns = {"ecc", "layer", "component", "direction", "threshold"}

for file_path in file_paths:
    df = pd.read_csv(file_path)

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{file_path}: 必要な列がありません: {sorted(missing_columns)}"
        )

    # summary_thresholds.csv の親ディレクトリ名（s00など）
    subject = file_path.parent.name

    target_df = df.loc[
        (df["layer"] == target_layer) & (df["ecc"] == target_ecc),
        ["component", "direction", "threshold"],
    ]

    if target_df.duplicated(["component", "direction"]).any():
        raise ValueError(
            f"{file_path}: {target_layer}/{target_ecc} に重複条件があります"
        )

    threshold_by_condition = target_df.set_index(
        ["component", "direction"]
    )["threshold"]

    missing_conditions = [
        condition
        for condition in conditions_order
        if condition not in threshold_by_condition.index
    ]
    if missing_conditions:
        print(f"skip {subject}: missing {missing_conditions}")
        continue

    values = [
        float(threshold_by_condition.loc[condition])
        for condition in conditions_order
    ]
    values += values[:1]  # 多角形を閉じる

    print(subject, values)
    ax.plot(angles, values, label=subject, linewidth=3, zorder=4)
    ax.fill(angles, values, alpha=0.1, zorder=2)

# 軸ラベル
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=24)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("center")
    label.set_verticalalignment("center_baseline")
    label.set_y(label.get_position()[1] - 0.1)

# ax.set_title(f"{target_layer} eccentricity=8", fontsize=28, pad=60)
# ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=20)
ax.set_yticklabels([])
ax.set_ylim(0, max_r)
ax.set_frame_on(False)
ax.spines["polar"].set_visible(False)

plt.tight_layout()
output_path = Path("output/ana") / (
    f"radar_chart_{target_layer}_{target_ecc}.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {output_path}")
