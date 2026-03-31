import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

file_paths = [
    "ana/results/s00.csv",
    "ana/results/s01.csv",
    "ana/results/s02.csv",
    "ana/results/s03.csv",
    "ana/results/s04.csv",
    "ana/results/s05.csv",
    "ana/results/s06.csv",
    "ana/results/s07.csv",
]

# Target condition
target_layer = "layer3"
target_ecc = "ecc8"
max_r = 1.2

components_order = [
    "component0_plus", "component1_plus", "component2_plus",
    "component0_minus", "component1_minus", "component2_minus"
]

# Labels
labels = [
    "comp0+", "comp1+", "comp2+",
    "comp0-", "comp1-", "comp2-"
]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_rgrids([])  # Hide the default grid
r_range = np.linspace(0, max_r, 6)
for r in r_range:
    grid_values = [r] * (num_vars + 1)
    
    ax.plot(angles, grid_values, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    if r == r_range[2] or r == r_range[4]:
        ax.text(np.pi / 2, r*0.9, f"{r}", ha="center", va="bottom", fontsize=20, zorder=100, rotation=0)

# Process each file
for file_path in file_paths:
    df = pd.read_csv(file_path)
    subject = Path(file_path).stem
    values = []
    for comp in components_order:
        query = df[
            df["name_cond"].str.contains(target_layer) &
            df["name_cond"].str.contains(target_ecc) &
            df["name_cond"].str.contains(comp)
        ]
        if not query.empty:
            values.append(query.iloc[0]["mean_threshold"])
        else:
            values.append(None)
    if None in values:
        continue
    values += values[:1]  # Close the polygon
    print(values)
    ax.plot(angles, values, label=subject, linewidth=3, zorder=4)
    ax.fill(angles, values, alpha=0.1, zorder=2)

# Axis labels and legend
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=24)
for label, angle in zip(ax.get_xticklabels(), angles):
    label.set_horizontalalignment('center')
    label.set_verticalalignment('center_baseline')
    label.set_y(label.get_position()[1] - 0.1)  # Move slightly outward

# ax.set_title(f"{target_layer} eccentricity=8", fontsize=28, pad=60)
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=20)
# ax.tick_params(axis='y', labelsize=24)
ax.set_yticklabels([])

ax.set_ylim(0, max_r)
ax.set_frame_on(False)
ax.spines["polar"].set_visible(False)

plt.tight_layout()
plt.savefig(f"output/ana/radar_chart_{target_layer}_{target_ecc}.png", dpi=300)
