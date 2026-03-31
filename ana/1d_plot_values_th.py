import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from abx_app.AttackCNN.utils.condition import Condition

if __name__ == "__main__":
    layers = ["conv1", "layer3", "avgpool"]

    results_by_subject_layer = {}

    # Load and aggregate data
    subjects = [
        "s00",
        "s01",
        "s02",
        "s03",
        "s04",
        "s05",
        "s06",
        "s07",
    ]
    for subject in subjects:
        path_csv = Path(f"ana/results/{subject}.csv")
        data = pd.read_csv(path_csv)
        data["Condition"] = data["name_cond"].apply(Condition.from_string)  # type: ignore

        for _, row in data.iterrows():
            cond: Condition = row["Condition"]
            key = (cond.layer, cond.component, cond.direction)
            if key not in results_by_subject_layer:
                results_by_subject_layer[key] = {"ecc": [], "mean_threshold": []}
            results_by_subject_layer[key]["ecc"].append(cond.ecc)
            results_by_subject_layer[key]["mean_threshold"].append(row["mean_threshold"])

    # Compute overall mean and SD across subjects
    overall_means = {layer: {} for layer in layers}
    overall_sds = {layer: {} for layer in layers}
    for layer in layers:
        for ecc in np.unique(
            np.concatenate(
                [
                    np.array(
                        [
                            results_by_subject_layer[(layer, comp, dir)]["ecc"]
                            for comp in range(3)
                            for dir in ["plus", "minus"]
                        ]
                    ).flatten()
                ]
            )
        ):
            all_values = []
            for comp in range(3):
                for dir in ["plus", "minus"]:
                    key = (layer, comp, dir)
                    if key in results_by_subject_layer:
                        values = np.array(results_by_subject_layer[key]["mean_threshold"])[
                            np.array(results_by_subject_layer[key]["ecc"]) == ecc
                        ]
                        all_values.extend(values)
            all_values = np.array(all_values)
            if all_values.size > 0:
                overall_means[layer][ecc] = np.mean(all_values)
                overall_sds[layer][ecc] = np.std(all_values)

    # Define theoretical bounds
    theoretical_bounds = {
        "conv1": {"min": (1 * 3 + 11 * 2) / 5, "max": (10**2.5 * 3 + (10**2.5 - 10) * 2) / 5},
        "layer3": {"min": (0.01 * 3 + 0.31 * 2) / 5, "max": (10**0.5 * 3 + (10**0.5 - 0.3) * 2) / 5},
        "avgpool": {"min": (0.01 * 3 + 0.03 * 2) / 5, "max": (10**0.5 * 3 + (10**0.5 - 0.02) * 2) / 5},
    }

    # Create plot for overall means with theoretical bounds
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_colors = {"conv1": "red", "layer3": "blue", "avgpool": "green"}
    markers = {4: "o", 8: "s", 12: "^"}

    x_positions = {"conv1": 1, "layer3": 2, "avgpool": 3}
    x_labels = ["conv1", "layer3", "avgpool"]
    ecc_shifts = {4: -0.06, 8: 0.0, 12: 0.06}

    for ecc in [4, 8, 12]:
        for layer in layers:
            x = x_positions[layer] + ecc_shifts[ecc]
            y = overall_means[layer].get(ecc, np.nan)
            error = overall_sds[layer].get(ecc, 0)
            ax.errorbar(
                x,
                y,
                yerr=error,
                color=layer_colors[layer],
                marker=markers[ecc],
                markersize=8,
                linestyle="--",
                capsize=6,
            )

    # Add theoretical bounds as horizontal lines
    for layer, bounds in theoretical_bounds.items():
        x = x_positions[layer]
        ax.axhline(y=bounds["min"], color=layer_colors[layer], linestyle="dotted")
        ax.axhline(y=bounds["max"], color=layer_colors[layer], linestyle="dashdot")

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(x_labels, fontsize=32)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Threshold", fontsize=36)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.grid(True)

    # Simplified legend
    legend_elements = [
        # Line2D([0], [0], color="red", lw=2, label="Conv1"),
        # Line2D([0], [0], color="blue", lw=2, label="Layer3"),
        # Line2D([0], [0], color="green", lw=2, label="Avgpool"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10, label="Ecc=4"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=10, label="Ecc=8"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=10, label="Ecc=12"),
        Line2D([0], [0], color="gray", linestyle="dotted", lw=2, label="Conditional Min"),
        Line2D([0], [0], color="gray", linestyle="dashdot", lw=2, label="Conditional Max"),
    ]
    ax.legend(handles=legend_elements, fontsize=21)

    plt.tight_layout()
    plt.savefig("output/ana/plot_values/overall_with_theoretical_bounds.png")
    plt.close()
