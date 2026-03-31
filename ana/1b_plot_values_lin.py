import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from abx_app.AttackCNN.utils.condition import Condition

if __name__ == "__main__":
    subjects = ["s00", "s01", "s02"]
    layers = ["conv1", "layer3", "avgpool"]

    results_by_subject_layer = {}

    for subject in subjects:
        path_csv = Path(f"ana/results/{subject}.csv")
        data = pd.read_csv(path_csv)
        data["Condition"] = data["name_cond"].apply(Condition.from_string)  # type: ignore

        for _, row in data.iterrows():
            cond: Condition = row["Condition"]
            key = (subject, cond.layer, cond.component, cond.direction)
            if key not in results_by_subject_layer:
                results_by_subject_layer[key] = {"ecc": [], "mean_threshold": [], "std_threshold": []}
            results_by_subject_layer[key]["ecc"].append(cond.ecc)
            results_by_subject_layer[key]["mean_threshold"].append(row["mean_threshold"])
            results_by_subject_layer[key]["std_threshold"].append(row["std_threshold"])

    # Sort ecc values in ascending order for proper line connections
    for key in results_by_subject_layer.keys():
        sorted_indices = np.argsort(results_by_subject_layer[key]["ecc"])
        results_by_subject_layer[key]["ecc"] = np.array(results_by_subject_layer[key]["ecc"])[sorted_indices].tolist()
        results_by_subject_layer[key]["mean_threshold"] = np.array(results_by_subject_layer[key]["mean_threshold"])[
            sorted_indices
        ].tolist()
        results_by_subject_layer[key]["std_threshold"] = np.array(results_by_subject_layer[key]["std_threshold"])[
            sorted_indices
        ].tolist()

    # Create 3x3 grid plot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True, sharey=False)
    component_direction_colors = {
        (0, "plus"): "red",
        (0, "minus"): "orange",
        (1, "plus"): "blue",
        (1, "minus"): "cyan",
        (2, "plus"): "green",
        (2, "minus"): "purple",
    }

    shift_offsets = {
        (0, "plus"): -0.12,
        (0, "minus"): -0.06,
        (1, "plus"): 0.0,
        (1, "minus"): 0.06,
        (2, "plus"): 0.12,
        (2, "minus"): 0.18,
    }

    for i, subject in enumerate(subjects):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            ax.set_title(f"{layer}", fontsize=18) if i == 0 else None
            ax.set_ylabel("Mean Threshold", fontsize=18) if j == 0 else None
            ax.set_xlabel("ecc", fontsize=18) if i == 2 else None
            ax.grid(True)
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.tick_params(axis="both", which="major", labelsize=16)

            for (subject_key, layer_key, component, direction), values in results_by_subject_layer.items():
                if subject_key == subject and layer_key == layer:
                    eccs = np.array(values["ecc"])
                    thresholds = values["mean_threshold"]
                    errors = values["std_threshold"]
                    color = component_direction_colors[(component, direction)]

                    shift = shift_offsets[(component, direction)]
                    shifted_eccs = eccs + shift

                    ax.errorbar(
                        shifted_eccs,
                        thresholds,
                        yerr=errors,
                        fmt="o-",
                        label=f"Comp {component} {direction}" if i == 0 and j == 0 else None,
                        color=color,
                        capsize=3,
                    )

    # Add legend only once in the top-left subplot
    # axes[0, 0].legend(title="Component-Direction", fontsize=14)
    plt.tight_layout()
    plt.savefig("output/ana/plot_values/1.b.plot_3x3_subjects_layers_log.png")
    plt.close()
