import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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
                results_by_subject_layer[key] = {"ecc": [], "mean_threshold": []}
            results_by_subject_layer[key]["ecc"].append(cond.ecc)
            results_by_subject_layer[key]["mean_threshold"].append(row["mean_threshold"])

    # Aggregate data by subject and ecc
    aggregated_data = {subject: {layer: {} for layer in layers} for subject in subjects}

    for (subject, layer, component, direction), values in results_by_subject_layer.items():
        eccs = np.array(values["ecc"])
        thresholds = np.array(values["mean_threshold"])
        for ecc in np.unique(eccs):
            mask = eccs == ecc
            mean_value = thresholds[mask].mean()
            if ecc not in aggregated_data[subject][layer]:
                aggregated_data[subject][layer][ecc] = []
            aggregated_data[subject][layer][ecc].append(mean_value)

    # Compute mean across components and directions
    subject_means = {subject: {layer: {} for layer in layers} for subject in subjects}
    for subject, layer_data in aggregated_data.items():
        for layer, ecc_data in layer_data.items():
            for ecc, values in ecc_data.items():
                subject_means[subject][layer][ecc] = np.mean(values)

    # Compute overall mean and SD across subjects
    overall_means = {layer: {} for layer in layers}
    overall_sds = {layer: {} for layer in layers}
    for layer in layers:
        for ecc in np.unique(np.concatenate([list(subject_means[s][layer].keys()) for s in subjects])):
            subject_values = [subject_means[s][layer].get(ecc, np.nan) for s in subjects]
            subject_values = np.array(subject_values)[~np.isnan(subject_values)]
            overall_means[layer][ecc] = np.mean(subject_values)
            overall_sds[layer][ecc] = np.std(subject_values)

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"s00": "red", "s01": "blue", "s02": "green", "overall": "black"}
    markers = {4: "o", 8: "s", 12: "^"}

    x_positions = {"conv1": 1, "layer3": 2, "avgpool": 3}
    x_labels = ["conv1", "layer3", "avgpool"]
    shift_offsets = {
        ("indiv", 4): -0.06,
        ("indiv", 8): -0.04,
        ("indiv", 12): -0.02,
        ("overall", 4): 0,
        ("overall", 8): 0.02,
        ("overall", 12): 0.04,
    }

    for subject in subjects:
        for ecc in [4, 8, 12]:
            x = [x_positions[layer] + shift_offsets[("indiv", ecc)] for layer in layers]
            y = [subject_means[subject][layer].get(ecc, np.nan) for layer in layers]
            ax.plot(x, y, label=f"Individual (ecc={ecc})", color=colors[subject], marker=markers[ecc], linestyle="-")

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=18)
    ax.set_ylabel("Mean Threshold", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig("output/ana/plot_values/aggregated_subjects_thresholds.png")
    plt.close()
