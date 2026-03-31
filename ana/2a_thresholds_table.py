import pandas as pd
from pathlib import Path
import numpy as np

from abx_app.AttackCNN.utils.condition import Condition

if __name__ == "__main__":
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

    # Aggregate data by subject and layer
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

    # Compute mean and SD across components and directions for each subject
    subject_means = {subject: {layer: {} for layer in layers} for subject in subjects}
    subject_sds = {subject: {layer: {} for layer in layers} for subject in subjects}

    for subject, layer_data in aggregated_data.items():
        for layer, ecc_data in layer_data.items():
            for ecc, values in ecc_data.items():
                subject_means[subject][layer][ecc] = round(np.mean(values), 4)
                subject_sds[subject][layer][ecc] = round(np.std(values), 4)

    # Prepare table data for each subject
    table_data = []
    for subject in subjects:
        for layer in layers:
            for ecc in [4, 8, 12]:
                mean_value = subject_means[subject][layer].get(ecc, np.nan)
                sd_value = subject_sds[subject][layer].get(ecc, np.nan)
                table_data.append(
                    {
                        "Subject": subject,
                        "Layer": layer,
                        "Eccentricity": ecc,
                        "Mean Threshold": mean_value,
                        "SD (Component x Direction)": sd_value,
                    }
                )

    # Convert to DataFrame and save as CSV
    table_df = pd.DataFrame(table_data)
    output_path = Path("output/ana/subject_layer_thresholds.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")
