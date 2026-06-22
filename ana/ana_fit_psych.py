import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psignifit as ps


def _binomial_log_likelihood(k, n, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def _extract_condition_labels(stem):
    match_ecc = re.search(r"(ecc\d+)", stem)
    match_layer = re.search(r"ica_([^_]+)_", stem)
    match_component = re.search(r"(component\d+)", stem)
    match_direction = re.search(r"_(plus|minus)$", stem)
    return {
        "ecc": match_ecc.group(1) if match_ecc else "",
        "layer": match_layer.group(1) if match_layer else "",
        "component": match_component.group(1) if match_component else "",
        "direction": match_direction.group(1) if match_direction else "",
    }


def build_psychometric_data(df, min_total):
    required_columns = {"list_seq_threshold", "list_seq_actual_value", "list_seq_hit"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    work = df.copy()
    work["level_key"] = work["list_seq_threshold"]
    grouped = (
        work.groupby("level_key", as_index=False)
        .agg(
            stimulus_value=("list_seq_actual_value", "mean"),
            stimulus_value_std=("list_seq_actual_value", "std"),
            nCorrect=("list_seq_hit", "sum"),
            nTotal=("list_seq_hit", "count"),
        )
        .sort_values("level_key")
    )
    grouped = grouped[grouped["nTotal"] > min_total].copy()
    grouped["proportion_correct"] = grouped["nCorrect"] / grouped["nTotal"]
    return grouped


def calculate_fit_metrics(res, grouped):
    stimulus = grouped["stimulus_value"].to_numpy(dtype=float)
    n_correct = grouped["nCorrect"].to_numpy(dtype=float)
    n_total = grouped["nTotal"].to_numpy(dtype=float)
    observed = n_correct / n_total
    predicted = np.asarray(res.proportion_correct(stimulus), dtype=float)
    residual = observed - predicted

    ll_model = _binomial_log_likelihood(n_correct, n_total, predicted)
    ll_saturated = _binomial_log_likelihood(n_correct, n_total, observed)
    null_rate = np.repeat(n_correct.sum() / n_total.sum(), len(n_correct))
    ll_null = _binomial_log_likelihood(n_correct, n_total, null_rate)

    deviance = 2 * (ll_saturated - ll_model)
    null_deviance = 2 * (ll_saturated - ll_null)
    pseudo_r2 = np.nan
    if null_deviance > 0:
        pseudo_r2 = 1 - deviance / null_deviance

    grouped = grouped.copy()
    grouped["predicted_proportion_correct"] = predicted
    grouped["residual"] = residual

    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "log_likelihood": ll_model,
        "deviance": float(deviance),
        "null_deviance": float(null_deviance),
        "pseudo_r2": float(pseudo_r2),
    }, grouped


def plot_fit_without_errorbars(res, grouped, stem, output_dir):
    stimulus = grouped["stimulus_value"].to_numpy(dtype=float)
    observed = grouped["proportion_correct"].to_numpy(dtype=float)
    n_total = grouped["nTotal"].to_numpy(dtype=float)

    x_min = float(stimulus.min())
    x_max = float(stimulus.max())
    x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    x = np.linspace(x_min - x_margin, x_max + x_margin, 300)
    y = np.asarray(res.proportion_correct(x), dtype=float)

    plt.figure(figsize=(5, 4))
    plt.plot(x, y, color="black", linewidth=2)
    plt.scatter(
        stimulus,
        observed,
        s=25 + 8 * np.sqrt(n_total),
        color="tab:blue",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    plt.ylim(0.45, 1.02)
    plt.xlabel("Stimulus value")
    plt.ylabel("Proportion correct")
    plt.title(stem)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fit_{stem}.png"), dpi=200, bbox_inches="tight")
    plt.close()


def fit_psych(csv_file, stem, min_total, output_dir):
    df = pd.read_csv(csv_file)
    grouped = build_psychometric_data(df, min_total)
    if len(grouped) < 2:
        raise ValueError(f"{stem}: not enough stimulus levels after min_total filtering")

    data = grouped[["stimulus_value", "nCorrect", "nTotal"]].to_numpy()

    conf = ps.Configuration()
    conf.experiment_type = "nAFC"
    conf.experiment_choices = 2
    conf.sigmoidName = "sigmoid"
    conf.threshPC = 0.707
    conf.stimulus_range = [data[:, 0].min(), data[:, 0].max()]

    res = ps.psignifit(data, conf)

    param = res.get_parameter_estimate()
    threshold = param["threshold"]
    width = param["width"]
    metrics, grouped_with_fit = calculate_fit_metrics(res, grouped)
    plot_fit_without_errorbars(res, grouped_with_fit, stem, output_dir)

    grouped_with_fit.insert(0, "condition", stem)
    grouped_with_fit.to_csv(os.path.join(output_dir, f"fit_points_{stem}.csv"), index=False)

    summary = {
        "condition": stem,
        **_extract_condition_labels(stem),
        "threshold": threshold,
        "width": width,
        "confidence_interval": width,
        "n_levels": int(len(grouped_with_fit)),
        "n_trials": int(grouped_with_fit["nTotal"].sum()),
        "stimulus_min": float(grouped_with_fit["stimulus_value"].min()),
        "stimulus_max": float(grouped_with_fit["stimulus_value"].max()),
        **metrics,
    }
    return summary, grouped_with_fit


if __name__ == "__main__":
    min_total = 2
    name_user = "s02"
    path_save_fig = os.path.join("ana", "results_20260331", name_user)
    os.makedirs(path_save_fig, exist_ok=True)

    list_summary = []
    list_fit_points = []

    csv_files = sorted(glob.glob(os.path.join(path_save_fig, "ecc*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No ecc*.csv files found in {path_save_fig}")

    for csv_file in csv_files:
        stem = os.path.splitext(os.path.basename(csv_file))[0]
        summary, fit_points = fit_psych(csv_file, stem, min_total, path_save_fig)
        list_summary.append(summary)
        list_fit_points.append(fit_points)
        print(
            stem,
            "threshold:",
            summary["threshold"],
            "width:",
            summary["width"],
            "rmse:",
            summary["rmse"],
            "deviance:",
            summary["deviance"],
        )

    df_summary = pd.DataFrame(list_summary)
    #df_summary.to_csv(os.path.join(path_save_fig, "summary_thresholds.csv"), index=False)

    if list_fit_points:
        df_fit_points = pd.concat(list_fit_points, ignore_index=True)
        df_fit_points.to_csv(os.path.join(path_save_fig, "summary_fit_points.csv"), index=False)

    if "layer" in df_summary.columns and len(df_summary) > 0:
        df_layer_summary = (
            df_summary.groupby("layer", as_index=False)
            .agg(
                n_conditions=("condition", "count"),
                mean_rmse=("rmse", "mean"),
                std_rmse=("rmse", "std"),
                mean_mae=("mae", "mean"),
                std_mae=("mae", "std"),
                mean_deviance=("deviance", "mean"),
                std_deviance=("deviance", "std"),
                mean_pseudo_r2=("pseudo_r2", "mean"),
                std_pseudo_r2=("pseudo_r2", "std"),
            )
            .sort_values("layer")
        )
        df_layer_summary.to_csv(os.path.join(path_save_fig, "summary_fit_by_layer.csv"), index=False)

    print("done")
