import torch
import torchvision.transforms as transforms
from abx_app.AttackCNN.generate_image_from_condition import generate_image_from_condition
from abx_app.AttackCNN.utils.condition import Condition, AttackParamsLoader
from abx_app.AttackCNN.utils.model_utils import load_model
from abx_app.AttackCNN.utils.data_utils import prepare_data_from_pool
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.fftpack import fft2, fftshift


def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])


def compute_radial_profile(data, num_bins=100):
    y, x = np.indices(data.shape)
    center = np.array([(s - 1) / 2 for s in data.shape])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    nyquist_frequency = 0.5
    freq_scale = nyquist_frequency / (224 / 2)
    radial_freq = r * freq_scale

    max_frequency = nyquist_frequency
    min_frequency = 1 / 224
    bin_edges = np.linspace(min_frequency, max_frequency, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_indices = np.digitize(radial_freq.ravel(), bin_edges, right=False)
    radial_mean = np.zeros(num_bins)
    for i in range(1, num_bins + 1):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            radial_mean[i - 1] = np.mean(data.ravel()[bin_mask])

    bin_centers_cpd = bin_centers * 224 / 4
    return bin_centers_cpd, radial_mean


def interpolate_sensitivity(frequency, sensitivity_data, ecc_idx):
    freq_vals = sensitivity_data[:, 0]
    sensitivity_vals = sensitivity_data[:, ecc_idx]
    return np.interp(frequency, freq_vals, sensitivity_vals)


def compute_weighted_spectrum(
    cond_layer, threshold, ecc, model, device, transform, num_images=30, seed=42, sensitivity_data=None, ecc_idx=None
):
    random.seed(seed)
    conds = [
        Condition(
            ecc=ecc,
            mode="ica",
            layer=cond_layer,
            component=random.choice([0, 1, 2]),
            direction=random.choice(["plus", "minus"]),
        )
        for _ in range(num_images)
    ]

    params_loader = AttackParamsLoader()
    attack_params = params_loader.get_attack_params(conds[0], threshold)
    data_loader = prepare_data_from_pool(num_images, transform, device, seed=seed)
    values = [threshold] * num_images

    results = generate_image_from_condition(
        attack_params=attack_params,
        cond=conds[0],
        model=model,
        device=device,
        data_loader_origin=data_loader,
        handler=conds[0].get_decomposition_handler(),
        conds=conds,
        values=values,
        layer1_name="conv1",
    )

    total_weighted_sum = []

    for result in results:
        orig_gray = rgb_to_grayscale(np.clip(result.original_image[0].squeeze().transpose((1, 2, 0)), 0, 1))  # type: ignore
        fake_gray = rgb_to_grayscale(np.clip(result.perturbed_images[-1].squeeze().transpose((1, 2, 0)), 0, 1))  # type: ignore
        diff_img = orig_gray - fake_gray - np.mean(orig_gray)

        diff_fft_shifted = fftshift(fft2(diff_img))
        amplitude_spectrum = np.abs(diff_fft_shifted)

        radial_freq, radial_profile = compute_radial_profile(amplitude_spectrum)
        sensitivities = interpolate_sensitivity(radial_freq, sensitivity_data, ecc_idx)

        weighted_spectrum = np.sum(radial_profile * sensitivities) / len(radial_profile)
        total_weighted_sum.append(weighted_spectrum)

    return np.mean(total_weighted_sum), np.std(total_weighted_sum)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    NUM_IMAGES = 100
    ecc_values = [4, 8, 12]
    thresholds = {
        "conv1": [47.915, 63.0404, 63.7918],
        "layer3": [0.486, 0.5983, 0.6407],
        "avgpool": [0.1538, 0.153, 0.1688],
    }
    sensitivity_data = pd.read_csv("/home/proj_texture_hil/data/csf_sensitivity_20250210.csv").values

    results_data = []

    for ecc_idx, ecc in enumerate(ecc_values):
        for layer in thresholds.keys():
            weighted_spectrum_mean, weighted_spectrum_std = compute_weighted_spectrum(
                layer,
                thresholds[layer][ecc_idx],
                ecc,
                model,
                device,
                transform,
                num_images=NUM_IMAGES,
                sensitivity_data=sensitivity_data,
                ecc_idx=ecc_idx + 1,
            )
            results_data.append(
                {
                    "ecc": ecc,
                    "layer": layer,
                    "weighted_spectrum_mean": weighted_spectrum_mean,
                    "weighted_spectrum_std": weighted_spectrum_std,
                }
            )

    df = pd.DataFrame(results_data)
    df.to_csv("output/ana/layer_weighted_spectrum.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"conv1": "red", "layer3": "blue", "avgpool": "green"}
    markers = {4: "o", 8: "s", 12: "^"}
    x_labels = list(thresholds.keys())
    x_positions = range(len(x_labels))
    x_labels = ["conv1", "layer3", "avgpool"]
    ecc_shifts = {4: -0.06, 8: 0.0, 12: 0.06}

    for data in results_data:
        x = x_positions[x_labels.index(data["layer"])] + ecc_shifts[data["ecc"]]
        ax.errorbar(
            x,
            data["weighted_spectrum_mean"],
            yerr=data["weighted_spectrum_std"],
            color=colors[data["layer"]],
            marker=markers[data["ecc"]],
            label=f"Ecc={data['ecc']}",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_xlabel("Layer", fontsize=22)
    ax.set_ylabel("Weighted Spectrum", fontsize=22)
    ax.set_ylim(0, 13)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True)

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Conv1"),
        Line2D([0], [0], color="blue", lw=2, label="Layer3"),
        Line2D([0], [0], color="green", lw=2, label="Avgpool"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10, label="Ecc=4"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=10, label="Ecc=8"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=10, label="Ecc=12"),
    ]
    ax.legend(handles=legend_elements, fontsize=18, loc="upper left")
    plt.text(
        0.11,
        0.01,
        f"n={NUM_IMAGES}",
        fontsize=18,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()
    plt.savefig("output/ana/layer_weighted_spectrum_plot.png")
    plt.show()
