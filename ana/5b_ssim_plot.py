from abx_app.AttackCNN.generate_image_from_condition import generate_image_from_condition
import torch
import torchvision.transforms as transforms
from abx_app.AttackCNN.utils.condition import Condition, AttackParamsLoader
from abx_app.AttackCNN.utils.model_utils import load_model
from abx_app.AttackCNN.utils.data_utils import prepare_data_from_pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import random
import pandas as pd


def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])


def compute_rms_contrast(image):
    """Calculate RMS contrast of a grayscale image."""
    return np.std(image)


def generate_and_calculate_rms(cond_layer, threshold, ecc, model, device, transform, num_images=30, seed=42):
    random.seed(seed)
    cond = Condition(
        ecc=ecc,
        mode="ica",
        layer=cond_layer,
        component=0,
        direction="plus",
    )
    components = [0, 1, 2]
    directions = ["plus", "minus"]
    conds = [
        Condition(
            ecc=ecc,
            mode="ica",
            layer=cond_layer,
            component=random.choice(components),
            direction=random.choice(directions),
        )
        for _ in range(num_images)
    ]

    handler = cond.get_decomposition_handler()
    params_loader = AttackParamsLoader()
    attack_params = params_loader.get_attack_params(cond, threshold)
    data_loader = prepare_data_from_pool(num_images, transform, device, seed=seed)
    values = [threshold] * num_images

    results = generate_image_from_condition(
        attack_params=attack_params,
        cond=cond,
        model=model,
        device=device,
        data_loader_origin=data_loader,
        handler=handler,
        conds=conds,
        values=values,
        layer1_name="conv1",
    )

    rms_values = []
    for result in results:
        orig_img = np.clip(result.original_image[0].squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
        fake_img = np.clip(
            result.perturbed_images[-1].squeeze().transpose((1, 2, 0)),  # type: ignore
            0,
            1,
        )  # type: ignore
        orig_gray = rgb_to_grayscale(orig_img)
        fake_gray = rgb_to_grayscale(fake_img)
        diff_img = orig_gray - fake_gray

        rms_contrast = compute_rms_contrast(diff_img)
        rms_values.append(rms_contrast)

    rms_mean = np.mean(rms_values)
    rms_std = np.std(rms_values)

    return rms_mean, rms_std


def plot_rms_vs_threshold(layer, thresholds, rms_means, rms_stds, output_dir="output"):
    """
    Plot RMS contrast vs threshold for a specific layer and save the plot.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        thresholds,
        rms_means,
        yerr=rms_stds,
        fmt="o-",
        ecolor="red",
        elinewidth=4.0,
        label=f"layer: {layer}",
        capsize=5,
        linewidth=7.5,
    )
    plt.title(f"{layer}", fontsize=72)
    plt.xlabel("ICA", fontsize=72)
    plt.ylabel("RMS C", fontsize=72)
    xticks = [thresholds[0], thresholds[-1]]
    plt.xticks(xticks, fontsize=54)
    plt.xticks(fontsize=54)
    ymin, ymax = plt.ylim()
    yticks = np.linspace(ymin, ymax, 2)
    yticks = np.round(yticks, 2)
    plt.yticks(yticks, fontsize=54)
    # plt.legend(fontsize=18, loc="best")
    plt.grid(True)
    plt.text(
        0.95,
        0.05,
        f"n={NUM_IMAGES}",
        fontsize=54,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ana/rms_plot_{layer}.png")
    plt.close()


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

    search_ranges = {
        "conv1": np.arange(10, 110, 10),
        "layer3": np.arange(0.1, 1.1, 0.1),
        "avgpool": np.arange(0.04, 0.44, 0.04),
    }

    ecc = 4
    NUM_IMAGES = 30
    results = {}

    for layer, thresholds in search_ranges.items():
        rms_means = []
        rms_stds = []
        print(f"Processing {layer}...")

        for threshold in thresholds:
            rms_mean, rms_std = generate_and_calculate_rms(
                layer, threshold, ecc, model, device, transform, num_images=NUM_IMAGES
            )
            rms_means.append(rms_mean)
            rms_stds.append(rms_std)
            print(f"Layer: {layer}, Threshold: {threshold:.2f}, RMS Mean: {rms_mean:.4f}, RMS Std: {rms_std:.4f}")

        # Store results and plot
        results[layer] = {"thresholds": thresholds, "rms_means": rms_means, "rms_stds": rms_stds}
        plot_rms_vs_threshold(layer, thresholds, rms_means, rms_stds)

    print("Plots saved for all layers.")
