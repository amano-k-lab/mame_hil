from abx_app.AttackCNN.generate_image_from_condition import generate_image_from_condition
import torch
import torchvision.transforms as transforms
from abx_app.AttackCNN.utils.condition import Condition, AttackParamsLoader
from abx_app.AttackCNN.utils.model_utils import load_model
from abx_app.AttackCNN.utils.data_utils import prepare_data_from_pool
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

        rms_contrast = np.std(diff_img)
        rms_values.append(rms_contrast)

    rms_mean = np.mean(rms_values)
    rms_std = np.std(rms_values)

    return rms_mean, rms_std


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
    NUM_IMAGES = 30

    ecc_values = [4, 8, 12]
    thresholds = {
        "conv1": [47.915, 63.0404, 63.7918],
        "layer3": [0.486, 0.5983, 0.6407],
        "avgpool": [0.1538, 0.153, 0.1688],
    }
    rms_results = {layer: {} for layer in thresholds.keys()}

    # print("RMS Contrast Results:")
    for ecc_idx, ecc in enumerate(ecc_values):
        for layer in thresholds.keys():
            threshold = thresholds[layer][ecc_idx]
            rms_mean, rms_std = generate_and_calculate_rms(
                layer, threshold, ecc, model, device, transform, num_images=NUM_IMAGES
            )
            rms_results[layer][ecc] = {"mean": rms_mean, "std": rms_std}
            # print(
            #     f"Layer: {layer}, ECC: {ecc}, Threshold: {threshold}, RMS Mean: {rms_mean:.4f}, RMS Std: {rms_std:.4f}"
            # )

    fig, ax = plt.subplots(figsize=(10, 6))
    layer_colors = {"conv1": "red", "layer3": "blue", "avgpool": "green"}
    markers = {4: "o", 8: "s", 12: "^"}
    x_positions = {"conv1": 1, "layer3": 2, "avgpool": 3}
    x_labels = ["conv1", "layer3", "avgpool"]
    ecc_shifts = {4: -0.06, 8: 0.0, 12: 0.06}

    # Plot each layer's data
    for layer, ecc_data in rms_results.items():
        for ecc, results in ecc_data.items():
            x = x_positions[layer] + ecc_shifts[ecc]
            y = results["mean"]
            error = results["std"]
            ax.errorbar(
                x,
                y,
                yerr=error,
                color=layer_colors[layer],
                marker=markers[ecc],
                markersize=12,
                linestyle="--",
                capsize=6,
                label=f"{layer} (Ecc={ecc})" if ecc == 4 else "",
            )

    # Set plot labels and formatting
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(x_labels, fontsize=32)
    ax.set_ylabel("RMS Contrast", fontsize=36)
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.grid(True)

    # Create legend for clarity
    legend_elements = [
        # Line2D([0], [0], color="red", lw=2, label="Conv1"),
        # Line2D([0], [0], color="blue", lw=2, label="Layer3"),
        # Line2D([0], [0], color="green", lw=2, label="Avgpool"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=15, label="Ecc=4"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=15, label="Ecc=8"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=15, label="Ecc=12"),
    ]
    ax.legend(handles=legend_elements, fontsize=28, loc="upper right")
    plt.tight_layout()
    plt.savefig("output/ana/rms_contrast.png")
