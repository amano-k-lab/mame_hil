from abx_app.AttackCNN.generate_image_from_condition import generate_image_from_condition
import torch
import torchvision.transforms as transforms
from abx_app.AttackCNN.utils.condition import Condition, AttackParamsLoader
from abx_app.AttackCNN.utils.model_utils import load_model
from abx_app.AttackCNN.utils.data_utils import prepare_data_from_pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift


def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])


def generate_images(layers_and_thresholds, model, device, transform):
    generated_results = {}

    for cond_layer, threshold in layers_and_thresholds.items():
        cond = Condition(
            ecc=4,
            mode="ica",
            layer=cond_layer,
            component=0,
            direction="plus",
        )

        handler = cond.get_decomposition_handler()
        params_loader = AttackParamsLoader()
        attack_params = params_loader.get_attack_params(cond, threshold)
        data_loader = prepare_data_from_pool(1, transform, device, example=True)

        results = generate_image_from_condition(
            attack_params=attack_params,
            cond=cond,
            model=model,
            device=device,
            data_loader_origin=data_loader,
            handler=handler,
            value=threshold,
            layer1_name="conv1",
        )

        generated_results[cond_layer] = results[0]

    return generated_results


def plot_combined_diff_images(generated_results):
    fig_diff, axes_diff = plt.subplots(len(generated_results), 3, figsize=(15 + 2, 5 * len(generated_results) + 2))

    for idx, (cond_layer, result) in enumerate(generated_results.items()):
        orig_img = np.clip(result.original_image[0].squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
        fake_img = np.clip(
            result.perturbed_images[-1].squeeze().transpose((1, 2, 0)),
            0,
            1,  # type: ignore
        )

        orig_gray = rgb_to_grayscale(orig_img)
        fake_gray = rgb_to_grayscale(fake_img)
        diff_img = orig_gray - fake_gray

        if idx == 0:
            fig_diff.subplots_adjust(left=0.2)
        axes_diff[idx, 0].text(
            -0.05,
            0.5,
            f"{cond_layer} (t={layers_and_thresholds[cond_layer]})",
            fontsize=36,
            va="center",
            ha="right",
            rotation=90,
            transform=axes_diff[idx, 0].transAxes,
        )

        # Original image
        axes_diff[idx, 0].imshow(orig_gray, cmap="gray")
        axes_diff[idx, 0].axis("off")

        # Perturbed image
        axes_diff[idx, 1].imshow(fake_gray, cmap="gray")
        axes_diff[idx, 1].axis("off")

        # Difference image
        axes_diff[idx, 2].imshow(diff_img, cmap="gray")
        axes_diff[idx, 2].axis("off")
    axes_diff[0, 0].set_title("Original", fontsize=40)
    axes_diff[0, 1].set_title("Perturbed", fontsize=40)
    axes_diff[0, 2].set_title("Difference", fontsize=40)

    plt.tight_layout()
    plt.savefig("output/ana/combined_diff_images.png")


def plot_fourier_transform(generated_results):
    layer_order = ["conv1", "layer3", "avgpool"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # Adjusted figure size to reduce right whitespace

    im = None

    for i, layer_name in enumerate(layer_order):
        result = generated_results[layer_name]

        orig_img = np.clip(result.original_image[0].squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
        fake_img = np.clip(
            result.perturbed_images[-1].squeeze().transpose((1, 2, 0)),
            0,
            1,  # type: ignore
        )
        orig_gray = rgb_to_grayscale(orig_img)
        fake_gray = rgb_to_grayscale(fake_img)
        diff_img = orig_gray - fake_gray
        diff_fft = fft2(diff_img)
        diff_fft_shifted = fftshift(diff_fft)
        amplitude_spectrum = np.abs(diff_fft_shifted)

        im = axes[i].imshow(
            np.log1p(amplitude_spectrum),  # log(1 + x)
            extent=(-0.5, 0.5, -0.5, 0.5),  # Nyquist frequency is +/-0.5
            cmap="gray",
            aspect="equal",  # Ensure square aspect ratio
        )

        axes[i].set_title(layer_name, fontsize=30, pad=20)  # Added padding to prevent overlap
        axes[i].set_xlabel("Spatial Frequency (Horizontal)", fontsize=18, labelpad=10)
        axes[i].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
        axes[i].tick_params(axis="x", labelsize=14)
        if i == 0:
            axes[i].set_ylabel("Spatial Frequency (Vertical)", fontsize=18, labelpad=10)
            axes[i].set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
            axes[i].tick_params(axis="y", labelsize=14)
        else:
            axes[i].set_yticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.15)  # Increased pad to add more space
    cbar.set_label("Log Amplitude", fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    # Ensure tight layout and adjust margins for better spacing
    plt.subplots_adjust(left=0.08, right=0.85, top=0.9, bottom=0.1)  # Adjusted right to provide more space for colorbar

    plt.savefig("output/ana/fourier_transform_combined.png")
    plt.show()


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
    NUM_SAMPLES = 1

    layers_and_thresholds = {
        "conv1": 55,
        "layer3": 0.6,
        "avgpool": 0.16,
    }

    generated_results = generate_images(layers_and_thresholds, model, device, transform)
    plot_combined_diff_images(generated_results)
    plot_fourier_transform(generated_results)
