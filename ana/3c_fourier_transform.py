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


def generate_and_plot_spectrum_with_error(
    cond_layer, threshold, ecc, model, device, transform, num_images=30, original=False, seed=42
):
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

    radial_freq = None
    all_radial_profiles = []
    weighted_freqs = []

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
        if original:
            diff_img = orig_gray
        diff_img = diff_img - np.mean(diff_img)

        diff_fft = fft2(diff_img)
        diff_fft_shifted = fftshift(diff_fft)
        amplitude_spectrum = np.abs(diff_fft_shifted)

        radial_freq, radial_profile = compute_radial_profile(amplitude_spectrum)
        all_radial_profiles.append(radial_profile)

        weighted_freq = np.sum(radial_freq * radial_profile) / np.sum(radial_profile)
        weighted_freqs.append(weighted_freq)

    all_radial_profiles = np.array(all_radial_profiles)
    radial_profile_mean = np.mean(all_radial_profiles, axis=0)
    radial_profile_std = np.std(all_radial_profiles, axis=0)

    weighted_freq_mean = np.mean(weighted_freqs)
    weighted_freq_std = np.std(weighted_freqs)

    return radial_freq, radial_profile_mean, radial_profile_std, weighted_freq_mean, weighted_freq_std


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

    thresholds = {
        "conv1": 55,
        "layer3": 0.6,
        "avgpool": 0.16,
    }
    colors = {"conv1": "blue", "layer3": "green", "avgpool": "red"}

    results_data = []

    plt.figure(figsize=(10, 6))
    for layer, color in colors.items():
        threshold = thresholds[layer]
        ecc = 4
        radial_freq, radial_profile_mean, radial_profile_std, freq_mean, freq_std = (
            generate_and_plot_spectrum_with_error(
                layer, threshold, ecc, model, device, transform, num_images=NUM_IMAGES
            )
        )
        plt.plot(radial_freq, radial_profile_mean, color=color, label=f"{layer} (target={threshold})")
        plt.fill_between(
            radial_freq,
            radial_profile_mean - radial_profile_std,
            radial_profile_mean + radial_profile_std,
            alpha=0.3,
            color=color,
        )
        results_data.append(
            {
                "ecc": ecc,
                "layer": layer,
                "threshold": threshold,
                "weighted_freq_mean": freq_mean,
                "weighted_freq_std": freq_std,
            }
        )

    plt.xlabel("Frequency (cpd)", fontsize=20)
    plt.ylabel("Amplitude", fontsize=20)
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.text(
        0.10,
        0.05,
        f"n={NUM_IMAGES}",
        fontsize=18,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
    )
    plt.tight_layout()
    plt.savefig(f"output/ana/radial_frequency_spectrum.png")
    plt.close()

    df = pd.DataFrame(results_data)
    df.to_csv("output/ana/layer_freqs.csv", index=False)
