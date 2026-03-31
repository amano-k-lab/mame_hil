import torch
import torchvision.transforms as transforms
from abx_app.AttackCNN.utils.data_utils import prepare_data_from_pool, prepare_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from scipy.stats import linregress


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

    return bin_centers, radial_mean


def plot_natural_spectrum_with_error(device, transform, num_images=30):
    data_loader, _ = prepare_data(num_images, transform, device, seed=41)

    radial_freq = None
    all_radial_profiles = []

    for img, _ in data_loader:
        orig_img = np.clip(img.squeeze().permute(1, 2, 0).cpu().numpy(), 0, 1)
        orig_gray = rgb_to_grayscale(orig_img)
        orig_gray = orig_gray - np.mean(orig_gray)
        orig_fft = fft2(orig_gray)
        orig_fft_shifted = fftshift(orig_fft)
        amplitude_spectrum = np.abs(orig_fft_shifted)
        radial_freq, radial_profile = compute_radial_profile(amplitude_spectrum)
        all_radial_profiles.append(radial_profile)

    all_radial_profiles = np.array(all_radial_profiles)
    radial_profile_mean = np.mean(all_radial_profiles, axis=0)
    radial_profile_std = np.std(all_radial_profiles, axis=0)

    return radial_freq, radial_profile_mean, radial_profile_std


def find_value_and_slope(radial_freq, radial_profile_mean):
    # Find the value at freq=0.1
    freq_01_index = np.argmin(np.abs(radial_freq - 0.1))
    value_at_01 = radial_profile_mean[freq_01_index]

    # Adjust intercept so that the approximation passes through freq=0.1
    log_freq = np.log(radial_freq)
    log_profile = np.log(radial_profile_mean)
    slope, intercept, _, _, _ = linregress(log_freq, log_profile)
    intercept = np.log(value_at_01) - slope * np.log(0.1)

    return value_at_01, slope, intercept


if __name__ == "__main__":
    NUM_SAMPLES = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    plt.figure(figsize=(10, 6))

    radial_freq, radial_profile_mean, radial_profile_std = plot_natural_spectrum_with_error(
        device, transform, num_images=NUM_SAMPLES
    )
    plt.plot(radial_freq, radial_profile_mean, label="Mean Spectrum")
    plt.fill_between(
        radial_freq,
        radial_profile_mean - radial_profile_std,
        radial_profile_mean + radial_profile_std,
        alpha=0.3,
        label="Standard Deviation",
    )

    value_at_01, slope, intercept = find_value_and_slope(radial_freq, radial_profile_mean)

    log_freq = np.log(radial_freq)  # type: ignore
    approx_profile = np.exp(slope * log_freq + intercept)
    plt.plot(radial_freq, approx_profile, linestyle="--", label="Linear Approximation")

    plt.scatter([0.1], [value_at_01], color="red", label=f"Value at 0.1: {value_at_01:.4f}")
    plt.text(0.1, value_at_01 * 1.5, f"Slope: {slope:.5f}", fontsize=18, color="blue")
    plt.text(
        0.05,
        0.01,
        f"n={NUM_SAMPLES}",
        fontsize=14,
        color="black",
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    plt.xlabel("Frequency", fontsize=18)
    plt.ylabel("Amplitude", fontsize=18)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.tight_layout()
    plt.savefig("output/ana/radial_frequency_spectrum_natural.png")

    print(f"Value at freq=0.1: {value_at_01:.5f}")
    print(f"Slope of the log-log spectrum: {slope:.5f}")
