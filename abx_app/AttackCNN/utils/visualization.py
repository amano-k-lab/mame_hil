from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .attack_result import AttackResult


def show_adversarial_examples(
    results: List[AttackResult],
    target_image,
    layer_name,
    num_iterations,
    alpha,
    title=None,
):
    plt.figure(figsize=(15, 40))
    images_per_row = 5
    num_rows = len(results) * (num_iterations // images_per_row + 2)

    if target_image is not None:
        plt.subplot(num_rows, images_per_row, 1)
        plt.title(f"Target Image\nLayer: {layer_name}\n Alpha: {alpha}")
        target_img = np.clip(target_image.cpu().squeeze().numpy().transpose((1, 2, 0)), 0, 1)
        plt.imshow(target_img)
        plt.axis("off")
    else:
        plt.subplot(num_rows, images_per_row, 1)
        plt.title(f"Target\nLayer: {layer_name}\n Alpha: {alpha}")
        plt.axis("off")

    plot_idx = images_per_row + 1
    for _, result in enumerate(results):
        plt.subplot(num_rows, images_per_row, plot_idx)
        plt.title(f"Original\nLoss: {result.original_loss:.1e}")
        orig_img = np.clip(result.original_image.squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
        plt.imshow(orig_img)
        plt.axis("off")
        plot_idx += 1

        for iter_idx, perturbed_img in enumerate(result.perturbed_images):
            plt.subplot(num_rows, images_per_row, plot_idx)
            plt.title(f"Iter {iter_idx + 1}\nLoss: {result.loss_list[iter_idx]:.1e}")
            perturbed_img_np = np.clip(perturbed_img.squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
            plt.imshow(perturbed_img_np)
            plt.axis("off")
            plot_idx += 1

        if (plot_idx - 1) % images_per_row > 0:
            plot_idx += images_per_row - ((plot_idx - 1) % images_per_row)

    plt.tight_layout()
    if title is None:
        plt.savefig(f"abx_app/AttackCNN/adversarial_examples_{layer_name}.png")
    else:
        plt.savefig(title)
    plt.close()


def plot_loss_trend(results: List[AttackResult], num_iterations, layer_name, title=None):
    plt.figure(figsize=(10, 6))
    for idx, result in enumerate(results):
        plt.plot(
            range(1, num_iterations + 1),
            result.loss_list,
            marker="o",
            label=f"Sample {idx + 1}",
        )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Trend During PGD Attack")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if title is None:
        plt.savefig(f"abx_app/AttackCNN/adversarial_examples_loss_{layer_name}.png")
    else:
        plt.savefig(title)
    plt.show()
