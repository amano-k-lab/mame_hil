from pathlib import Path
import yaml
import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils

from . import config
from .utils.decomposition_handler import DecompositionHandler
from .utils.ica import ICAHandler
from .utils.data_utils import prepare_data
from .utils.model_utils import load_model
from .utils.activation_manager import ActivationManager


config_ex = config["extract_original"]


def compute_distances(data_flat, handler: DecompositionHandler):
    coordinates = handler.transform_coordinate(data_flat)
    distances = np.linalg.norm(coordinates, axis=1)
    return distances


def select_images_within_top_percent(layer_distances, top_percent=10):
    top_indices_per_layer = []

    for distances in layer_distances:
        threshold = np.percentile(distances, top_percent)
        top_indices = set(np.where(distances <= threshold)[0])
        top_indices_per_layer.append(top_indices)

    selected_indices = set.intersection(*top_indices_per_layer)
    return selected_indices


def select_images_within_threshold(distances, threshold):
    return set(np.where(distances <= threshold)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ica model file")
    parser.add_argument(
        "--ica_model_top10",
        type=bool,
        default=None,
    )
    args = parser.parse_args()
    USE_TOP10 = args.ica_model_top10 if args.ica_model_top10 is not None else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    LAYERS = config["layers"]

    NUM_SAMPLES = config_ex["num_samples"]
    TOP_PERCENT = config_ex["top_percent"]
    BATCH_SIZE = config_ex["batch_size"]
    OUTPUT_DIR = Path(config["imgpool_path"])
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    model = load_model(device)
    activation_manager = ActivationManager(model, device)

    # compute thresholds
    thresholds = {}
    for layer_name in LAYERS:
        print(f"Calculating threshold for layer: {layer_name}")
        gram_matrix_file = Path(config["gram_matrix"]["gram_path"]) / f"gram_matrices_{layer_name}.npy"
        gram_matrix_flat = np.load(gram_matrix_file, allow_pickle=True)
        ica_model_file = (
            Path(config["ica"]["ica_models_top10"]) / f"{layer_name}.pkl"
            if USE_TOP10
            else Path(config["ica"]["ica_models"]) / f"{layer_name}.pkl"
        )
        handler = ICAHandler(ica_model_file)

        distances = compute_distances(gram_matrix_flat, handler)
        thresholds[layer_name] = np.percentile(distances, TOP_PERCENT)

    print(f"Calculated thresholds: {thresholds}")

    # extract original images
    total_processed = 0
    total_selected = 0
    data_loader, _ = prepare_data(NUM_SAMPLES, transform, device, batch_size=BATCH_SIZE)

    for batch_idx, (images, _) in enumerate(data_loader):
        batch_size = len(images)
        print(f"Processing batch {batch_idx + 1}: {batch_size} images")
        selected_indices = set(range(batch_size))

        for layer_name in LAYERS:
            gram_matrix_flat = activation_manager.extract_gram_matrix_flat(images, layer_name)

            ica_model_file = (
                Path(config["ica"]["ica_models_top10"]) / f"{layer_name}.pkl"
                if USE_TOP10
                else Path(config["ica"]["ica_models"]) / f"{layer_name}.pkl"
            )
            handler = ICAHandler(ica_model_file)
            distances = compute_distances(gram_matrix_flat, handler)

            layer_selected_indices = select_images_within_threshold(distances, thresholds[layer_name])
            selected_indices &= layer_selected_indices

        for idx in selected_indices:
            output_path = OUTPUT_DIR / f"image_{total_processed + idx}.png"
            torchvision.utils.save_image(images[idx], output_path)
            print(f"Saved: {output_path}")

        total_processed += batch_size
        total_selected += len(selected_indices)

        print(f"Batch {batch_idx + 1} completed. Total processed: {total_processed}, Total selected: {total_selected}")
        print("Process completed.")
