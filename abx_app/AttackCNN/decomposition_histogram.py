from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from .utils.activation_manager import ActivationManager
from .utils.decomposition_handler import DecompositionHandler
from .utils.pca import PCAHandler
from .utils.ica import ICAHandler
from .utils.data_utils import prepare_data
from .utils.model_utils import load_model


def compute_distances(data_flat, handler: DecompositionHandler):
    coordinates = handler.transform_coordinate(data_flat)
    distances = np.linalg.norm(coordinates, axis=1)
    return distances


def compute_average(data_flat, handler: DecompositionHandler):
    coordinates = handler.transform_coordinate(data_flat)
    averages = np.mean(coordinates, axis=0)
    return averages


def scatterplot(
    data_flat,
    handler: DecompositionHandler,
    layer_name,
    pca_or_ica,
    gram_or_activation,
):
    coordinates = handler.transform_coordinate(data_flat)
    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.7, edgecolor="k")
    plt.title("2D Scatter Plot of Coordinates")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.savefig(f"abx_app/AttackCNN/{pca_or_ica}_scatterplot_{layer_name}_{gram_or_activation}.png")


def visualize(distances, layer_name, pca_or_ica, gram_or_activation, bins=50):
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins, color="blue", alpha=0.7)
    plt.xlabel(f"Euclidean Distance from {pca_or_ica} Center")
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of Images in {pca_or_ica} Space of {layer_name}")
    plt.savefig(f"abx_app/AttackCNN/{pca_or_ica}_histogram_{layer_name}_{gram_or_activation}.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    GRAM_OR_ACTIVATION = "gram_matrix"
    PCA_OR_ICA = "ica"
    LAYER_NAME = "avgpool"

    NUM_SAMPLES = 500
    GRAM_MATRIX_FILE = Path(f"abx_app/AttackCNN/gram_matrix_{LAYER_NAME}.npy")
    ACTIVATION_FILE = Path(f"abx_app/AttackCNN/activation_{LAYER_NAME}.npy")
    DECOMPOSITION_MODEL_FILE = Path(f"abx_app/AttackCNN/{PCA_OR_ICA}_{GRAM_OR_ACTIVATION}_{LAYER_NAME}.pkl")

    model = load_model(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    data_loader, _ = prepare_data(NUM_SAMPLES, transform, device)
    activation_manager = ActivationManager(model, device)

    data_flat = None
    if GRAM_OR_ACTIVATION == "gram_matrix":
        data_flat = activation_manager.extract_gram_matrix_flat(data_loader, LAYER_NAME, save_path=None)
    elif GRAM_OR_ACTIVATION == "activation":
        data_flat = activation_manager.extract_activation_flat(data_loader, LAYER_NAME, save_path=None)

    handler = None
    if PCA_OR_ICA == "pca":
        handler = PCAHandler(DECOMPOSITION_MODEL_FILE)
    elif PCA_OR_ICA == "ica":
        handler = ICAHandler(DECOMPOSITION_MODEL_FILE)

    if handler is None or data_flat is None:
        raise ValueError("Invalid decomposition method specified. Use 'pca' or 'ica', 'gram_matrix' or 'activation'.")

    distances = compute_distances(data_flat, handler)
    visualize(distances, LAYER_NAME, PCA_OR_ICA, GRAM_OR_ACTIVATION)
    print(f"average: {compute_average(data_flat, handler)}")

    scatterplot(data_flat, handler, LAYER_NAME, PCA_OR_ICA, GRAM_OR_ACTIVATION)
