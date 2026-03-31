from pathlib import Path

import numpy as np
import torch

from . import config
from .utils.pca import PCAHandler


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    LAYERS = config["layers"]
    NUM_COMPONENTS_FULL = config["pca"]["num_components_full"]
    NUM_COMPONENTS_TOP_K = config["pca"]["num_components_top_k"]

    for layer_name in LAYERS:
        print(f"Loading gram matrix of {layer_name}")
        gram_matrix_file = Path(config["gram_matrix"]["gram_path"]) / f"gram_matrices_{layer_name}.npy"
        pca_model_file_full = Path(config["pca"]["pca_models_full"]) / f"{layer_name}.pkl"
        pca_model_file_full.parent.mkdir(parents=True, exist_ok=True)
        pca_model_file_top_10 = Path(config["pca"]["pca_models_top10"]) / f"{layer_name}.pkl"
        pca_model_file_top_10.parent.mkdir(parents=True, exist_ok=True)
        print(f"Perform PCA of {layer_name}")
        pca_handler = PCAHandler(pca_model_file_full)
        pca_result = pca_handler.perform_pca(gram_matrix_file, n_components=NUM_COMPONENTS_FULL)
        explained_variance = pca_result.explained_variance_ratio_
        sorted_indices = np.argsort(explained_variance)[::-1]

        pca_handler.plot_explained_variance(layer_name)
        pca_handler.change_model_top_k_components(
            sorted_indices, top_k=NUM_COMPONENTS_TOP_K, pca_model_file=pca_model_file_top_10
        )
        print(f"PCA completed")
