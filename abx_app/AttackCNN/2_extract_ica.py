from pathlib import Path

import torch

from . import config
from .utils.ica import ICAHandler


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    LAYERS = config["layers"]
    NUM_COMPONENTS_FULL = config["ica"]["num_components_full"]
    NUM_COMPONENTS_TOP_K = config["ica"]["num_components_top_k"]

    for layer_name in LAYERS:
        print(f"Loading gram matrix of {layer_name}")
        # gram_matrix_file = Path(config["gram_matrix"]["gram_path"]) / f"gram_matrices_{layer_name}.npy"
        gram_matrix_file = Path(config["gram_matrix"]["gram_path_stylized"]) / f"gram_matrices_{layer_name}.npy"

        # ica_model_file_full = Path(config["ica"]["ica_models_full"]) / f"{layer_name}.pkl"
        ica_model_file_full = Path(config["ica"]["ica_models_stylized_full"]) / f"{layer_name}.pkl"

        ica_model_file_full.parent.mkdir(parents=True, exist_ok=True)

        # ica_model_file_top_10 = Path(config["ica"]["ica_models_top10"]) / f"{layer_name}.pkl"
        ica_model_file_top_10 = Path(config["ica"]["ica_models_stylized_top10"]) / f"{layer_name}.pkl"

        ica_model_file_top_10.parent.mkdir(parents=True, exist_ok=True)
        print(f"Perform ICA of {layer_name}")
        ica_handler = ICAHandler(ica_model_file_full)
        ica_result = ica_handler.perform_ica(gram_matrix_file, n_components=NUM_COMPONENTS_FULL)
        explained_variance, sorted_indices = ica_handler.extract_explained_variance(gram_matrix_file)
        ica_handler.plot_explained_variance(explained_variance, layer_name)
        ica_handler.change_model_top_k_components(
            sorted_indices, top_k=NUM_COMPONENTS_TOP_K, ica_model_file=ica_model_file_top_10
        )
        print(f"ICA completed")
