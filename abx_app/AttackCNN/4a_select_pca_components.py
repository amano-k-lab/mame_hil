import yaml

from pathlib import Path

from . import config
from .utils.pca import PCAHandler

if __name__ == "__main__":
    # Parameters
    LAYERS = config["layers"]
    COMPONENTS = config["selected_components"]
    for layer_name in LAYERS:
        print(f"Processing {layer_name}")
        pca_model_file_top_10 = Path(config["pca"]["pca_models_top10"]) / f"{layer_name}.pkl"
        pca_model_file_selected = Path(config["pca"]["pca_models"]) / f"{layer_name}.pkl"
        pca_model_file_selected.parent.mkdir(parents=True, exist_ok=True)
        if layer_name not in COMPONENTS:
            print(f"No components specified for {layer_name}, skipping.")
            continue
        selected_indices = COMPONENTS[layer_name]
        pca_handler = PCAHandler(pca_model_file_top_10)
        pca_handler.change_model_top_k_components(selected_indices, 3, pca_model_file_selected)
        print(f"Saved selected components for {layer_name} to {pca_model_file_selected}")
