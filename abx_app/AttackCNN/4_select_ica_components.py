import yaml

from pathlib import Path

from . import config
from .utils.ica import ICAHandler

if __name__ == "__main__":
    # Parameters
    LAYERS = config["layers"]
    COMPONENTS = config["selected_components"]
    for layer_name in LAYERS:
        print(f"Processing {layer_name}")

        # ica_model_file_top_10 = Path(config["ica"]["ica_models_top10"]) / f"{layer_name}.pkl"
        # ica_model_file_selected = Path(config["ica"]["ica_models"]) / f"{layer_name}.pkl"

        # ica_model_file_top_10 = Path(config["ica"]["ica_models_default_top10"]) / f"{layer_name}.pkl"
        # ica_model_file_selected = Path(config["ica"]["ica_models_default"]) / f"{layer_name}.pkl"

        ica_model_file_top_10 = Path(config["ica"]["ica_models_stylized_top10"]) / f"{layer_name}.pkl"
        ica_model_file_selected = Path(config["ica"]["ica_models_stylized"]) / f"{layer_name}.pkl"

        ica_model_file_selected.parent.mkdir(parents=True, exist_ok=True)
        if layer_name not in COMPONENTS:
            print(f"No components specified for {layer_name}, skipping.")
            continue
        selected_indices = COMPONENTS[layer_name]
        ica_handler = ICAHandler(ica_model_file_top_10)
        ica_handler.change_model_selected_components(selected_indices, ica_model_file_selected)
        print(f"Saved selected components for {layer_name} to {ica_model_file_selected}")
