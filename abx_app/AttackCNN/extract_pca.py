from pathlib import Path

import torch

from .utils.pca import PCAHandler


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    LAYERS = ["conv1", "layer1", "layer3", "avgpool"]

    for layer_name in LAYERS:
        print(f"Getting gram matrix of {layer_name}")
        gram_matrix_file = Path(f"abx_app/AttackCNN/gram_matrix_{layer_name}.npy")
        pca_model_file = Path(f"abx_app/AttackCNN/pca_gram_matrix_{layer_name}.pkl")
        print(f"Perform PCA of {layer_name}")
        pca_handler = PCAHandler(pca_model_file)
        # pca_result = pca_handler.perform_pca(gram_matrix_file, explained_variance=0.95)
        pca_result = pca_handler.perform_pca(gram_matrix_file, n_components=3)
        print(f"PCA completed")
        # pca_handler.plot_explained_variance(layer_name, "gram")
        # print(f"visualized explained variance of {layer_name}")
