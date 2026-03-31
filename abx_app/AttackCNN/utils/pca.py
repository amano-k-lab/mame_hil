import copy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from .. import config
from .decomposition_handler import DecompositionHandler, V

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCAHandler(DecompositionHandler):
    pca_result: IncrementalPCA
    components_torch: torch.Tensor
    mean_torch: torch.Tensor
    explained_variance_torch: torch.Tensor

    def __init__(self, pca_model_file: Path | None):
        self.pca_model_file = pca_model_file
        self.pca_result: IncrementalPCA

        if pca_model_file and pca_model_file.exists():
            loaded_data = joblib.load(pca_model_file)
            self.pca_result = loaded_data["pca_result"]
            self.components_torch = torch.tensor(self.pca_result.components_, dtype=torch.float32, device=device)
            self.mean_torch = torch.tensor(self.pca_result.mean_, dtype=torch.float32, device=device)
            self.explained_variance_torch = torch.tensor(self.pca_result.explained_variance_, dtype=torch.float32, device=device)
            

    def perform_pca(self, data_file, n_components=None, data=None) -> IncrementalPCA:
        if data is None:
            if data_file and data_file.exists():
                data = np.load(data_file, allow_pickle=True)
            else:
                raise ValueError("Gram matrix data is not provided and data_file is None or does not exists")

        pca = IncrementalPCA(n_components=n_components, whiten=True)
        pca.fit(data)
        self.pca_result = pca
        self.components_torch = torch.tensor(self.pca_result.components_, dtype=torch.float32, device=device)
        self.mean_torch = torch.tensor(self.pca_result.mean_, dtype=torch.float32, device=device)

        if self.pca_model_file:
            model_data = {"pca_result": pca}
            joblib.dump(model_data, self.pca_model_file)
        
        return self.pca_result

    def transform_coordinate(self, data_flat: V):
        if self.pca_result is None:
            raise ValueError("PCAResult is not initialized. Please perform PCA first.")
        if isinstance(data_flat, np.ndarray):
            return self.pca_result.transform(data_flat)
        elif isinstance(data_flat, torch.Tensor):
            data_flat -= self.mean_torch
            transformed = torch.matmul(data_flat, self.components_torch.T)
            if self.pca_result.whiten:
                transformed = transformed / torch.sqrt(self.explained_variance_torch.reshape(1, -1))
            return transformed
    
    def inverse_coordinate(self, coordinates: V):
        if self.pca_result is None:
            raise ValueError("PCAResult is not initialized. Please perform PCA first.")
        if isinstance(coordinates, np.ndarray):
            return self.pca_result.inverse_transform(coordinates)
        elif isinstance(coordinates, torch.Tensor):
            if self.pca_result.whiten:
                coordinates = coordinates * torch.sqrt(self.explained_variance_torch.reshape(1, -1))
            return torch.matmul(coordinates, self.components_torch) + self.mean_torch

    def plot_explained_variance(self, layer_name):
        explained_variance_ratios = self.pca_result.explained_variance_ratio_
        fig, ax1 = plt.subplots(figsize=(10, 6))

        indices = np.arange(1, len(explained_variance_ratios) + 1)

        ax1.bar(
            indices,
            explained_variance_ratios,
            color="m",
            width=0.8,
            alpha=0.7,
            edgecolor="black",
            label="Contribution Ratio",
        )
        ax1.set_xlabel("Independent Components", fontsize=42)
        ax1.set_ylabel("Contribution Ratio", color="m", fontsize=42)
        ax1.tick_params(axis="x", labelsize=32)
        ax1.tick_params(axis="y", labelcolor="m", labelsize=32)
        ax1.set_ylim(0, max(explained_variance_ratios) * 1.1)

        ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        fig.tight_layout()

        plt.title(f"Explained Variance of {layer_name}", fontsize=42)
        title = Path(config["pca"]["pca_explained_variance_fig"]) / f"{layer_name}.png"
        title.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            title,
            bbox_inches="tight",
        )

    def change_model_top_k_components(self, sorted_indices, top_k, pca_model_file):
        if not hasattr(self.pca_result, "components_"):
            raise ValueError("PCAResult is not initialized or invalid. Please perform PCA first.")

        top_indices = sorted_indices[:top_k]
        pca_top_k = copy.deepcopy(self.pca_result)

        pca_top_k.components_ = self.pca_result.components_[top_indices, :]
        pca_top_k.n_components = top_k
        pca_top_k.explained_variance_ = self.pca_result.explained_variance_[top_indices]


        model_data = {"pca_result": pca_top_k}
        joblib.dump(model_data, pca_model_file)




