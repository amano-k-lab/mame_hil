import copy
from pathlib import Path
import yaml


import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
import torch

from .. import config
from .decomposition_handler import DecompositionHandler, V

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ICAHandler(DecompositionHandler):
    ica_result: FastICA
    components_torch: torch.Tensor
    mean_torch: torch.Tensor
    mixing_torch: torch.Tensor

    def __init__(self, ica_model_file: Path | None):
        self.ica_model_file = ica_model_file
        self.ica_result: FastICA

        if ica_model_file and ica_model_file.exists():
            loaded_data = joblib.load(ica_model_file)
            self.ica_result = loaded_data["ica_result"]
            self.components_torch = torch.tensor(self.ica_result.components_, dtype=torch.float32, device=device)
            self.mean_torch = torch.tensor(self.ica_result.mean_, dtype=torch.float32, device=device)
            self.mixing_torch = torch.tensor(self.ica_result.mixing_, dtype=torch.float32, device=device)

    def perform_ica(self, data_file, n_components=None, data=None, random_seed=42):
        if data is None:
            if data_file and data_file.exists():
                data = np.load(data_file, allow_pickle=True)
            else:
                raise ValueError("Gram matrix data is not provided and data_file is None or does not exists")

        ica = FastICA(
            n_components=n_components, whiten="unit-variance", max_iter=500, tol=1e-3, random_state=random_seed
        )
        ica.fit(data)
        self.ica_result = ica
        self.components_torch = torch.tensor(self.ica_result.components_, dtype=torch.float32, device=device)
        self.mean_torch = torch.tensor(self.ica_result.mean_, dtype=torch.float32, device=device)
        self.mixing_torch = torch.tensor(self.ica_result.mixing_, dtype=torch.float32, device=device)

        if self.ica_model_file:
            model_data = {"ica_result": ica}
            joblib.dump(model_data, self.ica_model_file)

        return self.ica_result

    def transform_coordinate(self, data_flat: V):
        if self.ica_result is None:
            raise ValueError("ICAResult is not initialized. Please perform ICA first.")
        if isinstance(data_flat, np.ndarray):
            return self.ica_result.transform(data_flat)
        elif isinstance(data_flat, torch.Tensor):
            data_flat -= self.mean_torch
            return torch.matmul(data_flat, self.components_torch.T)
        else:
            raise ValueError("Input must be np.ndarray or torch.Tensor")

    def inverse_coordinate(self, coordinates: V):
        if self.ica_result is None:
            raise ValueError("ICAResult is not initialized. Please perform ICA first.")
        if isinstance(coordinates, np.ndarray):
            return self.ica_result.inverse_transform(coordinates)
        elif isinstance(coordinates, torch.Tensor):
            coordinates = torch.matmul(coordinates, self.mixing_torch.T)
            return coordinates + self.mean_torch

    def extract_explained_variance(self, train_data_file):
        if train_data_file and train_data_file.exists():
            X = np.load(train_data_file, allow_pickle=True)
        else:
            raise ValueError("Gram matrix data is not provided and data_file is None or does not exists")
        S = self.ica_result.transform(X)
        X_centered = X - self.ica_result.mean_
        A = self.ica_result.mixing_

        explained_variance_ratios = []

        for i in range(S.shape[1]):
            S_i = np.zeros_like(S)
            S_i[:, i] = S[:, i]

            X_hat_i = np.dot(S_i, A.T)

            explained_variance_i = (
                1 - np.linalg.norm(X_centered - X_hat_i, ord="fro") ** 2 / np.linalg.norm(X_centered, ord="fro") ** 2
            )
            explained_variance_ratios.append(explained_variance_i)

        sorted_indices = np.argsort(explained_variance_ratios)[::-1]
        sorted_ratios = np.array(explained_variance_ratios)[sorted_indices]

        return sorted_ratios, sorted_indices

    def plot_explained_variance(self, explained_variance_ratios, layer_name):
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
        title = Path(config["ica"]["ica_explained_variance_fig"]) / f"{layer_name}.png"
        title.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            title,
            bbox_inches="tight",
        )

    def change_model_top_k_components(self, sorted_indices, top_k, ica_model_file):
        if not hasattr(self.ica_result, "components_"):
            raise ValueError("ICAResult is not initialized or invalid. Please perform ICA first.")

        top_indices = sorted_indices[:top_k]

        ica_top_k = copy.deepcopy(self.ica_result)

        ica_top_k.components_ = self.ica_result.components_[top_indices, :]
        ica_top_k.mixing_ = self.ica_result.mixing_[:, top_indices]
        ica_top_k.n_components = top_k  # type: ignore

        model_data = {"ica_result": ica_top_k}
        joblib.dump(model_data, ica_model_file)

    def change_model_selected_components(self, selected_indices, ica_model_file):
        if not hasattr(self.ica_result, "components_"):
            raise ValueError("ICAResult is not initialized or invalid. Please perform ICA first.")

        ica_selected = copy.deepcopy(self.ica_result)

        ica_selected.components_ = self.ica_result.components_[selected_indices, :]
        ica_selected.mixing_ = self.ica_result.mixing_[:, selected_indices]
        ica_selected.n_components = len(selected_indices)  # type: ignore

        # Save the updated ICA model
        model_data = {"ica_result": ica_selected}
        joblib.dump(model_data, ica_model_file)

        print(f"Saved updated ICA model with selected components to {ica_model_file}")
