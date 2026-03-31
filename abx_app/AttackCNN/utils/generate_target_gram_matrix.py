import numpy as np
from torch.utils.data import DataLoader
import torch

from .activation_manager import ActivationManager
from .decomposition_handler import DecompositionHandler


def generate_target_gram_matrix(
    data_loader: DataLoader,  # 1-image
    layer_name,
    activation_manager: ActivationManager,
    handler: DecompositionHandler,
    component_index=0,
    value=1e5,
):
    origin_gram_matrix_flat = torch.tensor(
        activation_manager.extract_gram_matrix_flat(data_loader, layer_name, save_path=None, use_tqdm=False),
        device=activation_manager.device,
    )
    if origin_gram_matrix_flat is None:
        raise ValueError("Failed to extract gram matrix.")

    with torch.no_grad():
        decomposition_coefficients = handler.transform_coordinate(origin_gram_matrix_flat)
        reconstructed = handler.inverse_coordinate(decomposition_coefficients)
        residual = origin_gram_matrix_flat - reconstructed

        decomposition_coefficients[0, component_index] = decomposition_coefficients[0, component_index] + value
        target_decomposition_inverse = handler.inverse_coordinate(decomposition_coefficients)

        target_gram_matrix_flat = target_decomposition_inverse + residual
        target_gram_matrix = activation_manager.reconstruct_gram_matrix(target_gram_matrix_flat, layer_name)

    return target_gram_matrix, decomposition_coefficients


def generate_original_activation(
    data_loader,
    layer_name,
    activation_manager: ActivationManager,
):
    original_activation_flat = activation_manager.extract_activation_flat(data_loader, layer_name, use_tqdm=False)
    return activation_manager.reconstruct_activation(original_activation_flat, layer_name)
