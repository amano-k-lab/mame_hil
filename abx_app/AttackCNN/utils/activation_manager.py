from typing import Union

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .decomposition_handler import V


class ActivationManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_activation(self, input: V, LAYER_NAME) -> V:
        raw_activation = {}

        def hook(module, input, output):
            raw_activation[LAYER_NAME] = output

        layer = dict(self.model.named_modules())[LAYER_NAME]
        hook_handle = layer.register_forward_hook(hook)

        _ = self.model(input)
        hook_handle.remove()

        return raw_activation[LAYER_NAME]

    @staticmethod
    def calculate_gram_matrix(activation):
        batch_size, num_filters, height, width = activation.shape
        reshaped_activation = activation.view(
            batch_size, num_filters, -1
        )  # reshape to [batch_size, num_filters, height * width]

        # G^l_ij = F^l_ik * F^l_jk
        gram_matrix = torch.einsum("bij,bkj->bik", reshaped_activation, reshaped_activation)
        return gram_matrix

    def extract_gram_matrix_flat(
        self, data: Union[DataLoader, torch.Tensor], layer_name, save_path=None, use_tqdm=True
    ):
        all_gram_matrices = []

        if isinstance(data, DataLoader):
            with torch.no_grad():
                iterator = tqdm(data, desc="Extracting Gram matrix") if use_tqdm else data
                for images, _ in iterator:
                    images = images.to(self.device)
                    activation = self.get_activation(images, layer_name)
                    gram_matrix = self.calculate_gram_matrix(activation)
                    gram_matrix_flat = gram_matrix.cpu().numpy().reshape(len(images), -1)

                    all_gram_matrices.append(gram_matrix_flat)
        else:  # batch data
            images = data.to(self.device)
            if len(images) > 1:
                with torch.no_grad():
                    activation = self.get_activation(images, layer_name)
                    gram_matrix = self.calculate_gram_matrix(activation)
                    gram_matrix_flat = gram_matrix.cpu().numpy().reshape(len(images), -1)
                    all_gram_matrices.append(gram_matrix_flat)
            else:
                activation = self.get_activation(images, layer_name)
                gram_matrix = self.calculate_gram_matrix(activation)
                gram_matrix_flat = gram_matrix.view(len(images), -1)
                return gram_matrix_flat

        all_gram_matrices = np.vstack(all_gram_matrices)

        if save_path:
            np.save(save_path, all_gram_matrices)
        else:
            return all_gram_matrices

    def extract_activation_flat(self, data_loader, layer_name, save_path=None, use_tqdm=True):
        all_activations = []

        with torch.no_grad():
            iterator = tqdm(data_loader, desc="Extracting Activation") if use_tqdm else data_loader
            for data, _ in iterator:
                data = data.to(self.device)
                activation = self.get_activation(data, layer_name)
                activation_flat = activation.cpu().numpy().reshape(len(data), -1)

                all_activations.append(activation_flat)

            all_activations = np.vstack(all_activations)

            if save_path:
                np.save(save_path, all_activations)
            else:
                return all_activations

    def reconstruct_gram_matrix(self, gram_matrix_flat: V, layer_name) -> V:
        # Get the original activation shape by passing a dummy input through the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            original_activation = self.get_activation(dummy_input, layer_name)
            _, channels, _, _ = original_activation.shape

        if isinstance(gram_matrix_flat, np.ndarray):
            gram_matrix = gram_matrix_flat.reshape(channels, channels)
            gram_matrix = np.expand_dims(gram_matrix, axis=0)
            return gram_matrix.astype(np.float32)
        else:
            gram_matrix = gram_matrix_flat.reshape(channels, channels)
            gram_matrix = gram_matrix.unsqueeze(0).to(self.device).to(torch.float32)
            return gram_matrix

    def reconstruct_activation(self, activation_flat, layer_name):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            original_activation = self.get_activation(dummy_input, layer_name)
            _, channels, height, width = original_activation.shape

        activation = activation_flat.reshape(channels, height, width)
        activation = torch.from_numpy(activation).to(self.device).unsqueeze(0).to(torch.float)
        return activation
