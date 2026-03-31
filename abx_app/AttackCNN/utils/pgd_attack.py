from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .activation_manager import ActivationManager
from .decomposition_handler import DecompositionHandler


@dataclass
class AttackParams:
    alpha: float
    beta: float = 0.01
    num_iterations: int = 50


class PGDAttack:
    def __init__(
        self,
        attack_params: AttackParams,
        layer_name,
        activation_manager: ActivationManager,
        layer1_name=None,
        handler: Optional[DecompositionHandler] = None,
    ):
        self.model = activation_manager.model
        self.device = activation_manager.device
        self.alpha = attack_params.alpha
        self.num_iterations = attack_params.num_iterations
        self.layer_name = layer_name
        self.activation_manager = activation_manager
        self.layer1_name = layer1_name
        self.beta = attack_params.beta
        self.handler = handler

    def attack(
        self,
        input_image,
        target_gram_matrix,
        original_layer1_activation=None,
        target_coordinate: Optional[torch.Tensor] = None,
    ):
        perturbed_image = input_image.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([perturbed_image], lr=self.alpha)
        perturbed_image_list = []
        pred_list = []
        loss_list = []
        loss = torch.tensor(0.0, device=self.device)

        for iteration in range(self.num_iterations):
            if iteration > 0:
                loss_list.append(loss.item())

            loss = self._loss(perturbed_image, target_gram_matrix, original_layer1_activation, target_coordinate)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                perturbed_image.clamp_(0, 1)

            pred = self.model(perturbed_image).max(1, keepdim=True)[1]
            pred_list.append(pred.squeeze().detach().cpu().numpy())
            perturbed_image_list.append(perturbed_image.clone().detach().cpu().numpy())

        final_loss = self._loss(perturbed_image, target_gram_matrix, original_layer1_activation, target_coordinate)
        loss_list.append(final_loss.item())

        return perturbed_image, perturbed_image_list, pred_list, loss_list

    def _loss(
        self,
        input_image: torch.Tensor,
        target_gram_matrix,
        original_layer1_activation=None,
        target_coordinate: Optional[torch.Tensor] = None,
    ):
        # current_activation = self.activation_manager.get_activation(input_image, self.layer_name)
        # current_gram_matrix = self.activation_manager.calculate_gram_matrix(current_activation)
        # gram_loss = nn.MSELoss()(current_gram_matrix, target_gram_matrix)

        if target_coordinate is not None and self.handler is not None:
            original_gram_matrix_flat = self.activation_manager.extract_gram_matrix_flat(input_image, self.layer_name)
            if not isinstance(original_gram_matrix_flat, torch.Tensor):
                raise ValueError("Unexpected Error.")
            original_coordinates = self.handler.transform_coordinate(original_gram_matrix_flat)
            ica_loss = nn.MSELoss()(original_coordinates, target_coordinate)
        else:
            raise ValueError("Error")

        # current_layer1_activation = self.activation_manager.get_activation(input_image, self.layer1_name)
        # content_loss = nn.MSELoss()(current_layer1_activation, original_layer1_activation)
        # content_loss = 0.0

        loss = ica_loss
        return loss
