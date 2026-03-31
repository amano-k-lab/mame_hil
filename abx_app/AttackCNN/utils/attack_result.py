from dataclasses import dataclass, field
from typing import List

import torch

from .activation_manager import ActivationManager
from .decomposition_handler import DecompositionHandler
from .condition import Condition


@dataclass
class AttackResult:
    original_image: torch.Tensor
    original_loss: float
    perturbed_images: List[torch.Tensor] = field(default_factory=list)
    loss_list: List[float] = field(default_factory=list)


def get_actual_value(
    result: AttackResult, activation_manager: ActivationManager, handler: DecompositionHandler, cond: Condition
) -> float:
    original_image = torch.tensor(result.original_image, device=activation_manager.device)
    original_gram_matrix_flat = activation_manager.extract_gram_matrix_flat(original_image, cond.layer)
    if not isinstance(original_gram_matrix_flat, torch.Tensor):
        raise ValueError("Unexpected Error.")
    original_coordinates = handler.transform_coordinate(original_gram_matrix_flat)
    original_value = original_coordinates[0, cond.component]

    perturbed_image = torch.tensor(result.perturbed_images[-1], device=activation_manager.device)
    perturbed_gram_matrix_flat = activation_manager.extract_gram_matrix_flat(perturbed_image, cond.layer)
    if not isinstance(perturbed_gram_matrix_flat, torch.Tensor):
        raise ValueError("Unexpected Error.")
    coordinates = handler.transform_coordinate(perturbed_gram_matrix_flat)
    perturbed_value = coordinates[0, cond.component]
    print(f"perturbed: {perturbed_value}")

    actual_value = perturbed_value - original_value if cond.direction == "plus" else original_value - perturbed_value

    return actual_value.detach().cpu().numpy()
