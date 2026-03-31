from typing import Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .utils.activation_manager import ActivationManager
from .utils.pgd_attack import PGDAttack, AttackParams
from .attack_examples import run_attack
from .utils.data_utils import prepare_data, get_single_data_loader
from .utils.model_utils import load_model
from .utils.visualization import plot_loss_trend, show_adversarial_examples
from .utils.generate_target_gram_matrix import generate_target_gram_matrix, generate_original_activation
from .utils.decomposition_handler import DecompositionHandler
from .utils.condition import Condition
from .utils.attack_result import AttackResult


def generate_image_from_condition(
    attack_params: AttackParams,
    cond: Condition,
    model,
    device,
    data_loader_origin: DataLoader,
    handler: DecompositionHandler,
    value: Optional[float] = None,
    conds: Optional[list[Condition]] = None,
    values: Optional[list[float]] = None,
    layer1_name: str = "conv1",
) -> list[AttackResult]:
    """
    Perform an attack based on the given parameters and return the results.

    Args:
        attack_params (AttackParams): Attack parameters such as alpha, epsilon, etc.
        cond (Condition): Experimental condition including mode, layer, component, and direction.
        model: The target model.
        device: The device (CPU or CUDA).
        data_loader_origin: DataLoader for original (input) data.
        decomposition_model_path (Path): Path to the PCA/ICA decomposition model.
        value (float): Target (unsigned) value for generating the gram matrix.
        layer1_name (str, optional): Name of the layer1 for generating original activations. Default is "conv1".

    Returns:
        list: Results of the attack containing perturbed images and loss trends.
    """
    if values is not None and len(values) != len(data_loader_origin):
        raise ValueError("The length of 'random_values' must match the size of 'data_loader_origin'.")

    activation_manager = ActivationManager(model, device)

    results: list[AttackResult] = []

    if value is not None:
        # Generate 1 image with 1 value
        signed_value = value if cond.direction == "plus" else -value
        target_gram_matrix, target_coordinate = generate_target_gram_matrix(
            data_loader_origin,
            cond.layer,
            activation_manager,
            handler,
            component_index=cond.component,
            value=signed_value,
        )
        # Generate original layer1 activation
        original_layer1_activation = generate_original_activation(data_loader_origin, layer1_name, activation_manager)

        # Initialize PGD attack
        pgd_attack = PGDAttack(
            attack_params=attack_params,
            layer_name=cond.layer,
            activation_manager=activation_manager,
            layer1_name=layer1_name,
            handler=handler,
        )

        # Run the attack
        results = run_attack(
            data_loader_origin,
            pgd_attack,
            target_gram_matrix,
            original_layer1_activation,
            target_coordinate=target_coordinate,
        )

    elif values is not None:
        # generate multiple images from multiple values
        for item_idx, item in enumerate(data_loader_origin):
            single_loader = get_single_data_loader(data_loader_origin, item_idx)

            random_value = values[item_idx]
            if conds:
                cond = conds[item_idx]
            signed_value = random_value if cond.direction == "plus" else -random_value

            target_gram_matrix, target_coordinate = generate_target_gram_matrix(
                single_loader,
                cond.layer,
                activation_manager,
                handler,
                component_index=cond.component,
                value=signed_value,
            )
            original_layer1_activation = generate_original_activation(single_loader, layer1_name, activation_manager)

            pgd_attack = PGDAttack(
                attack_params=attack_params,
                layer_name=cond.layer,
                activation_manager=activation_manager,
                layer1_name=layer1_name,
                handler=handler,
            )

            result = run_attack(
                single_loader,
                pgd_attack,
                target_gram_matrix,
                original_layer1_activation,
                target_coordinate=target_coordinate,
            )
            results.extend(result)

    else:
        raise ValueError("'value' or 'random_values' must be provided.")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Parameters
    attack_params = AttackParams(
        alpha=0.001,
        num_iterations=50,
        beta=1,
    )

    # Condition
    cond = Condition(
        ecc=3,
        mode="ica",
        layer="avgpool",
        component=0,
        direction="minus",
    )
    VALUE = 100

    LAYER1_NAME = "conv1"

    model = load_model(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    data_loader, _ = prepare_data(1, transform, device)

    handler = cond.get_decomposition_handler()

    results = generate_image_from_condition(
        attack_params=attack_params,
        cond=cond,
        model=model,
        device=device,
        data_loader_origin=data_loader,
        value=VALUE,
        handler=handler,
        layer1_name="conv1",
    )

    print(results[0].loss_list)

    show_adversarial_examples(
        results,
        None,
        cond.layer,
        attack_params.num_iterations,
        attack_params.alpha,
        f"abx_app/AttackCNN/{cond.mode}_imggenerate_{cond.layer}_comp_{cond.component}.png",
    )
    plot_loss_trend(
        results,
        attack_params.num_iterations,
        cond.layer,
        f"abx_app/AttackCNN/loss_{cond.mode}_{cond.layer}_comp_{cond.component}",
    )
