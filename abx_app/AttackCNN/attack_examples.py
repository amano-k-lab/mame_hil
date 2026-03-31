import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from .utils.activation_manager import ActivationManager
from .utils.pgd_attack import PGDAttack, AttackParams
from .utils.attack_result import AttackResult
from .utils.data_utils import prepare_data
from .utils.model_utils import load_model
from .utils.visualization import (
    plot_loss_trend,
    show_adversarial_examples,
)


def run_attack(
    data_loader,
    pgd_attack: PGDAttack,
    target_gram_matrix,
    original_layer1_activation=None,
    use_tqdm=False,
    target_coordinate=None,
) -> list[AttackResult]:
    results = []
    total_samples = len(data_loader.dataset)
    device = pgd_attack.device

    iterator = tqdm(data_loader, total=total_samples, desc="Processing", unit="Sample") if use_tqdm else data_loader

    for batch_idx, (input_image, label) in enumerate(iterator):
        start_time = time.time()
        input_image, label = input_image.to(device), label.to(device)

        original_image = input_image.clone().detach().cpu().numpy()

        original_loss = pgd_attack._loss(input_image, target_gram_matrix, original_layer1_activation, target_coordinate)

        _, perturbed_image_list, _, loss_list = pgd_attack.attack(
            input_image, target_gram_matrix, original_layer1_activation, target_coordinate
        )

        attack_result = AttackResult(
            original_image=original_image,
            original_loss=original_loss,
            perturbed_images=perturbed_image_list,
            loss_list=loss_list,
        )
        results.append(attack_result)

        if use_tqdm:
            iterator.update(len(input_image))
            print(f"Image {batch_idx + 1}/{total_samples} processed in {time.time() - start_time:.2f} seconds.")
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    ALPHA = 0.03
    NUM_ITERATIONS = 14
    LAYER_NAME = "layer4.0.conv3"
    NUM_SAMPLES = 3

    model = load_model(device)
    input_size = (3, 224, 224)
    # summary(model, input_size=input_size)
    # for name, module in model.named_modules():
    #     print(name)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_loader, target_image = prepare_data(NUM_SAMPLES, transform, device)
    activation_manager = ActivationManager(model, device)
    target_activation = activation_manager.get_activation(target_image, LAYER_NAME)
    target_gram_matrix = activation_manager.calculate_gram_matrix(target_activation)

    attack_params = AttackParams(
        alpha=ALPHA,
        num_iterations=NUM_ITERATIONS,
    )

    pgd_attack = PGDAttack(
        attack_params=attack_params,
        layer_name=LAYER_NAME,
        activation_manager=activation_manager,
    )

    results = run_attack(data_loader, pgd_attack, target_gram_matrix, original_layer1_activation=None)

    show_adversarial_examples(results, target_image, LAYER_NAME, NUM_ITERATIONS, ALPHA)
    plot_loss_trend(results, NUM_ITERATIONS, LAYER_NAME)
