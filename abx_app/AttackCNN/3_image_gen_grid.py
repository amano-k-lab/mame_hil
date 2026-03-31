from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from abx_app.AttackCNN.attack_examples import run_attack

from . import config
from .utils.activation_manager import ActivationManager
from .utils.pgd_attack import PGDAttack, AttackParams
from .utils.ica import ICAHandler
from .utils.data_utils import prepare_data_from_pool
from .utils.model_utils import load_model
from .utils.attack_result import AttackResult

@dataclass
class ImageGridResult:
    coordinates: List[List[float]]
    original_image: np.ndarray
    images: Dict[str, torch.Tensor]

def save_grid_images(image_grid_result: ImageGridResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    original_image_path = original_dir / "original.png"
    original_image = np.clip(image_grid_result.original_image.squeeze().transpose((1, 2, 0)), 0, 1)
    plt.imsave(str(original_image_path), original_image)

    for key, img_tensor in image_grid_result.images.items():
        image = np.clip(img_tensor.squeeze().transpose((1, 2, 0)), 0, 1) # type: ignore
        path = output_dir / f"{key}.png"
        plt.imsave(str(path), image)

def generate_component_grid(
    data_loader,
    activation_manager,
    model,
    device,
    values=[0, 20, 40, 60, 80],
) -> ImageGridResult:

    original_image, _ = next(iter(data_loader))
    original_image = original_image.clone().detach().cpu().numpy()
    images = {}

    handler = ICAHandler(Path(config["ica"]["ica_models"]) / "conv1.pkl")
    attack_params = AttackParams(**config["img_components"]["attack_params"]["conv1"])

    origin_gram_matrix_flat = torch.tensor(
        activation_manager.extract_gram_matrix_flat(data_loader, "conv1", save_path=None, use_tqdm=False),
        device=device,
    )

    with torch.no_grad():
        coeffs = handler.transform_coordinate(origin_gram_matrix_flat)
        reconstructed = handler.inverse_coordinate(coeffs)
        residual = origin_gram_matrix_flat - reconstructed

    for x in values:
        for y in values:
            coeffs_xy = coeffs.clone()
            coeffs_xy[0, 0] += x
            coeffs_xy[0, 1] += y
            target_gram_flat = handler.inverse_coordinate(coeffs_xy) + residual
            target_gram = activation_manager.reconstruct_gram_matrix(target_gram_flat, "conv1")

            pgd_attack = PGDAttack(
                attack_params=attack_params,
                layer_name="conv1",
                activation_manager=activation_manager,
                handler=handler,
            )
            results: List[AttackResult] = run_attack(
                data_loader,
                pgd_attack,
                target_gram,
                original_layer1_activation=torch.zeros(1),  # dummy
                target_coordinate=coeffs_xy,
            )
            images[f"grid_comp0_{x}_comp1_{y}"] = results[0].perturbed_images[-1]

    return ImageGridResult(coordinates=[values, values], original_image=original_image, images=images)

if __name__ == "__main__":
    config_img = config["img_components"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
    ])
    data_loader = prepare_data_from_pool(1, transform, device, example=True)
    activation_manager = ActivationManager(model, device)

    result = generate_component_grid(data_loader, activation_manager, model, device)
    save_grid_images(result, Path("output/figures/ica_components_img/grid"))
