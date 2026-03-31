from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from . import config
from .utils.activation_manager import ActivationManager
from .utils.pgd_attack import AttackParams
from .utils.ica import ICAHandler
from .utils.data_utils import prepare_data_from_pool
from .utils.model_utils import load_model
from .utils.condition import Condition
from .generate_image_from_condition import generate_image_from_condition


@dataclass
class ImageGenResult:
    num_components: int
    examples: List[Dict[str, torch.Tensor]]
    coordinates: List[float]
    original_image: np.ndarray


def save_images(image_gen_result: ImageGenResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the original image
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    original_image_path = Path(original_dir / "original.png")
    original_image = np.clip(image_gen_result.original_image.squeeze().transpose((1, 2, 0)), 0, 1)

    plt.imsave(str(original_image_path), original_image)

    # Save generated images
    for i, component_results in enumerate(image_gen_result.examples):
        for key, img_tensor in component_results.items():
            layer_name, value = key.split("_")
            value = value.replace("-", "minus").replace("+", "plus")
            file_name = f"{layer_name}_component{i}_{value}.png"
            image_path = output_dir / file_name

            print(img_tensor.shape)
            image = np.clip(img_tensor.squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
            plt.imsave(str(image_path), image)


def generate_component_images(
    data_loader,
    layer_names,
    activation_manager,
    num_components=3,
) -> ImageGenResult:

    examples = []
    coordinates = []
    original_image, _ = next(iter(data_loader))
    original_image = original_image.clone().detach().cpu().numpy()

    for i in range(num_components):
        component_results = {}
        for layer_name in layer_names:
            handler = ICAHandler(Path(config["ica"]["ica_models"]) / f"{layer_name}.pkl")
            attack_params = AttackParams(
                alpha=config["img_components"]["attack_params"][layer_name]["alpha"],
                beta=config["img_components"]["attack_params"][layer_name]["beta"],
                num_iterations=config["img_components"]["attack_params"][layer_name]["num_iterations"],
            )

            original_gram_matrix_flat = activation_manager.extract_gram_matrix_flat(
                data_loader, layer_name, save_path=None, use_tqdm=False
            )
            coordinates_layer = next(iter(handler.transform_coordinate(original_gram_matrix_flat))).tolist()
            coordinates.append(coordinates_layer[i])

            values = [-80, 80] if layer_name == "conv1" else [-0.8, 0.8] if layer_name == "layer3" else [-0.3, 0.3]

            for value in values:
                cond = Condition(
                    ecc=3, mode="ica", layer=layer_name, component=i, direction="plus" if value > 0 else "minus"
                )

                results = generate_image_from_condition(
                    attack_params=attack_params,
                    cond=cond,
                    model=model,
                    device=device,
                    data_loader_origin=data_loader,
                    handler=handler,
                    value=abs(value),
                    layer1_name="conv1",
                )

                generated_image = results[0].perturbed_images[-1]
                component_results[f"{layer_name}_{value}"] = generated_image

        examples.append(component_results)

    return ImageGenResult(
        num_components=num_components,
        examples=examples,
        coordinates=coordinates,
        original_image=original_image,
    )


if __name__ == "__main__":
    config_img = config["img_components"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LAYERS = ["conv1", "layer3", "avgpool"]

    model = load_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    data_loader = prepare_data_from_pool(1, transform, device, example=True)
    activation_manager = ActivationManager(model, device)

    image_gen_result = generate_component_images(
        data_loader=data_loader,
        layer_names=LAYERS,
        activation_manager=activation_manager,
    )

    output_directory = Path("output/figures/ica_components_img/example")
    save_images(image_gen_result, output_directory)
