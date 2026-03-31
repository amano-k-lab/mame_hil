from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from . import config
from .utils.activation_manager import ActivationManager
from .utils.pgd_attack import AttackParams
from .utils.pca import PCAHandler
from .utils.ica import ICAHandler
from .utils.decomposition_handler import DecompositionHandler
from .utils.data_utils import prepare_data_from_pool
from .utils.model_utils import load_model
from .utils.condition import Condition
from .utils.attack_result import get_actual_value
from .generate_image_from_condition import generate_image_from_condition


@dataclass
class ImageGenResult:
    num_components: int
    examples: List[Dict[str, torch.Tensor]]
    coordinates: List[float]
    alpha: float
    layer_name: str


def generate_component_images(
    attack_params: AttackParams,
    handler: DecompositionHandler,
    data_loader,
    layer_name,
    layer1_name,
    activation_manager,
    num_components=None,
) -> ImageGenResult:

    dec_components = None
    mode = ""
    if isinstance(handler, PCAHandler):
        dec_result = handler.pca_result
        if dec_result is None:
            raise ValueError("PCA result does not exists. perform PCA.")
        dec_components = dec_result.components_.shape[0]
        mode = "pca"
    elif isinstance(handler, ICAHandler):
        dec_result = handler.ica_result
        if dec_result is None:
            raise ValueError("ICA result does not exists. perform ICA.")
        dec_components = dec_result.components_.shape[0]
        mode = "ica"
    if dec_components is None:
        raise ValueError("set decomposition model of `pca` or `ica`")

    if num_components is None:
        num_components = min(dec_components, config_img["num_components_max"])

    original_image, _ = next(iter(data_loader))
    original_image = original_image.clone().detach().cpu().numpy()

    original_gram_matrix_flat = activation_manager.extract_gram_matrix_flat(data_loader, layer_name, save_path=None)

    coordinates = next(iter(handler.transform_coordinate(original_gram_matrix_flat))).tolist()[:num_components]


    examples = []

    for i in tqdm(range(num_components), desc="Processing Components", total=num_components):
        component_results = {}

        values = config_img["target_values"][layer_name]["values"]
        for value in values:
            cond = Condition(
                ecc=3, mode=mode, layer=layer_name, component=i, direction="plus" if value > 0 else "minus"
            )

            results = generate_image_from_condition(
                attack_params=attack_params,
                cond=cond,
                model=model,
                device=device,
                data_loader_origin=data_loader,
                handler=handler,
                value=abs(value),
                layer1_name=layer1_name,
            )

            result = results[0]
            # Debug: retrieve and log the actual value
            print(f"target: {value}")
            print(f"actual: {get_actual_value(result, activation_manager, handler=handler, cond=cond)}")

            generated_image = result.perturbed_images[-1]
            component_results[str(value)] = generated_image

        component_results["original"] = original_image
        examples.append(component_results)

    return ImageGenResult(
        num_components=num_components,
        examples=examples,
        coordinates=coordinates,
        alpha=attack_params.alpha,
        layer_name=layer_name,
    )


def visualize(image_gen_result: ImageGenResult, model_type: str):
    fig, ax = plt.subplots(
        image_gen_result.num_components,
        5,
        figsize=(15, 3 * (image_gen_result.num_components)),
    )

    # fig.suptitle(
    #     f"num_components: {image_gen_result.num_components}",
    #     x=0.5,
    #     ha="center",
    #     fontsize=16,
    # )
    keys_order = config_img["target_values"][LAYER_NAME]["keys_order"]

    for j, key in enumerate(keys_order):
        ax[0, j].text(
            0.5,
            1.1,
            f"{key}",
            ha="center",
            va="center",
            fontsize=20,
            transform=ax[0, j].transAxes,
        )
        ax[0, j].axis("off")

    for i, component_results in enumerate(image_gen_result.examples):
        for j, key in enumerate(keys_order):
            img_tensor = component_results[key]
            image = np.clip(img_tensor.squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
            ax[i, j].imshow(image)
            ax[i, j].axis("off")

        ax[i, 0].text(
            0.0,
            0.5,
            f"Comp{i}: {image_gen_result.coordinates[i]:.4f}",
            ha="right",
            va="center",
            transform=ax[i, 0].transAxes,
            fontsize=20,
            color="blue",
            rotation=90,
        )

    plt.tight_layout()
    if model_type == "ica":
        title = Path(config_img["ica_components_fig"]) / f"{image_gen_result.num_components}" / f"{LAYER_NAME}.png"
    else:
        title = Path(config_img["pca_components_fig"]) / f"{image_gen_result.num_components}" / f"{LAYER_NAME}.png"
    title.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(title)


if __name__ == "__main__":
    config_img = config["img_components"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="number of components")
    parser.add_argument(
        "--layer_name",
        type=str,
        required=True,
        help="The name of the layer for which to generate images.",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=None,
        help="Number of components to use (overrides config if specified).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ica", "pca"],
        default="ica",
        help="Choose model type to use: 'ica' or 'pca'."
    )
    parser.add_argument(
        "--model_file",
        type=Path,
        default=None,
        help="Path to the decomposition model file (overrides config if specified).",
    )
    args = parser.parse_args()

    LAYER_NAME = args.layer_name
    LAYER1_NAME = config["layer1_name"]
    NUM_COMPONENTS = args.num_components if args.num_components else config_img["num_components_default"]
    DECOMPOSITION_MODEL_FILE = (
        args.model_file if args.model_file else Path(config["ica"]["ica_models"]) / f"{LAYER_NAME}.pkl"
    )
    # Parameters
    ATTACK_PARAMS = AttackParams(
        alpha=config_img["attack_params"][LAYER_NAME]["alpha"],
        beta=config_img["attack_params"][LAYER_NAME]["beta"],
        num_iterations=config_img["attack_params"][LAYER_NAME]["num_iterations"],
    )

    model = load_model(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    data_loader = prepare_data_from_pool(1, transform, device, example=True)
    if args.model_type == "ica":
        handler = ICAHandler(DECOMPOSITION_MODEL_FILE)
    else:
        handler = PCAHandler(DECOMPOSITION_MODEL_FILE)
    activation_manager = ActivationManager(model, device)

    image_gen_result = generate_component_images(
        attack_params=ATTACK_PARAMS,
        data_loader=data_loader,
        layer_name=LAYER_NAME,
        layer1_name=LAYER1_NAME,
        handler=handler,
        activation_manager=activation_manager,
        num_components=NUM_COMPONENTS,
    )

    visualize(image_gen_result, args.model_type)
