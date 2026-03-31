from itertools import product

import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from . import config
from .utils.condition import AttackParamsLoader
from .utils.model_utils import load_model
from .utils.activation_manager import ActivationManager
from .utils.data_utils import prepare_data_from_pool
from .generate_image_from_condition import generate_image_from_condition
from .utils.attack_result import get_actual_value
from .utils.condition import Condition, TuneCondition

config_sp = config["spearman_test"]


def generate_random_target_values(value_orders_log10, num_samples, seed=42):
    np.random.seed(seed)
    min_value = value_orders_log10[0]
    max_value = value_orders_log10[-1]

    target_values_log10 = np.random.uniform(low=min_value, high=max_value, size=num_samples)
    return 10**target_values_log10


def generate_actual_values(target_values, cond: Condition, param_loader: AttackParamsLoader, transform, device, model):
    activation_manager = ActivationManager(model, device)
    actual_values = []
    for target_value in target_values:
        data_loader = prepare_data_from_pool(1, transform, device, seed=None)
        param = param_loader.get_attack_params(cond, target_value)
        handler = cond.get_decomposition_handler(top10=False)

        results = generate_image_from_condition(
            attack_params=param,
            cond=cond,
            model=model,
            device=device,
            data_loader_origin=data_loader,
            handler=handler,
            value=target_value,
            layer1_name=config["layer1_name"],
        )
        actual_value = get_actual_value(results[0], activation_manager, handler, cond)
        actual_values.append(actual_value)

    return actual_values


def plot_test(target_values, actual_values, cond: Condition):
    plt.figure(figsize=(8, 6))
    plt.scatter(target_values, actual_values, c="blue")
    plt.plot(
        [min(target_values), max(target_values)],
        [min(target_values), max(target_values)],
        color="red",
        linestyle="--",
        label="y=x (Ideal Line)",
    )

    plt.title(f"{cond.layer}, component: {cond.component}, {cond.direction}", fontsize=16)
    plt.xlabel("Target Values", fontsize=20)
    plt.ylabel("Actual Values", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(f"output/figures/spearman_test_{cond.layer}_{cond.component}_{cond.direction}.png")


def compute_spearman_correlation(target_values, actual_values):
    rho, p_value = spearmanr(target_values, actual_values)
    return rho, p_value


if __name__ == "__main__":
    NUM_SAMPLES = config_sp["num_samples"]

    modes = config["modes"]
    components = config["components"]
    directions = config["directions"]
    layers = config["layers"]

    conditions = [
        Condition(mode, layer, component, direction)
        for mode, layer, component, direction in product(modes, layers, components, directions)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    param_loader = AttackParamsLoader()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    for cond in conditions:
        value_orders_log10 = TuneCondition.get_value_orders_log10(cond.layer, border=True)
        target_values = generate_random_target_values(value_orders_log10, NUM_SAMPLES)
        actual_values = generate_actual_values(target_values, cond, param_loader, transform, device, model)
        rho, p_value = compute_spearman_correlation(target_values, actual_values)
        plot_test(target_values, actual_values, cond)
        print(f"{cond}: Spearman's rho = {rho:.4f}, p-value = {p_value:.4e}")
