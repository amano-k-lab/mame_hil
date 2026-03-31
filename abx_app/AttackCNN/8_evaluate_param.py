from itertools import product

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np

from . import config
from .utils.condition import AttackParamsLoader, Condition, TuneCondition
from .utils.model_utils import load_model
import torchvision.transforms as transforms
import torch
from .utils.activation_manager import ActivationManager
from .utils.data_utils import prepare_data_from_pool, get_single_data_loader
from .generate_image_from_condition import generate_image_from_condition
from .utils.attack_result import AttackResult

config_ev = config["eval_param"]

loss_data: dict[str, list[dict[str, list[float]]]] = {}


def check_params(layer: str, num_samples: int, data_loader, seed=42):
    global loss_data

    torch.manual_seed(seed)
    np.random.seed(seed)
    value_orders_log10 = TuneCondition.get_value_orders_log10(layer)
    values = [10**v for v in np.linspace(value_orders_log10[0], value_orders_log10[-1], num_samples)]

    modes = config["modes"]
    components = config["components"]
    directions = config["directions"]

    conditions = [
        Condition(mode, layer, component, direction)
        for mode, component, direction in product(modes, components, directions)
    ]
    sampled_conditions = [conditions[np.random.randint(len(conditions))] for _ in range(num_samples)]

    results: list[tuple[AttackResult, float]] = []
    for i, cond in tqdm(enumerate(sampled_conditions), total=num_samples, desc=f"Processomg Loss of {layer}"):
        single_loader = get_single_data_loader(data_loader, i)
        value = values[i]
        handler = cond.get_decomposition_handler(top10=True)
        param_loader = AttackParamsLoader()
        params = param_loader.get_attack_params(cond, value)

        result = generate_image_from_condition(
            params,
            cond,
            activation_manager.model,
            activation_manager.device,
            single_loader,
            handler,
            value=values[i],
            layer1_name=LAYER1_NAME,
        )
        results.append((result[0], value))

    loss_data[layer] = [
        {
            "normalized_losses": [float(loss / result.original_loss) for loss in result.loss_list],
            "value": [value],
        }
        for result, value in results
    ]


def plot_loss_data(loss_data: dict[str, list[dict[str, list[float]]]]):
    plt.figure(figsize=(24, 16))

    cmap = get_cmap("viridis")

    for i, (layer, loss_entries) in enumerate(tqdm(loss_data.items(), desc="Plotting loss data")):
        title = f"{layer}"
        values = [entry["value"][0] for entry in loss_entries]
        vmin, vmax = min(values), max(values)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        plt.subplot((len(loss_data) + 2) // 3, 3, i + 1)
        for idx, entry in enumerate(loss_entries):
            value = entry["value"][0]
            loss_list = entry["normalized_losses"]
            color = cmap(norm(value))
            plt.plot(range(1, len(loss_list) + 1), loss_list, marker="o", label=f"{value:.2f}", color=color)
        plt.xlabel("Iterations", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.ylim(0, 2)
        plt.title(title, fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.legend(fontsize=14)
        plt.grid(True)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical")
        cbar.set_label("Target Value", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plot_path = Path(config_ev["plot_path"])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    NUM_SAMPLES = config_ev["num_samples"]
    LAYER1_NAME = config["layer1_name"]
    layers = config["layers"]

    param_loader = AttackParamsLoader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    data_loader = prepare_data_from_pool(NUM_SAMPLES, transform, device, seed=42)
    activation_manager = ActivationManager(model, device)

    for layer in layers:
        check_params(layer, NUM_SAMPLES, data_loader)

    plot_loss_data(loss_data)
