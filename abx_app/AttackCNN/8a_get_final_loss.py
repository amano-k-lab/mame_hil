from itertools import product
import torch
from tqdm import tqdm
import numpy as np

from . import config
from .utils.condition import AttackParamsLoader, Condition, TuneCondition
from .utils.model_utils import load_model
import torchvision.transforms as transforms
from .utils.activation_manager import ActivationManager
from .utils.data_utils import prepare_data_from_pool, get_single_data_loader
from .generate_image_from_condition import generate_image_from_condition

config_ev = config["eval_param"]

def check_params(layer: str, num_samples: int, data_loader, activation_manager, seed=42):
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

    loss_dict = {
        (cond.component, cond.direction): []
        for cond in conditions
    }

    param_loader = AttackParamsLoader()

    handler = conditions[0].get_decomposition_handler(top10=True)

    for cond in conditions:

        for i in tqdm(range(num_samples), desc=f"Processing {layer}-{cond.component}-{cond.direction}"):
            single_loader = get_single_data_loader(data_loader, i)
            value = values[i]

            params = param_loader.get_attack_params(cond, value)

            result = generate_image_from_condition(
                params,
                cond,
                activation_manager.model,
                activation_manager.device,
                single_loader,
                handler,
                value=value,
                layer1_name=config["layer1_name"],
            )[0]

            final_loss = float(result.loss_list[-1])
            original_loss = float(result.original_loss)
            normalized_loss = final_loss / original_loss

            loss_dict[(cond.component, cond.direction)].append(normalized_loss)

    return loss_dict


if __name__ == "__main__":
    NUM_SAMPLES = config_ev["num_samples"]
    layers = config["layers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    data_loader = prepare_data_from_pool(NUM_SAMPLES, transform, device, seed=42)
    activation_manager = ActivationManager(model, device)

    print("\n===== FINAL LOSS SUMMARY =====\n")

    for layer in layers:
        loss_dict = check_params(layer, NUM_SAMPLES, data_loader, activation_manager)

        for (component, direction), vals in loss_dict.items():
            vals = np.array(vals)
            mean = np.mean(vals)
            std = np.std(vals)

            print(
                f"Layer: {layer}, Component: {component}, Direction: {direction}, "
                f"n={len(vals)}, Mean={mean:.6f}, Std={std:.6f}"
            )