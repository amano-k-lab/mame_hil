import numpy as np
import optuna
from torch.utils.data import DataLoader
import json
import argparse

import torchvision.transforms as transforms
import torch

from . import config
from .utils.condition import TuneCondition
from .utils.model_utils import load_model
from .utils.pgd_attack import AttackParams
from .utils.attack_result import AttackResult
from .utils.activation_manager import ActivationManager
from .utils.data_utils import prepare_data_from_pool, get_single_data_loader
from .generate_image_from_condition import generate_image_from_condition

config_tu = config["tune_param"]
VALUE_ORDER_INTERVAL: dict[str, float] = config_tu["value_step_log10"]


def evaluate_param_set(
    param_set: AttackParams,
    tune_cond: TuneCondition,
    data_loader: DataLoader,
    activation_manager: ActivationManager,
    seed=42,
) -> float:
    """
    Args:
        param_set (AttackParams): parameter set of evaluation target.
        tune_cond (TuneCondition): experiamental condition with value order.
        data_loader,
        activation_manager (ActivationManager)
        seed (int, optional): random seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    values = tune_cond.sample_value_from_order(len(data_loader))
    cond_list = tune_cond.gen_condition(len(data_loader))

    results: list[AttackResult] = []
    for i, cond in enumerate(cond_list):
        single_loader = get_single_data_loader(data_loader, i)
        handler = cond.get_decomposition_handler(top10=USE_TOP10)

        result = generate_image_from_condition(
            param_set,
            cond,
            activation_manager.model,
            activation_manager.device,
            single_loader,
            handler,
            value=values[i],
            layer1_name=LAYER1_NAME,
        )
        results.extend(result)

    loss_max = 0.0
    for result in results:
        loss = result.loss_list[-1] / result.original_loss
        loss_max = max(loss, loss_max)
    return loss_max


def get_init_alpha(value_order: float, layer_name):
    return 0.01


def optimize_params(
    tune_cond: TuneCondition,
    data_loader: DataLoader,
    activation_manager,
    n_trials,
    seed=42,
) -> AttackParams:

    beta = config["img_components"]["attack_params"][tune_cond.layer]["beta"]
    num_iterations = config["img_components"]["attack_params"][tune_cond.layer]["num_iterations"]

    initial_alpha = get_init_alpha(tune_cond.value_order, tune_cond.layer)
    search_range_log10 = config_tu["search_range_log10"][tune_cond.layer]

    def objective(trial):
        alpha = trial.suggest_float(
            "alpha",
            initial_alpha * (10 ** search_range_log10[0]),
            initial_alpha * (10 ** search_range_log10[1]),
            log=True,
        )
        param_set = AttackParams(alpha=alpha, num_iterations=num_iterations, beta=beta)

        return evaluate_param_set(param_set, tune_cond, data_loader, activation_manager, seed)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"best params: {best_params}")
    return AttackParams(
        alpha=best_params["alpha"],
        beta=beta,
        num_iterations=num_iterations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ica model file")
    parser.add_argument(
        "--ica_model_top10",
        type=bool,
        default=None,
    )
    args = parser.parse_args()
    USE_TOP10 = args.ica_model_top10 if args.ica_model_top10 is not None else False
    NUM_SAMPLES = config_tu["num_samples"]
    LAYER1_NAME = config["layer1_name"]

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

    tune_conditions: list[TuneCondition] = []
    layers = config["layers"]

    for layer in layers:
        value_orders_log10 = TuneCondition.get_value_orders_log10(layer)
        value_orders = 10**value_orders_log10

        for idx, value_order in enumerate(value_orders):
            tune_conditions.append(
                TuneCondition(
                    layer=layer,
                    value_order=value_order,
                )
            )

    results_dict = {}
    for tune_cond in tune_conditions:
        best_params = optimize_params(
            tune_cond, data_loader, activation_manager, n_trials=config_tu["num_trials"][tune_cond.layer], seed=42
        )
        results_dict[tune_cond.to_string()] = best_params

    # Save results
    with open(config_tu["optimized_params"], "w") as f:
        json.dump({str(key): vars(value) for key, value in results_dict.items()}, f, indent=4)
