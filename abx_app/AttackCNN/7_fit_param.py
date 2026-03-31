import numpy as np

from . import config
from .utils.alpha_function import AlphaFunction
from .utils.condition import AttackParamsLoader, TuneCondition
from .utils.pgd_attack import AttackParams


alpha_list: dict[str, float] = {}


def get_alpha_from_tune_cond(tune_cond: TuneCondition, params: AttackParams):
    global alpha_list
    key = tune_cond.to_string()
    alpha_list[key] = params.alpha


def extract_layer_data(alpha_list: dict[str, float], layer: str):
    layer_values = []
    alphas = []
    for key, alpha in alpha_list.items():
        tune_cond = TuneCondition.from_string(key)
        if tune_cond.layer == layer:
            layer_values.append(tune_cond.value_order)
            alphas.append(alpha)

    sorted_indices = np.argsort(layer_values)
    layer_values = np.array(layer_values)[sorted_indices]
    alphas = np.array(alphas)[sorted_indices]
    return layer_values, alphas


if __name__ == "__main__":
    layers = config["layers"]

    param_loader = AttackParamsLoader()
    param_loader.run_for_all_tune_conditions(get_alpha_from_tune_cond)

    for layer in layers:
        value_orders, alphas = extract_layer_data(alpha_list, layer)
        alpha_func = AlphaFunction.create_function(value_orders, alphas)
        alpha_func.plot(layer)
        alpha_func.save(layer)
        print(f"{layer}: Function and domain saved successfully.")
