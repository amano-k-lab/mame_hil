from dataclasses import dataclass
import math
import json
from typing import Callable
import random
import yaml

from pathlib import Path
from tqdm import tqdm
import numpy as np

from .. import config
from .pgd_attack import AttackParams
from .pca import PCAHandler
from .ica import ICAHandler
from .decomposition_handler import DecompositionHandler
from .alpha_function import AlphaFunction


VALUE_ORDER_INTERVAL: dict[str, float] = config["tune_param"]["value_step_log10"]


@dataclass
class Condition:
    mode: str  # "pca" | "ica"
    layer: str  # "layer1" | "layer3" | "avgpool"
    component: int  # 0 | 1 | 2
    direction: str  # "plus" | "minus"
    ecc: int = 3  # "3" | "6" | "9"

    def to_string(self):
        return f"ecc{self.ecc}_{self.mode}_{self.layer}_component{self.component}_{self.direction}"

    def to_tune_condition(self, threshold: float):
        if threshold <= 0:
            raise ValueError("Threshold must be a positive value.")
        value_order = 10 ** (
            math.floor(math.log10(threshold) * (1 / VALUE_ORDER_INTERVAL[self.layer]))
            * VALUE_ORDER_INTERVAL[self.layer]
        )
        return TuneCondition(
            layer=self.layer,
            value_order=value_order,
        )

    def get_decomposition_handler(self, top10=False) -> DecompositionHandler:
        if top10 is True:
            decomposition_model_file = Path(config["ica"]["ica_models_top10"]) / f"{self.layer}.pkl"
        else:
            decomposition_model_file = Path(config["ica"]["ica_models"]) / f"{self.layer}.pkl"
        handler = None
        if self.mode == "pca":
            handler = PCAHandler(decomposition_model_file)
        elif self.mode == "ica":
            handler = ICAHandler(decomposition_model_file)
        if handler is None:
            raise ValueError("Invalid decomposition method. Use 'pca' or 'ica'.")
        return handler

    @classmethod
    def from_string(cls, cond_str):
        parts = cond_str.split("_")
        if len(parts) != 5:
            raise ValueError("Invalid cond format")
        ecc_str, mode, layer, component_str, direction = parts
        if not component_str.startswith("component"):
            raise ValueError("Invalid component format")
        component = int(component_str.replace("component", ""))
        if not ecc_str.startswith("ecc"):
            raise ValueError("Invalid ecc format")
        ecc = int(ecc_str.replace("ecc", ""))
        return cls(ecc=ecc, mode=mode, layer=layer, component=component, direction=direction)


@dataclass
class TuneCondition:
    layer: str
    value_order: float

    def __post_init__(self):
        if self.value_order < 0:
            raise ValueError("value order must be a positive value.")
        self.lower_bound = self.value_order
        self.upper_bound = 10 ** (math.log10(self.value_order) + VALUE_ORDER_INTERVAL[self.layer])

    def to_string(self):
        return f"{self.layer}_valueorder{self.value_order}"

    def gen_condition(self, num_samples, seed=42) -> list[Condition]:
        random.seed(seed)
        ecc_choices = config["ecc"]
        mode_choices = config["modes"]
        component_choices = config["components"]
        direction_choices = config["directions"]

        conditions: list[Condition] = []
        for _ in range(num_samples):
            ecc = random.choice(ecc_choices)
            mode = random.choice(mode_choices)
            component = random.choice(component_choices)
            direction = random.choice(direction_choices)
            layer = self.layer
            conditions.append(Condition(ecc=ecc, mode=mode, layer=layer, component=component, direction=direction))

        return conditions

    @classmethod
    def from_string(cls, tune_cond_str):
        parts = tune_cond_str.split("_")
        if len(parts) != 2:
            raise ValueError("Invalid cond format")
        layer, value_order_str = parts
        if not value_order_str.startswith("valueorder"):
            raise ValueError("Invalid value_order format")
        value_order = float(value_order_str.replace("valueorder", ""))
        return cls(layer=layer, value_order=value_order)

    def sample_value_from_order(self, num_samples) -> list[float]:
        return np.exp(np.linspace(np.log(self.lower_bound), np.log(self.upper_bound), num_samples))

    @staticmethod
    def get_value_orders_log10(layer, border=False):
        value_range_log10 = config["tune_param"]["value_range_log10"][layer]
        if border:
            value_range_log10 = config["tune_param"]["value_range_border_log10"][layer]
        return np.arange(value_range_log10[0], value_range_log10[-1], VALUE_ORDER_INTERVAL[layer])


class AttackParamsLoader:
    data: dict[str, AttackParams]

    def __init__(self):
        json_path = Path(config["tune_param"]["optimized_params"])
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("The JSON file must contain a distionary.")

        validated_data: dict[str, AttackParams] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key type: {key} (must be a string).")
            if not isinstance(value, dict):
                raise ValueError(f"Invalid value for key {key}: {value} (must be a dictionary).")
            try:
                validated_data[key] = AttackParams(**value)
            except TypeError as e:
                raise ValueError(f"Invalid AttackParams for key {key}: {e}")
        self.data = validated_data

    def find_closest_key(self, key: str):
        prefix, value_str = key.split("valueorder")
        target_value = float(value_str)
        closest_key = None
        closest_distance = float("inf")
        for existing_key in self.data.keys():
            if existing_key.startswith(prefix):
                existing_value = float(existing_key.split("valueorder")[1])
                log_distance = abs(math.log(existing_value) - math.log(target_value))
                if log_distance < closest_distance:
                    closest_distance = log_distance
                    closest_key = existing_key
        if closest_key is not None:
            return self.data[closest_key]
        raise KeyError(f"No matching key found for {key} in data.")

    def get_attack_params(self, cond: Condition, value: float) -> AttackParams:
        tune_cond = cond.to_tune_condition(value)
        key = tune_cond.to_string()
        params = self.find_closest_key(key)
        alpha_func = AlphaFunction(tune_cond.layer)
        alpha = alpha_func.map([value])[0]
        return AttackParams(alpha=alpha, beta=params.beta, num_iterations=params.num_iterations)

    def run_for_all_tune_conditions(self, func: Callable[[TuneCondition, AttackParams], None]):
        for key in tqdm(self.data.keys(), desc="Processing conditions"):
            tune_cond = TuneCondition.from_string(key)
            params = self.data[key]
            func(tune_cond, params)
