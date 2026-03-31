import pickle
from typing import Callable
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .. import config


class AlphaFunction:
    func: Callable[[float], float]
    domain: tuple[float, float]  # (min, max)

    def __init__(self, layer=None, domain=None, func=None, use_path=True):
        if use_path:
            if layer is None:
                raise ValueError("layer must be provided when use_path is True.")
            filepath = Path(config["alpha_func"]["func_path"]) / f"alpha_func_of_{layer}.pkl"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.func = data["function"]
                self.domain = data["domain"]
        else:
            if func is None:
                raise ValueError("func must be provided when use_path is False.")
            self.func = func
            if not isinstance(domain, tuple) or len(domain) != 2 or not all(isinstance(x, float) for x in domain):
                raise ValueError("domain must be a tuple of two floats.")
            self.domain = domain

    def map(self, values):
        min_val, max_val = self.domain
        if any(v < min_val or v > max_val for v in values):
            raise ValueError(f"Some values {values} are out of the valid range: {self.domain}")
        outputs = list(map(self.func, values))
        return [out if out >= 0 else 1e-15 for out in outputs]

    def plot(self, layer):
        min_val, max_val = self.domain
        x = np.linspace(min_val, max_val, 1000)
        y = self.map(x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, color="red", linewidth=2.5, label=f"Alpha Function of {layer}")
        plt.xlabel("Value", fontsize=14)
        plt.ylabel("Alpha", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid()

        path = Path(config["alpha_func"]["func_fig"]) / f"{layer}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()

    def save(self, layer):
        filepath = Path(config["alpha_func"]["func_path"]) / f"alpha_func_of_{layer}.pkl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"function": self.func, "domain": self.domain}, f)

    @classmethod
    def create_function(cls, value_orders: list[float], alphas: list[float]):
        domain = (value_orders[0], value_orders[-1])
        func = interp1d(value_orders, alphas, kind="cubic", bounds_error=False, fill_value="extrapolate")  # type: ignore
        return cls(domain=domain, func=func, use_path=False)
