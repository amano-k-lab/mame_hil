from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import torch

V = TypeVar("V", np.ndarray, torch.Tensor)


class DecompositionHandler(ABC):
    @abstractmethod
    def transform_coordinate(self, data_flat: V) -> V:
        pass

    @abstractmethod
    def inverse_coordinate(self, coordinates: V) -> V:
        pass
