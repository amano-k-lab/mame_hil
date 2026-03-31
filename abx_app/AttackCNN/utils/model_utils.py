from collections import OrderedDict
import yaml
from pathlib import Path

import torch
from torchvision import models

from .. import config


def load_model(device):
    model_path = Path(config["model_path"])

    model = models.resnet50(weights=None).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint["model"]

    new_state_dict = OrderedDict(
        {
            k.replace("module.model.", "").replace("module.attacker.model.", "").replace("module.", ""): v
            for k, v in model_state_dict.items()
        }
    )
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def load_default_resnet(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    return model

def load_stylized_resnet(device):
    model = models.resnet50(weights=None).to(device)
    state_dict = torch.load("data/models/stylized_resnet.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model