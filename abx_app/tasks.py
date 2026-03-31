# from celery import shared_task
import time
start_time = time.time()
import os
from pathlib import Path
import pickle
import yaml

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


from .AttackCNN.utils.model_utils import load_model
from .AttackCNN.utils.ica import ICAHandler
from .AttackCNN.utils.condition import Condition, AttackParamsLoader
from .AttackCNN.generate_image_from_condition import generate_image_from_condition
from .AttackCNN.utils.activation_manager import ActivationManager
from .AttackCNN.utils.data_utils import prepare_data_from_pool
from .AttackCNN.utils.attack_result import get_actual_value

end_time = time.time()
print(f'Library importing takes {end_time - start_time}s')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    CONFIG_PATH = Path("config_attackcnn.yaml")
    with CONFIG_PATH.open("r") as config_file:
        return yaml.safe_load(config_file)


config = load_config()


def worker_function(*args, **kwargs):
    make_perturbed_imgs(*args, **kwargs)


# class to make an image dataset from file list
class ImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path_img = self.file_list[idx]
        file_name = os.path.basename(path_img)
        img = Image.open(path_img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, file_name


def make_perturbed_imgs(cond: Condition, threshold, path_save: Path, seed=42):

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    data_loader = prepare_data_from_pool(1, transform, device, seed=None)

    LAYER_NAME = cond.layer
    DECOMPOSITION_MODEL_FILE = Path(config["ica"]["ica_models"]) / f"{LAYER_NAME}.pkl"

    model = load_model(device)

    handler = ICAHandler(DECOMPOSITION_MODEL_FILE)

    params_loader = AttackParamsLoader()
    attack_params = params_loader.get_attack_params(cond, threshold)

    results = generate_image_from_condition(
        attack_params=attack_params,
        cond=cond,
        model=model,
        device=device,
        data_loader_origin=data_loader,
        handler=handler,
        value=threshold,
        layer1_name="conv1",
    )

    orig_img = np.clip(results[0].original_image[0].squeeze().transpose((1, 2, 0)), 0, 1)  # type: ignore
    save_image_from_numpy(orig_img, path_save, prefix="orig")
    fake_img = np.clip(
        results[0].perturbed_images[-1].squeeze().transpose((1, 2, 0)),  # type: ignore
        0,
        1,
    )  # type: ignore

    save_image_from_numpy(fake_img, path_save, prefix="fake")

    activation_manager = ActivationManager(model, device)
    actual_value = get_actual_value(results[0], activation_manager, handler, cond)
    name_cond = Condition(cond.mode, cond.layer, cond.component, cond.direction, cond.ecc).to_string()
    save_actual_value(name_cond, actual_value, path_save)


def save_actual_value(name_cond, actual_value, path_save):
    path_save_file = path_save + f"_{name_cond}.pkl"
    with open(path_save_file, "wb") as f:
        pickle.dump(actual_value, f)


def load_actual_value(name_cond, path_save):
    path_save_file = path_save + f"_{name_cond}.pkl"
    with open(path_save_file, "rb") as f:
        return pickle.load(f)


def save_image_from_numpy(img, path_save, prefix="orig"):
    # save images from a set of tensor image
    # assume image is a tensor of [1, C, H. W]
    path_save_file = path_save + f"_{prefix}.jpg"
    img = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(path_save_file)


def save_image_from_tensor(img, path_save, prefix="orig"):
    # save images from a set of tensor image
    # assume image is a tensor of [1, C, H. W]
    path_save_file = path_save + f"_{prefix}.jpg"
    img_pil = transforms.ToPILImage()(img.squeeze(0))  # to [C, H, W]
    img_pil.save(path_save_file)


def modulate_something(list_imgs, threshold, path_save):
    # example modulation as a background processing

    # dataset and dataloader
    dataset = ImageDataset(list_imgs, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # batch process (but batch_size is one considering cnn synthesis)
    for batch_idx, (images, file_name) in enumerate(dataloader):

        images = images.to(device)

        # save original images.
        save_image_from_tensor(images.cpu(), path_save, prefix="orig")

        # example function to modulate images
        ### here, image modulation
        images[:, 0, :, :] *= 1 + threshold  # to R channel
        images.clamp_(0, 1)
        ### replace to the cnn synyhesis

        save_image_from_tensor(images.cpu(), path_save, prefix="fake")


if __name__ == "__main__":
    # for debug
    list_imgs = Path("../static/abx_app/img/pool")
    threshold = 1
    cond = Condition(ecc=3, mode="pca", layer="layer1", component=1, direction="plus")
    path_save = Path("../static/abx_app/img/test/target_0")
    make_perturbed_imgs(list_imgs, cond, threshold, path_save)
