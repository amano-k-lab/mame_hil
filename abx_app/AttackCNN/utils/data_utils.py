import random
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from .. import config


class SingleDataset(Dataset):
    def __init__(self, item):
        if isinstance(item[0], torch.Tensor):
            self.item = (item[0].squeeze(dim=0), *item[1:])
        else:
            self.item = item

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.item


def prepare_data(num_samples, transform, device, batch_size=1, seed=51):
    val_root = config["val_paths"]["val_path"]

    random.seed(seed)
    torch.manual_seed(seed)

    val_dataset = ImageFolder(root=val_root, transform=transform)
    subset_indices = random.sample(range(len(val_dataset)), num_samples)
    val_subset = Subset(val_dataset, subset_indices)

    target_image_idx = random.randint(0, len(val_dataset) - 1)
    target_image, _ = val_dataset[target_image_idx]
    return DataLoader(val_subset, batch_size, shuffle=True), target_image.unsqueeze(0).to(device)


def prepare_data_from_pool(num_samples, transform, device, seed=None, example=False):
    if seed is not None:
        random.seed(seed)
    if example:
        imgpool_path = Path(config["example_imgpool_path"])
    else:
        imgpool_path = Path(config["imgpool_path"])
    img_files = [str(file) for file in imgpool_path.glob("*.png")]
    if not img_files:
        raise ValueError(f"No images found in directory: {imgpool_path}")

    selected_img_files = random.sample(img_files, num_samples)
    dataset = []
    for index, img_path in enumerate(selected_img_files):
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img = img.to(device)
        dataset.append((img, index))

    return DataLoader(dataset, batch_size=1, shuffle=False)  # type: ignore


def get_single_data_loader(data_loader: DataLoader, idx: int) -> DataLoader:
    transform = data_loader.collate_fn
    item = data_loader.dataset[idx]
    single_dataset = SingleDataset(item)
    return DataLoader(single_dataset, collate_fn=transform)
