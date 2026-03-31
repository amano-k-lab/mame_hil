import torch
import torchvision.transforms as transforms
from pathlib import Path

from . import config
from .utils.activation_manager import ActivationManager
from .utils.data_utils import prepare_data
from .utils.model_utils import load_model, load_default_resnet, load_stylized_resnet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_SAMPLES = config["gram_matrix"]["num_samples"]
    LAYERS = config["layers"]

    # OUTPUT_DIR = Path(config["gram_matrix"]["gram_path"])
    # model = load_model(device)

    # OUTPUT_DIR = Path("data/gram_matrices_default")
    # model = load_default_resnet(device)

    OUTPUT_DIR = Path("data/gram_matrices_stylized")
    model = load_stylized_resnet(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    activation_manager = ActivationManager(model, device)
    for layer_name in LAYERS:
        data_loader, _ = prepare_data(NUM_SAMPLES[layer_name], transform, device)
        num_samples = NUM_SAMPLES[layer_name]
        gram_matrix_file = OUTPUT_DIR / f"gram_matrices_{layer_name}.npy"
        print(f"Extracting Gram Matrix for layer: {layer_name}")

        activation_manager.extract_gram_matrix_flat(data_loader, layer_name, save_path=gram_matrix_file)

        print(f"Gram Matrix for {layer_name} has been saved to {gram_matrix_file}")
