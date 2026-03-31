from pathlib import Path
import yaml


def load_config():
    CONFIG_PATH = Path("config_attackcnn.yaml")
    with CONFIG_PATH.open("r") as config_file:
        return yaml.safe_load(config_file)


config = load_config()
