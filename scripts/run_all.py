import yaml
import subprocess
from pathlib import Path
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


CONFIG_PATH = Path("config_attackcnn.yaml")
with CONFIG_PATH.open("r") as config_file:
    config = yaml.safe_load(config_file)

LAYERS = config["layers"]

# Define the modules and their corresponding log files
tasks = [
    # # ("abx_app.AttackCNN.0_classification", "logs/output_0.log"),
    # ("abx_app.AttackCNN.1_extract_gram_matrix", "logs/output_1.log"),
    # ("abx_app.AttackCNN.2_extract_ica", "logs/output_2.log"),
    # *[
    #     (
    #         [
    #             "abx_app.AttackCNN.3_image_decomposition_components",
    #             "--layer_name",
    #             layer,
    #             "--num_components",
    #             "10",
    #             "--model_type",
    #             "pca",
    #             "--model_file",
    #             str(Path(config["pca"]["pca_models_top10"]) / f"{layer}.pkl"),
    #         ],
    #         f"logs/output_3a-top10_{layer}.log",
    #     )
    #     for layer in LAYERS
    # ],
    # (
    #     [
    #         "abx_app.AttackCNN.5_extract_original_images",
    #         "--ica_model_top10",
    #         "True",
    #     ],
    #     "logs/output_5_top10.log",
    # ),
    # (
    #     [
    #         "abx_app.AttackCNN.6_tune_param",
    #         "--ica_model_top10",
    #         "True",
    #     ],
    #     "logs/output_6_top10.log",
    # ),
    # ("abx_app.AttackCNN.7_fit_param", "logs/output_7.log"),
    # ("abx_app.AttackCNN.8_evaluate_param", "logs/output_8.log"),
    # ("abx_app.AttackCNN.9_spearman_test", "logs/output_9.log"),
    ("abx_app.AttackCNN.4_select_ica_components", "logs/output_4.log"),
    # *[
    #     (
    #         [
    #             "abx_app.AttackCNN.3_image_decomposition_components",
    #             "--layer_name",
    #             layer,
    #             "--num_components",
    #             "3",
    #             "--model_type",
    #             "pca",
    #             "--model_file",
    #             str(Path(config["pca"]["pca_models"]) / f"{layer}.pkl"),
    #         ],
    #         f"logs/output_3a_{layer}.log",
    #     )
    #     for layer in LAYERS
    # ],
    # (
    #     [
    #         "abx_app.AttackCNN.5_extract_original_images",
    #         "--ica_model_top10",
    #         "False",
    #     ],
    #     "logs/output_5_ex.log",
    # ),
]

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

for task in tasks:
    if isinstance(task, list):  # Handle lists of tasks (e.g., per-layer tasks)
        for module, log_file in task:
            log_path = Path(log_file)
            print(f"[{get_timestamp()}] Running: {module}")
            with log_path.open("w") as log:
                result = subprocess.run(
                    module.split(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode != 0:
                    print(
                        f"[{get_timestamp()}] Error: {module} exited with code {result.returncode}. Check {log_file} for details."
                    )
                else:
                    print(f"[{get_timestamp()}] Completed: {module}. Logs written to {log_file}.")
    else:
        module, log_file = task
        log_path = Path(log_file)
        print(f"[{get_timestamp()}] Running: {module}")
        with log_path.open("w") as log:
            if isinstance(module, list):
                code = ["python", "-m", *module]
            else:
                code = ["python", "-m", module]
            result = subprocess.run(
                code,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                print(
                    f"[{get_timestamp()}] Error: {module} exited with code {result.returncode}. Check {log_file} for details."
                )
            else:
                print(f"[{get_timestamp()}] Completed: {module}. Logs written to {log_file}.")
