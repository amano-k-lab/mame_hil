# ImageNet Adversarial Attack
## Environment Setup

### 1. Python setup

This module uses Docker, so please follow the setup instructions in the project root `README.md`.

### 2. Prepare the data

1. Register for [ImageNet](https://www.image-net.org/) and request download access.
2. Download the following files.
   - `ILSVRC2012_img_val.tar`
   - `ILSVRC2012_devkit_t12.tar.gz`

3. Create the `data/images/validation` directory and extract the files there.

```bash
tar -xvf ILSVRC2012_img_val.tar -C data/images/validation/ILSVRC2012_img_val
tar -zxvf ILSVRC2012_devkit_t12.tar.gz -C data/images/validation
```

4. Run `classification.py` to reorganize the ImageNet data.

```python
python -m abx_app.AttackCNN.0_classification
```

Reference: [Handling the ImageNet dataset with PyTorch](https://zenn.dev/hidetoshi/articles/20210717_pytorch_dataset_for_imagenet)

### Prepare the model

1. Download ImageNet L2-norm (ResNet50), `epsilon = 3.0`, from [robustness](https://github.com/MadryLab/robustness).
2. Create the `data/models/` directory and place the downloaded `imagenet_l2_3_0.pt` file there.

## Execution

- Select the process you want to run from `scripts/run_all`.
- If validation has already finished, select an option other than `0`.
- Run `scripts/run_all` from the project root.
- With the default configuration, the full run takes roughly overnight on `amano compute 01`.

```python
PYTHONUNBUFFERED=1 python -m scripts.run_all > logs/output_all.log 2>&1 &
```

### Output

- Logs are stored in the `logs` directory.
- Generated data such as gram matrices are stored in the `data` directory.
- Generated models such as ICA results, tuned parameters, and alpha functions are stored in the `model` directory.
- Images extracted from ImageNet are stored in the `images` directory.
- Graph outputs are stored in the `output` directory.
