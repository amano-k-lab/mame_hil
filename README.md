# MAME: Multidimensional Adaptive Metamer Exploration with Human Perceptual Feedback

This repository provides the analysis programs for the experimental results reported in the following paper, together with the programs used to run the experiments.

Kamao, M., Ono, H., Yamashita, A., Amano, K., & Sawayama, M. (2025). MAME: Multidimensional Adaptive Metamer Exploration with Human Perceptual Feedback. arXiv preprint arXiv:2503.13212.

Please download the model files and images used for the analysis from the following database, then place the `data` folder in the project root.

https://osf.io/3tfhu/overview?view_only=0a265341b217429599ee7bca3d2bcbef

We also provide summary files for the experimental and analysis results in the same database as `results`, so place the `results` folder in the project root as well.

https://osf.io/3tfhu/overview?view_only=0a265341b217429599ee7bca3d2bcbef


## Analysis of Experimental Results

Run `threshold.ipynb` sequentially to reproduce the result analysis.

Run `analysis.ipynb` sequentially to perform the analysis based on image metrics.

All computed data is also stored in our OSF results folder.

## Environment Setup for Running the Experiment

### 1. Python setup

- Use the Docker container published by NVIDIA: `nvcr.io/nvidia/pytorch:23.05-py3` ([reference](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-05.html#rel-23-05)).
- Environment used in this project: `python==3.10.6`, `pytorch==2.0.0`, `torchvision==0.15.1`, `NVIDIA CUDA 12.1`, `Ubuntu==22.04`

- The `-v` option of `docker run` mounts a host directory into a directory inside the Docker container. The format is:

```bash
-v host_directory:container_directory
```

- In the command below, replace the host path (`-v ~/workspace/myprj/proj_texture_hil:`) with the correct path on your machine.

```bash
$ docker run --gpus all --network=host -it -e DISPLAY=$DISPLAY -v ~/mame_hil:/home/mame_hil  -v /tmp/.X11-unix/:/tmp/.X11-unix -v /opt/tobiipdk:/opt/tobiipdk -v /opt/TobiiProEyeTrackerManager:/opt/TobiiProEyeTrackerManager --shm-size 45G --device /dev/bus/usb:/dev/bus/usb --privileged -p 8000:8000 nvcr.io/nvidia/pytorch:23.05-py3
```

- From the second launch onward, you can reuse the existing Docker container instead of running `docker run` again. Use `docker exec` for that.

- If the Docker container starts correctly, the console prompt should look like `root@docker-container-name:/path#`.

- Install the required packages with `pip` inside the Docker container.

```bash
root@docker-container-name:/path# cd /home/mame_hil
root@docker-container-name:/path# pip install -r requirements.txt
```

### 2. Prepare the models and dataset

For model and dataset preparation, refer to the [AttackCNN README](abx_app/AttackCNN/README.md).


## Running the Django Server

- When creating the Docker container, `-p` is used so that the host can access the local server running inside the container.
- Run the following command from the project root inside the Docker container.

```bash
root@docker-container-name:/path# python manage.py runserver 0.0.0.0:8000
```

- Open http://127.0.0.1:8000/ in Google Chrome to see the login page.
