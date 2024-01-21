# Partial Model Merging

This code belongs to my Master's thesis on partial model merging, submitted on December 21st 2023.

The main functionality is concentrated in the `utils.py` and `evaluate.py` files, including docstrings.

A basic usage of the code would look like this for forced buffer assignment:

## Training

Models can be trained using `train.py`, e.g.:

```commandline
python src/train.py --dataset CIFAR10 --model ResNet --size 18 --width 1 --lr 0.4 --weight_decay 0.0001 --epochs 200 --variant a -bn -wandb -test
```

## Usage 

```python
from src.utils import *
from src.evaluate import *

# load models and dataloaders
model_a, model_b = load_models_ab("CIFAR10-VGG11-bn-1x")
train_aug_loader, train_noaug_loader, test_loader = get_loaders("CIFAR10")

# add 50% buffer neurons
model_a = expand_model(model_a, 1.5)
model_b = expand_model(model_b, 1.5)

# permute model b to fit model a
model_b_perm = permute_model(model_a, model_b, train_aug_loader)

# interpolate both models and preserve the buffer matches
model_interpolated = smart_interpolate_models(model_a, model_b_perm)

# use REPAIR (if desired)
reset_bn_stats(model_interpolated, train_aug_loader)

# use REPAIR if no batch norm layers are included in the model
# model_interpolated = repair(model_interpolated, model_a, model_b_perm, train_aug_loader)

# get final performance
get_acc_and_loss(model_interpolated, test_loader)
```

for adaptive buffer assignment the permutation must include a threshold:

```python
model_b_perm = permute_model(model_a, model_b, train_aug_loader, threshold=0.45)
```

## Setup

The ffcv python package is a bit moody when you are not using conda. Before installing the requirements.txt, please run the following commands:

```
sudo apt update
sudo apt install pkg-config libturbojpeg0-dev libopencv-dev build-essential python3-dev -y
sudo apt install cupy-cuda133 numba -y
```

