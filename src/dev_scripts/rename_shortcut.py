# This script adds the is_buffer flag to old .safetensors checkpoints that don't have it.
# If you are using the most current version of this repo you will never need it.

import torch
import os
from src.utils import get_all_model_names, _get_checkpoints_dir
from safetensors.torch import save_file, load_file
from collections import defaultdict


def _sd_item_to_key(item):
    _key = item[0].split(".")
    _key = [_convert(x) for x in _key]
    return _key


def _convert(x):
    try:
        return int(x)
    except ValueError:
        return x


#######################
# fix ResNets with bn #
#######################

model_names = [name for name in get_all_model_names() if "ResNet" in name and "bn" in name]

for filename in model_names:
    filename += ".safetensors"
    checkpoints_dir = _get_checkpoints_dir()
    filename = os.path.join(checkpoints_dir, filename)

    state_dict = load_file(filename)
    new_state_dict = {}

    for key, value in state_dict.items():
        new_state_dict[key.replace("shortcut", "downsample")] = value

    new_state_dict = dict(sorted(new_state_dict.items(), key=_sd_item_to_key))

    save_file(new_state_dict, filename)
