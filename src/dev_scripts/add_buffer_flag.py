# This script adds the is_buffer flag to old .safetensors checkpoints.
# If you are using the most current version of this repo you will never need it.

import torch
import os
from src.utils import get_all_model_names, _get_checkpoints_dir
from safetensors.torch import save_file, load_file
from collections import defaultdict


def _sd_item_to_key(item):
    _key = item[0].split(".")
    _key[1] = _convert(_key[1])
    return _key


def _convert(x):
    try:
        return int(x)
    except ValueError:
        return x


####################
# fix VGGs with bn #
####################

model_names = [name for name in get_all_model_names() if "VGG" in name and "bn" in name]

for filename in model_names:
    filename += ".safetensors"
    checkpoints_dir = _get_checkpoints_dir()
    filename = os.path.join(checkpoints_dir, filename)

    state_dict = load_file(filename)

    d = defaultdict(list)
    entries = list(map(lambda x: x.split("."), state_dict.keys()))
    for entry in entries:
        if len(entry) == 3:
            d[entry[1]].append(entry[2])

    for key, values in d.items():
        if "is_buffer" not in values:
            state_dict[f"features.{key}.is_buffer"] = torch.zeros_like(state_dict[f"features.{key}.bias"])

    state_dict = dict(sorted(state_dict.items(), key=_sd_item_to_key))

    save_file(state_dict, filename)
