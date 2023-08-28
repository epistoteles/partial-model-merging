# This script adds the is_buffer flag to old .safetensors checkpoints.
# If you are using the most current version of this repo you will never need it.

import torch
import os
from src.utils import get_all_model_names, _get_checkpoints_dir, model_like
from safetensors.torch import save_file, load_file


############
# fix VGGs #
############

# model_names = [name for name in get_all_model_names() if "VGG" in name]
#
# for filename in model_names:
#     filename += ".safetensors"
#     checkpoints_dir = _get_checkpoints_dir()
#     filename = os.path.join(checkpoints_dir, filename)
#
#     state_dict = load_file(filename)
#
#     if not any(["is_buffer" in k for k in state_dict.keys()]):
#         for k in [k for k in state_dict.keys() if "features" in k and "bias" in k]:
#             state_dict[k.replace("bias", "is_buffer")] = torch.zeros_like(state_dict[k]).bool()
#
#     state_dict = dict(sorted(state_dict.items()))
#
#     save_file(state_dict, filename)

######################
# fix overfixed VGGs #
######################

model_names = [name for name in get_all_model_names() if "VGG" in name and "bn" in name]

for filename in model_names:
    dummy_model = model_like(filename)
    filename += ".safetensors"
    checkpoints_dir = _get_checkpoints_dir()
    filename = os.path.join(checkpoints_dir, filename)

    sd = load_file(filename)
    sd_dummy = dummy_model.state_dict()

    for key in sd.keys():
        if key not in sd_dummy.keys():
            del sd[key]

    sd = dict(sorted(sd.items()))

    save_file(sd, filename)

###############
# fix ResNets #  TODO
###############

# model_names = [name for name in get_all_model_names() if 'VGG' in name]
#
# for filename in model_names:
#     if not filename.endswith(".safetensors"):
#         filename += ".safetensors"
#     checkpoints_dir = _get_checkpoints_dir()
#     if checkpoints_dir not in filename:
#         filename = os.path.join(checkpoints_dir, filename)
#
#     state_dict = load_file(filename)
#
#     if not any(['is_buffer' in k for k in state_dict.keys()]):
#         for k in [k for k in state_dict.keys() if 'features' in k and 'bias' in k]:
#             state_dict[k.replace('bias', 'is_buffer')] = torch.zeros_like(state_dict[k]).bool()
#
#     save_file(state_dict, filename)
