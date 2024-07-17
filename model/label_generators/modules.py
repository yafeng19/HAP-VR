
import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import model.label_generators.vision_transformer as vits


class DINO(nn.Module):
    
    def __init__(self, pretrained_DINO_path, DINO_arch='vit_small', 
                 patch_size=16, n_last_blocks=1, **kwargs):
        super(DINO, self).__init__()
        self.DINO_pth = pretrained_DINO_path
        self.arch = DINO_arch
        self.patch_size = patch_size
        self.n_last_blocks = n_last_blocks
        self.model = vits.__dict__[DINO_arch](patch_size=patch_size, num_classes=0)
        self.embed_dim = self.model.embed_dim * n_last_blocks
        load_pretrained_weights(self.model, self.DINO_pth, 'teacher', DINO_arch, patch_size)


    @torch.no_grad()
    def get_features(self, inp, model, n):
        inp = inp.to(torch.float16)
        inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            if "vit" in self.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            else:
                output = model(inp)
            features = output
        return features

    def forward(self, x):
        return self.get_features(x, self.model, self.n_last_blocks)


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if pretrained_weights is not None and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
