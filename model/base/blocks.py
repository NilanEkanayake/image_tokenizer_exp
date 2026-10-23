import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base.transformer import ResidualAttentionBlock, RMSNorm
from model.base.utils import patch_rearrange, unpatch_rearrange
from einops.layers.torch import Rearrange
from einops import rearrange
import math
from typing import List, Optional
        
    
class TiTokEncoder(nn.Module):
    def __init__(
            self,
            patch_size=(8, 8),
            in_channels=3,
            out_channels=5,
            model_params={
                'width': 256,
                'layers': 4,
                'heads': [4, 2],
                'head_dim': 64,
                'mlp_ratio': 4.0,
            },
        ):
        super().__init__()
        self.patch_size = torch.tensor(patch_size, dtype=torch.int32)
        self.token_size = out_channels
        self.in_channels = in_channels

        self.width = model_params['width']
        scale = self.width ** -0.5

        self.patching = patch_rearrange(patch_size)
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)
        self.mask_token = nn.Parameter(scale * torch.randn(1, self.width))
        self.ln_pre = RMSNorm(self.width)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=model_params['heads'],
            mlp_ratio=model_params['mlp_ratio'],
            num_layer=model_params['layers'],
        )

        self.ln_post = RMSNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)


    def forward(self, images, masks_dict): # X = list of CTHW
        device = images[0].device
        dtype = images[0].dtype
        mask = masks_dict['token_mask']

        patches = [self.patching(im) for im in images]
        patches = torch.cat(patches, dim=0) # LC tensor

        x = torch.zeros([mask.shape[0], self.width], device=device, dtype=dtype)
        x[mask] = self.mask_token.to(dtype)
        x[~mask] = self.proj_in(patches)
        x = self.ln_pre(x)

        x = self.model_layers(x, masks_dict)

        tokens = x[mask].contiguous() # get tokens
        tokens = self.ln_post(tokens)
        tokens = self.proj_out(tokens)
        return tokens
    


class TiTokDecoder(nn.Module):
    def __init__(
            self,
            patch_size=(8, 8),
            in_channels=5,
            out_channels=3,
            model_params={
                'width': 256,
                'layers': 4,
                'heads': [4, 2],
                'head_dim': 64,
                'mlp_ratio': 4.0,
            },
        ):
        super().__init__()
        self.patch_size = torch.tensor(patch_size, dtype=torch.int32)
        self.token_size = in_channels
        self.out_channels = out_channels

        self.width = model_params['width']
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, self.width))
        self.ln_pre = RMSNorm(self.width)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=model_params['heads'],
            mlp_ratio=model_params['mlp_ratio'],
            num_layer=model_params['layers'],
        )

        self.ln_post = RMSNorm(self.width)
        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))
        self.unpatching = unpatch_rearrange(patch_size)


    def forward(self, tokens, masks_dict):
        device = tokens.device
        dtype = tokens.dtype
        mask = masks_dict['token_mask']
        grids = masks_dict['grids']

        x = torch.zeros([mask.shape[0], self.width], device=device, dtype=dtype)
        x[mask] = self.proj_in(tokens)
        x[~mask] = self.mask_token.to(dtype)
        x = self.ln_pre(x)

        x = self.model_layers(x, masks_dict)

        patches = x[~mask].contiguous() # get patches
        patches = self.ln_post(patches)
        patches = self.proj_out(patches)
        patches = torch.split(patches, grids.prod(dim=-1).tolist(), dim=0)

        images = [self.unpatching(p, g) for p, g in zip(patches, grids)]
        return images # list of image tensors in (CHW) out
