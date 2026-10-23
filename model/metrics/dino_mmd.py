import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
import requests
from tqdm import tqdm

from sklearn import metrics
from torchvision.transforms import v2, functional

from transformers import AutoModel


# def make_transform(resize_size: int = 256):
#     to_tensor = v2.ToImage()
#     resize = v2.Resize((resize_size, resize_size), antialias=True)
#     to_float = v2.ToDtype(torch.float32, scale=True) # 0-1
#     normalize = v2.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     )
#     return v2.Compose([to_tensor, resize, to_float, normalize])


class DINOMetric(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base'): # facebook/dinov3-vitb16-pretrain-lvd1689m
        super().__init__()
        
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="cpu", # cuda?
        ).eval()

        for p in self.parameters():
            p.requires_grad = False

        self.resize = v2.Resize((224, 224), antialias=True) # 256, 256 for dino v3
        self.normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.metric_name = 'dino_mmd'
        self.reset()
            
    def reset(self):
        self.recon_feats = []
        self.target_feats = []
    
    def update(self, recon, target): # BCTHW range -1, 1 in
        with torch.no_grad():
            recon_feat = self.get_feats(recon)
            target_feat = self.get_feats(target)

        self.recon_feats.append(recon_feat)
        self.target_feats.append(target_feat)

    def get_feats(self, images): # list of tensors
        batched_im = []
        for image in images:
            image = (image.unsqueeze(0).clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1] | CHW -> BCHW

            image = self.resize(image)
            image = self.normalize(image)

            batched_im.append(image)

        batched_im = torch.cat(batched_im, dim=0) # BCHW
        outputs = self.model(batched_im)
        feats = outputs.pooler_output
        # feats = outputs.last_hidden_state.mean(dim=1) # test acc vs pool

        return feats

    def gather(self):
        recon_feats = torch.cat(self.recon_feats, dim=0).cpu().float().numpy()
        target_feats = torch.cat(self.target_feats, dim=0).cpu().float().numpy()
        stats = mmd_poly(target_feats, recon_feats, degree=2, coef0=0)*100

        return stats
            
    def forward(self):
        pass

    
"""
FROM: https://github.com/oooolga/JEDi/blob/main/videojedi/mmd_polynomial.py
"""
def mmd_poly(X, Y, degree=2, gamma=None, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()