import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import random
import math
from functools import partial

from model.base.blocks import TiTokEncoder
from model.base.mask_utils import MaskUtils
from model.base.utils import init_weights, get_model_dims

from model.metrics.lpips_gram import LPIPS

from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as TF


def l1(x, y):
    return torch.abs(x - y)

    
class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        loss_g = config.tokenizer.losses
        loss_d = config.discriminator.losses

        self.perceptual_weight = loss_g.perceptual_weight
        self.gram_weight = loss_g.gram_weight
        if self.perceptual_weight > 0.0 or self.gram_weight > 0.0:
            self.perceptual_model = LPIPS().eval().requires_grad_(False)
        
        model_d = config.discriminator.model
        self.disc_weight = loss_g.disc_weight

        if self.disc_weight > 0.0:
            disc_params = get_model_dims(model_d.model_size)
            self.disc_model = TiTokEncoder(
                patch_size=model_d.patch_size,
                in_channels=3,
                out_channels=1,
                model_params=disc_params,
            ).apply(init_weights)

            self.disc_mask_utils = MaskUtils(
                patch_size=model_d.patch_size,
                head_dim=disc_params['head_dim'],
            )
            self.disc_masks_dict = None

        self.commit_weight = loss_g.commit_weight
        self.codebook_weight = loss_g.codebook_weight

        self.cr_weight = loss_d.cr_weight
        self.cr_jitter = loss_d.cr_max_jitter
        self.total_steps = config.training.main.max_steps
        self.resize = partial(TF.resize, interpolation=InterpolationMode.BICUBIC, antialias=False)


    def get_perceptual_loss(self, recon, target): # recon/target = lists of CHW frames
        B = len(target)
        device, dtype = target[0].device, target[0].dtype
        target_out = []
        recon_out = []

        percep_pixels = self.config.training.sampling.perceptual_pixels
        trg_hw = target[torch.randint(B, [1]).item()].shape[-2:]
        num_frames = min(B, int(percep_pixels/math.prod(trg_hw)))
        selected = torch.randperm(B, device=device)[:num_frames]

        for i in selected.tolist():
            rec_frame = recon[i].clamp(-1, 1)
            trg_frame = target[i]

            # random resize
            out_hw = in_hw = trg_frame.shape[-2:]
            scale = max([x/y for x, y in zip(trg_hw, in_hw)])
            if scale > 1.0: # resize if smaller than trg
                out_hw = [math.ceil(x*scale) for x in in_hw]
                trg_frame = self.resize(trg_frame, size=out_hw)
                rec_frame = self.resize(rec_frame, size=out_hw)

            # Use resized crop similar to that in video_dataset for better coverage? Test.
            h_off, w_off = [random.randrange(0, (x-y)+1) for x, y in zip(out_hw, trg_hw)]
            target_out.append(trg_frame[..., h_off:h_off+trg_hw[0], w_off:w_off+trg_hw[1]])
            recon_out.append(rec_frame[..., h_off:h_off+trg_hw[0], w_off:w_off+trg_hw[1]])

        target_out = torch.stack(target_out, dim=0) # BCHW
        recon_out = torch.stack(recon_out, dim=0)

        selected_loss = torch.stack(self.perceptual_model(recon_out, target_out), dim=0) # [2, num_frames]

        losses = selected_loss.mean(dim=-1, keepdim=True).repeat(1, B) # avg loss on unseen frames
        losses[:, selected] = selected_loss
        return losses[0], losses[1] # lpips, gram


    def consistency_augment(self, real, fake):
        rand = torch.rand([len(real), 4]).tolist()
        out_rf = []
        for x in [real, fake]:
            out = []
            for img, r in zip(x, rand):
                H, W = img.shape[-2:]
                jitter = r[0] * self.cr_jitter # 0-1, scaled to set max - eg. <0.3, ratio of H/W
                jh, jw = int(H*jitter), int(W*jitter)
                top, left = int(r[1] * jh), int(r[2] * jw)
                img = img[..., top:top+(H-jh), left:left+(W-jw)] # random crop
                
                if r[3] < 0.5:
                    img = img.flip(-1) # horizontal flip
                    
                out.append(self.resize(img, size=[H, W]))
            out_rf.append(out) # [real, fake] equally transformed
        return out_rf


    def disc_wrapper(self, real, fake, use_cached_masks=False): # L-Cat
        B = len(real)
        device = real[0].device
        swaps = torch.rand([B], device=real[0].device) < 0.5 # rand bool

        combined = []
        for s, r, f in zip(swaps.tolist(), real, fake):
            order = [f, r] if s else [r, f]
            combined.append(torch.cat(order, dim=-1)) # chW cat

        if not use_cached_masks:
            token_counts = torch.ones([B], device=device, dtype=torch.int32)
            pixel_grids = torch.tensor([im.shape[1:] for im in combined], device=device, dtype=torch.int32) # c|HW|
            self.disc_masks_dict = self.disc_mask_utils(pixel_grids, token_counts)

        logits = self.disc_model(combined, self.disc_masks_dict).view(B)
        logits = torch.where(swaps, logits, -logits) # pull low for real, high for fake?
        # logits = logits * (swaps.to(logits)*2-1) # swaps [0, 1] -> [0, 2] -> [-1, 1]
        return logits



    def forward(self, target, recon, disc_forward=False):
        if disc_forward:
            return self._forward_discriminator(target, recon)
        else:
            return self._forward_generator(target, recon)
        

    def _forward_generator(self, target, recon):
        # target and recon are now lists of CHW tensors.
        loss_dict = {}
        target = [i.contiguous() for i in target]
        recon = [i.contiguous() for i in recon]

        recon_loss = torch.stack([l1(x, y).mean() for x, y in zip(target, recon)])
        loss_dict['recon_loss'] = recon_loss

        perceptual_loss = 0.0
        gram_loss = 0.0
        if self.perceptual_weight > 0.0 or self.gram_weight > 0.0:
            perceptual_loss, gram_loss = self.get_perceptual_loss(recon, target)

            if self.perceptual_weight > 0.0:
                loss_dict['perceptual_loss'] = perceptual_loss

            if self.gram_weight > 0.0:
                loss_dict['gram_loss'] = gram_loss

        g_loss = 0.0
        if self.disc_weight > 0.0:
            self.disc_model.requires_grad_(False)
            target = [i.detach().contiguous() for i in target]
            
            logits = self.disc_wrapper(target, recon)
            g_loss = F.softplus(-logits)**2
            loss_dict['gan_loss'] = g_loss


        total_loss = (
            recon_loss
            + (self.perceptual_weight * perceptual_loss)
            + (self.gram_weight * gram_loss)
            + (self.disc_weight * g_loss)
        ).mean()
        loss_dict['total_loss'] = total_loss
        return total_loss, {'gen/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}
    
    
    
    def _forward_discriminator(self, target, recon):
        self.disc_model.requires_grad_(True)
        loss_dict = {}
        target = [i.detach().requires_grad_(True).contiguous() for i in target]
        recon = [i.detach().requires_grad_(True).contiguous() for i in recon]
        
        logits = self.disc_wrapper(target, recon, use_cached_masks=True)
        d_loss = F.softplus(logits)**2

        loss_dict['d_loss'] = d_loss
        loss_dict['logits_relative'] = logits

        # https://www.arxiv.org/pdf/2509.24935
        cr_loss = 0.0
        if self.cr_weight > 0.0:
            logits_aug = self.disc_wrapper(*self.consistency_augment(target, recon), use_cached_masks=True)
            cr_loss = (logits-logits_aug)**2
            loss_dict['grad_penalty'] = cr_loss

        total_loss = (
            d_loss
            + (self.cr_weight * cr_loss)
        ).mean()
        loss_dict['total_loss'] = total_loss
        return total_loss, {'disc/'+k:v.clone().mean().detach() for k,v in loss_dict.items()}