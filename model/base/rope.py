import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from einops import einsum, rearrange, reduce

"""
References:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_lumina2.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""


@torch.compiler.disable() # remove?
def apply_rotary_emb(x, freqs_cis):
    with torch.autocast(x.device.type, enabled=False):
        x_rot = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape) # unsqueeze B/H dims -> [B=1, H=1, L, D]
        x_rot[..., :freqs_cis.shape[-1]] = x_rot[..., :freqs_cis.shape[-1]] * freqs_cis.to(x_rot) # exclude pad dims
        x_out = torch.view_as_real(x_rot).flatten(-2) # 1HLD?

    return x_out.type_as(x)


# # @torch.compiler.disable()
# def apply_rotary_emb(x, freqs_cis):
#     # with torch.autocast(x.device.type, enabled=False):
#     cos, sin = [f.view(1, 1, *f.shape) for f in freqs_cis]
#     x_partial = x[..., :cos.shape[-1]].to(cos)

#     x_real, x_imag = x_partial.reshape(*x_partial.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
#     x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
#     x_out = (x_partial * cos) + (x_rotated * sin)
#     return torch.cat([x_out.to(x), x[..., x_out.shape[-1]:]], dim=-1)


class RoPE(nn.Module):
    def __init__(
            self,
            theta=10000.0,
            head_dim=64,
            grid_dims=2, # HW | THW / 3 for video
            interleave=True,
        ):
        super(RoPE, self).__init__()
        self.interleave = interleave
        self.grid_dims = grid_dims # + 1 # +1 for separate 1D dim
        grid_dim = head_dim // (self.grid_dims * 2) # same for each dim. Excess head_dim is excluded from rotation application.

        self.inv_freqs = torch.pow(
            theta,
            torch.linspace(0.0, 1.0, grid_dim, dtype=torch.float64) # dtype=torch.float64
        ) * torch.pi / 2.0
    

    def _get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        if self.interleave:
            freqs = self.inv_freqs.to(ids.device).view(1, -1, 1) * ids.to(self.inv_freqs.dtype).unsqueeze(-2) # [1, F, 1] * [L, 1, 2] -> [L, F, 2]
        else:
            freqs = self.inv_freqs.to(ids.device).view(1, 1, -1) * ids.to(self.inv_freqs.dtype).unsqueeze(-1) # [1, 1, F] * [L, 2, 1] -> [L, 2, F]
        freqs = freqs.reshape(ids.shape[0], -1) # flatten: [L, F, 2] -> [L, F*2]
        return torch.polar(torch.tensor([1]).to(freqs), freqs)
        
        # freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)  # [S, D]
        # freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)  # [S, D]
        # return freqs_cos.float(), freqs_sin.float()
    

    def forward(self, grids, token_counts, device): # MRoPE
        with torch.autocast(device.type, enabled=False):
            combined_ids = []
            for grid, token_count in zip(grids.unbind(0), token_counts.unbind(0)):
                token_ids = torch.arange(token_count, dtype=torch.float32, device=device).unsqueeze(-1).expand(-1, len(grid))

                coords = [torch.arange(g, dtype=torch.float32, device=device) for g in grid]
                grid_ids = torch.cartesian_prod(*coords)

                combined_ids.extend([grid_ids, token_ids+grid_ids.max()+1])

            combined_ids = torch.cat(combined_ids, dim=0) # cat along L
            return self._get_freqs_cis(combined_ids)