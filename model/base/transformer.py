import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention
from model.base.rope import apply_rotary_emb
from einops import rearrange
import numbers
import math


flex_attention = torch.compile(flex_attention, dynamic=True, mode='max-autotune-no-cudagraphs')


# # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_krea2.py
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(hidden_states.float(), (self.dim,), weight=self.weight.float(), eps=self.eps) # zero-centered, remove 1.0?
        return hidden_states.to(dtype)
    


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4, mult_of=32, dropout=0.0):
        super().__init__()
        inner_dim = int(mult * (2 / 3) * dim)
        inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)

        self.norm = RMSNorm(dim)
        self.w12 = nn.Linear(dim, inner_dim * 2, bias=True)
        self.drop1 = nn.Dropout(dropout)
        self.w3 = nn.Linear(inner_dim, dim, bias=True)


    def forward(self, x):
        x = self.norm(x.contiguous())
        x = self.w12(x)

        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up

        x = self.drop1(x)
        x = self.w3(x)
        return x
    


class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.q_heads, self.kv_heads = heads
        self.head_dim = dim//self.q_heads
        self.gqa_dim = self.head_dim * self.kv_heads

        self.pre_ln = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, (self.gqa_dim * 2) + dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)


    def forward(self, x, masks_dict):
        x = self.pre_ln(x.contiguous())
        q, k, v = self.to_qkv(x).split([self.dim, self.gqa_dim, self.gqa_dim], dim=-1)

        q = rearrange(q, 'l (h d) -> 1 h l d', h=self.q_heads)
        k, v = rearrange([k, v], 'n l (h d) -> n 1 h l d', h=self.kv_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, masks_dict['freqs'])
        k = apply_rotary_emb(k, masks_dict['freqs'])

        x = flex_attention(q, k, v, block_mask=masks_dict['block_mask'], enable_gqa=True)
        x = rearrange(x, '1 h l d -> l (h d)')
        return self.out_proj(x)
    


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            heads=[8, 2],
            mlp_ratio=4,
            num_layer=2,
        ):
        super(ResidualAttentionBlock, self).__init__()
        self.num_layer = num_layer
        self.attn_layer = nn.ModuleList([Attn(embed_dim, heads) for _ in range(num_layer)])
        self.ffd_layer = nn.ModuleList([SwiGLU(embed_dim, mlp_ratio) for _ in range(num_layer)])

    @torch.compile(dynamic=True, mode='max-autotune-no-cudagraphs')  
    def forward(self, x, masks_dict):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x, masks_dict)
            x = x + self.ffd_layer[i](x)
        return x