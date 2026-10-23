import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
from model.base.rope import RoPE

create_block_mask = torch.compile(create_block_mask, dynamic=True, mode='max-autotune-no-cudagraphs')


class MaskUtils(nn.Module):
    def __init__(self, patch_size, head_dim=64):
        super().__init__()
        self.patch_size = torch.tensor(patch_size, dtype=torch.int32)
        self.rope = RoPE(head_dim=head_dim, grid_dims=len(patch_size))
        

    def setup_flex_attn(self, seq_lens):
        L, device = seq_lens.sum().item(), seq_lens.device
        sample_ids = torch.arange(seq_lens.shape[0], device=device, dtype=torch.int32).repeat_interleave(seq_lens) 

        def mask_mod(b, h, q_idx, kv_idx):
            sample_block = sample_ids[q_idx] == sample_ids[kv_idx]
            return sample_block


        block_mask = create_block_mask(
            mask_mod=mask_mod,
            B=None, H=None,
            Q_LEN=L, KV_LEN=L,
            device=device.type,
        )
        return block_mask


    def forward(self, pixel_grids, token_counts):
        grids = pixel_grids // self.patch_size.to(pixel_grids).unsqueeze(0)
        grid_sizes = grids.prod(dim=-1)
        seq_lens = grid_sizes + token_counts

        values = torch.tensor([0, 1], device=grids.device, dtype=torch.bool).repeat(grids.shape[0])
        token_mask = values.repeat_interleave(torch.stack([grid_sizes, token_counts], dim=-1).flatten())
        block_mask = self.setup_flex_attn(seq_lens)

        out_dict = {
            'grids': grids,
            'token_counts': token_counts,
            'seq_lens': seq_lens,
            'token_mask': token_mask,
            'block_mask': block_mask,
            'freqs': self.rope(grids, token_counts, grids.device)
        }
        return out_dict


