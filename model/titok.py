import torch
import torch.nn as nn
from model.base.blocks import TiTokEncoder, TiTokDecoder
from model.base.utils import init_weights, get_model_dims
from model.base.mask_utils import MaskUtils
from model.quantizer.fsq import FSQ
import math

class TiTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        titok_conf = config.tokenizer.model
        token_size = len(titok_conf.fsq_levels)

        enc_params = get_model_dims(titok_conf.encoder_size)
        dec_params = get_model_dims(titok_conf.decoder_size)
        assert enc_params['head_dim'] == dec_params['head_dim']

        self.quantize = FSQ(titok_conf.fsq_levels)

        self.encoder = TiTokEncoder(
            patch_size=titok_conf.patch_size,
            in_channels=3,
            out_channels=token_size,
            model_params=enc_params,
        )
        self.decoder = TiTokDecoder(
            patch_size=titok_conf.patch_size,
            in_channels=token_size,
            out_channels=3,
            model_params=dec_params,
        )

        self.mask_utils = MaskUtils(
            patch_size=titok_conf.patch_size,
            head_dim=enc_params['head_dim'],
        )
        self.apply(init_weights)


    def encode(self, x, masks_dict):
        x = self.encoder(x, masks_dict)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict
    
    
    def decode_indices(self, indices, masks_dict):
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype) # expects B*L in
        return self.decoder(x_q, masks_dict)
    
    
    def decode(self, x, masks_dict):
        x = self.decoder(x, masks_dict)
        return x
    
    
    def forward(self, x, token_counts): # token_counts = tensor
        pixel_grids = torch.tensor([im.shape[1:] for im in x], device=x[0].device, dtype=torch.int32) # c|HW|
        masks_dict = self.mask_utils(pixel_grids, token_counts)
        
        x_q, out_dict = self.encode(x, masks_dict)
        x = self.decode(x_q, masks_dict)
        return x, out_dict