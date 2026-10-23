import torch
import lightning as L
from torchvision.transforms import v2, functional
import io
import numpy as np
import uuid
import webdataset as wds
from torch.utils.data import default_collate
from einops import rearrange
from torch.utils.data import DataLoader
import glob
import random
import math

from webdataset.filters import pipelinefilter
from PIL import Image


def resized_crop(image, trg_grid):
    h, w = image.shape[-2:]
    th, tw = trg_grid

    max_scale = min(h/th, w/tw)
    scale = 1.0 + torch.rand(1).item() * (max_scale - 1.0)

    ch, cw = int(scale * th), int(scale * tw)
    oh = int(torch.randint(0, h - ch + 1, [1]).item())
    ow = int(torch.randint(0, w - cw + 1, [1]).item())
    
    return v2.functional.resize(
        image[..., oh:oh+ch, ow:ow+cw],
        size=trg_grid,
        interpolation=v2.InterpolationMode.BICUBIC,
        antialias=True,
    )


def get_hw(min_grid, max_grid, patch_size): # range small enough that aspect ratio capping doesn't matter.
    H = int(min_grid[0] + torch.randint(0, (max_grid[0]-min_grid[0])//patch_size[0]+1, [1]).item() * patch_size[0])
    W = int(min_grid[1] + torch.randint(0, (max_grid[1]-min_grid[1])//patch_size[1]+1, [1]).item() * patch_size[1])
    return H, W


def custom_collate(batch): # list of dicts to dict of lists
    keys = batch[0].keys()
    return {k: [b[k] for b in batch] for k in keys}


def _image_process(data, config=None, eval=False):
    cs = config.training.sampling

    max_grid = cs.max_grid # HW
    min_grid = cs.min_grid
    max_aspect_ratio = cs.max_aspect_ratio
    min_scale = cs.min_scale

    patch_size = config.tokenizer.model.patch_size # eg. [8, 8]
    
    dtypes = {
        '16': torch.float16,
        '32': torch.float32,
        '64': torch.float64,
        'bf16': torch.bfloat16,
    }

    if config.training.main.precision == 'transformer-engine':
        out_dtype = torch.bfloat16
    else:
        out_dtype = dtypes[config.training.main.precision.split('-')[0]]

    # verify sane grid settings
    # assert all([dim % ps == 0 for dim, ps in zip(min_grid, patch_size)]) and all([dim % ps == 0 for dim, ps in zip(max_grid, patch_size)]), "dimensions in min_grid and max_grid must be evenly divisible by their respective patch size"

    for sample in data: # in = {'image': tensor, 'class_id': int}
        try:
            if sample['__key__']:
                out_key = sample['__key__']
            else:
                out_key = str(uuid.uuid4())

            image = sample['image']
            in_grid = image.size[::-1] # WH -> HW | PIL is backwards, WH

            if all([x >= y for x, y in zip(in_grid, min_grid)]):
                hw_max = [min(x, y) for x, y in zip(in_grid, max_grid)]
                image = resized_crop(v2.ToImage()(image), get_hw(min_grid, hw_max, patch_size))
                image = v2.RandomHorizontalFlip(p=0.5)(image)

                image = image.to(out_dtype) / 255
                image = (image * 2) - 1 # [-1, 1]
                yield {'image': image, '__key__': out_key} 

        except Exception as error:
            print(f'Dataset process fail: {error}')


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def _dynamic_batching(data, config, eval=False):
    cs = config.training.sampling
    patch_size = config.tokenizer.model.patch_size 
    token_range = cs.token_range
    max_grid = cs.max_grid # HW
    max_samples = config.training.eval.eval_samples
    max_seq_len = cs.eval_seq_len if eval else cs.train_seq_len

    assert math.prod(x//y for x, y in zip(max_grid, patch_size)) + token_range[1] <= max_seq_len, "max seq_len (max_grid/patch_size + token_range[1]) must be less than trg_seq_len"

    chunks = []
    token_counts = []
    curr_seq_len = 0
    seen_samples = 0

    for sample in data:
        grid_size = math.prod([x//y for x, y in zip(sample['image'].shape[1:], patch_size)]) # c|HW|
        token_count = random.randrange(token_range[0], token_range[1]+1)

        if eval:
            if seen_samples > max_samples:
                break # return?
            else:
                seen_samples += 1

        if (curr_seq_len + grid_size + token_count) > max_seq_len:
            chunks = custom_collate(chunks)
            chunks['token_counts'] = torch.tensor(token_counts, dtype=torch.int32)

            yield chunks # dynamic batch size, keeping under a target sequence length

            chunks = []
            token_counts = []
            curr_seq_len = 0

        curr_seq_len += grid_size + token_count
        chunks.append(sample)
        token_counts.append(token_count)


image_process = pipelinefilter(_image_process)
dynamic_batching = pipelinefilter(_dynamic_batching)

class WDSImageDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        cd = config.dataset

        self.config = config
        self.num_workers = cd.workers
        self.pin_memory = cd.pin_memory
        self.train_shard_path = cd.train_dataset
        self.eval_shard_path = cd.eval_dataset

    def setup(self, stage: str):
        train_pipeline = [
            wds.ResampledShards(self.train_shard_path), # no handler?
            wds.split_by_worker, # no overlapping entries between workers
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=1000, initial=200, handler=wds.warn_and_continue),
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                # class_id="cls",
                handler=wds.warn_and_continue,
            ),
            # wds.map(filter_keys(set(["image", "class_id"]))),
            wds.map(filter_keys(set(["image"]))),
            image_process(self.config, eval=False),
            dynamic_batching(self.config, eval=False),
        ]

        eval_pipeline = [
            wds.SimpleShardList(self.eval_shard_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                # class_id="cls",
                handler=wds.warn_and_continue,
            ),
            # wds.map(filter_keys(set(["image", "class_id"]))),
            wds.map(filter_keys(set(["image"]))),
            image_process(self.config, eval=True),
            dynamic_batching(self.config, eval=True),
        ]
        
        self.train_dataset = wds.DataPipeline(*train_pipeline)
        self.eval_dataset = wds.DataPipeline(*eval_pipeline)

    
    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return wds.WebLoader(self.eval_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=1, persistent_workers=True)

    def teardown(self, stage: str):
        pass