import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics.dino_mmd import DINOMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MetricCollection
from torchvision.transforms import v2

from einops import rearrange

class EvalMetrics(nn.Module):
    def __init__(self, config, eval_prefix='eval'):
        super().__init__()
        self.eval_prefix = eval_prefix

        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=2), # -1, 1
                "ssim": StructuralSimilarityIndexMeasure(data_range=2),
            },
            prefix=f"{eval_prefix}/",
        )

        self.dino_mmd = DINOMetric()
    
    def update(self, recon, target):
        self.dino_mmd.update(recon, target)
        for x, y in zip(recon, target):
            x = x.clamp(-1, 1)
            self.eval_metrics.update(x.unsqueeze(0), y.unsqueeze(0)) # CHW -> BCHW

    def compute(self):
        out_dict = self.eval_metrics.compute()
        out_dict[f"{self.eval_prefix}/{self.dino_mmd.metric_name}"] = self.dino_mmd.gather()
        return out_dict
    
    def reset(self):
        self.eval_metrics.reset()
        self.dino_mmd.reset()