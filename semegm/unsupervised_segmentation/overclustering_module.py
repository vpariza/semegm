
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn as nn

from typing import Tuple, Union, Callable

from semegm.utils import PredsmIoUKmeans
from collections import defaultdict

class Overclustering(pl.LightningModule):

    def __init__(self,  model:Union[torch.nn.Module, pl.LightningModule], 
                 ftr_extr_fn: Callable[[Union[torch.nn.Module, pl.LightningModule], torch.Tensor], torch.Tensor],
                 patch_size: int, num_classes: int, spatial_res: int, embed_dim:int, 
                 kmeans_seeds: int, num_clusters_kmeans:int, val_iters=None, val_downsample_masks: bool = True,
                 size_masks=100, ignore_index=255):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.ftr_extr_fn = ftr_extr_fn
        self.embed_dim=embed_dim

        self.val_downsample_masks = val_downsample_masks
        self.patch_size = patch_size
        self.kmeans_seeds = kmeans_seeds
        self.spatial_res = spatial_res
        self.num_classes = num_classes
        self.size_masks = size_masks
        self.ignore_index=ignore_index
        self.val_iters=val_iters
        self.num_clusters_kmeans=num_clusters_kmeans
        self.preds_miou_layer4 = PredsmIoUKmeans(num_clusters_kmeans, num_classes)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Validate for self.val_iters. Constrained to only parts of the validation set as mIoU calculation
        # would otherwise take too long.
        if self.val_iters is None or batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, mask = batch

                # Process gt seg masks
                bs = imgs.size(0)
                assert torch.max(mask).item() <= 1 and torch.min(mask).item() >= 0
                gt = mask * 255
                if self.val_downsample_masks:
                    gt = nn.functional.interpolate(gt, size=(self.size_masks, self.size_masks), mode='nearest')
                valid = (gt != self.ignore_index)  # mask to remove object boundary class

                # Get backbone embeddings
                backbone_embeddings = self.ftr_extr_fn(self.model, imgs)

                # store embeddings, valid masks and gt for clustering after validation end
                res_w = int(np.sqrt(backbone_embeddings.size(1)))
                backbone_embeddings = backbone_embeddings.permute(0, 2, 1).reshape(bs, self.model.embed_dim,
                                                                                   res_w, res_w)
                self.preds_miou_layer4.update(valid, backbone_embeddings, gt)

    def on_validation_epoch_end(self) -> None:
        metrics = defaultdict(lambda: 0.0) 
        counts = defaultdict(lambda: 0) 
        for seed in self.kmeans_seeds:
            # Trigger computations for rank 0 process
            res_kmeans = self.preds_miou_layer4.compute(self.trainer.is_global_zero, seed=seed)
            if res_kmeans is not None:  # res_kmeans is none for all processes with rank != 0
                for k, name, res_k in res_kmeans:
                    miou_kmeans, tp, fp, fn, _, matched_bg = res_k
                    metrics[f'K={name}_miou_layer4'] += round(miou_kmeans, 8)
                    counts[f'K={name}_miou_layer4'] += 1
                    # Log precision and recall values for each class
                    for i, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
                        try:
                            class_name = self.trainer.datamodule.class_id_to_name(i)
                        except:
                            class_name = str(i)
                        score1 = round(tp_class / max(tp_class + fp_class, 1e-8), 8)
                        metrics[f'K={name}_{class_name}_precision'] += score1
                        counts[f'K={name}_{class_name}_precision'] += 1
                        score2 = round(tp_class / max(tp_class + fn_class, 1e-8), 8)
                        metrics[f'K={name}_{class_name}_recall'] += score2
                        counts[f'K={name}_{class_name}_recall'] += 1
                    if k > self.num_classes:
                        # Log percentage of clusters assigned to background class
                        metrics[f'K={name}-percentage-bg-cluster'] += round(matched_bg, 8)
                        counts[f'K={name}-percentage-bg-cluster'] += 1
        self.preds_miou_layer4.reset()
        # Average over all seeds
        for key in metrics.keys():
            metrics[key] /= counts[key]
        # Log metrics
        metrics = dict(sorted(metrics.items()))
        for key, value in metrics.items():
            self.log(key, round(value, 6))
