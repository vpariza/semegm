import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR
from typing import Tuple, Union, Callable

from semegm.utils import PredsmIoU

import logging
logger = logging.getLogger(__name__)

class LinearSegmentation(pl.LightningModule):
    def __init__(self, model:Union[torch.nn.Module, pl.LightningModule], 
                 ftr_extr_fn: Callable[[Union[torch.nn.Module, pl.LightningModule], torch.Tensor], torch.Tensor], 
                 num_classes: int, lr: float, input_size: int, embed_dim:int, spatial_res: int, drop_at: int, val_iters: int=None, decay_rate: float = 0.1, 
                 ignore_index: int = 255, train_mask_size=100,  val_mask_size=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.ftr_extr_fn = ftr_extr_fn
        self.embed_dim=embed_dim
        
        self.finetune_head = nn.Conv2d(self.embed_dim, num_classes, 1)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.miou_metric = PredsmIoU(num_classes, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.val_iters = val_iters
        self.input_size = input_size
        self.spatial_res = spatial_res
        self.drop_at = drop_at
        self.ignore_index = ignore_index
        self.decay_rate = decay_rate
        self.train_mask_size = train_mask_size
        self.val_mask_size = val_mask_size

    def on_after_backward(self):
        # Freeze all layers of backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.finetune_head.parameters(), weight_decay=0.0001,
                                    momentum=0.9, lr=self.lr)
        scheduler = StepLR(optimizer, gamma=self.decay_rate, step_size=self.drop_at)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        imgs, masks = batch
        bs = imgs.size(0)
        res = imgs.size(3)
        assert res == self.input_size
        try:
            self.model.eval()
        except:
            logger.warning(f"Could not successfully call model.eval() in {__name__}. Please make sure that the model is properly configured.")
            
        with torch.no_grad():
            tokens = self.ftr_extr_fn(self.model, imgs)
            tokens = tokens.reshape(bs, self.spatial_res, self.spatial_res, self.embed_dim).permute(0, 3, 1, 2)
            tokens = nn.functional.interpolate(tokens, size=(self.train_mask_size, self.train_mask_size),
                                               mode='bilinear')
        mask_preds = self.finetune_head(tokens)

        masks *= 255
        if self.train_mask_size != self.input_size:
            with torch.no_grad():
                masks = nn.functional.interpolate(masks, size=(self.train_mask_size, self.train_mask_size),
                                                  mode='nearest')

        loss = self.criterion(mask_preds, masks.long().squeeze())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if self.val_iters is None or batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, masks = batch
                bs = imgs.size(0)
                tokens = self.ftr_extr_fn(self.model, imgs)
                tokens = tokens.reshape(bs, self.spatial_res, self.spatial_res, self.embed_dim).permute(0, 3, 1, 2)
                tokens = nn.functional.interpolate(tokens, size=(self.val_mask_size, self.val_mask_size),
                                                   mode='bilinear')
                mask_preds = self.finetune_head(tokens)

                # downsample masks and preds
                gt = masks * 255
                gt = nn.functional.interpolate(gt, size=(self.val_mask_size, self.val_mask_size), mode='nearest')
                valid = (gt != self.ignore_index) # mask to remove object boundary class
                mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)

                # update metric
                self.miou_metric.update(gt[valid], mask_preds[valid])

    def on_validation_epoch_end(self):
        miou = self.miou_metric.compute(True, many_to_one=False, linear_probe=True)[0]
        self.miou_metric.reset()
        self.log('miou_val', round(miou, 6))

