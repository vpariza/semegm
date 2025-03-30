import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from torch import nn as nn

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Trainer
from torchvision.transforms.functional import InterpolationMode
from typing import Union, Callable

from semegm.unsupervised_segmentation.overclustering_module import Overclustering
from semegm.data import get_data_module

def overclustering(model:Union[torch.nn.Module, pl.LightningModule], ftr_extr_fn: Callable[[Union[torch.nn.Module, pl.LightningModule], torch.Tensor], torch.Tensor], 
                        patch_size:int, dataset_name:str, data_dir:str, embed_dim:int, input_size:int=448, seed:int=42, kmeans_seeds=[42], batch_size:int=32, num_workers:int=4, 
                        val_iters=None, pl_logger=None, n_devices=1, accelerator='cuda', val_downsample_masks=True, num_clusters_kmeans_miou=[-1, 300, 500], size_masks=100):
    if not num_clusters_kmeans_miou:
        raise ValueError("num_clusters_kmeans_miou must be a list/tuple of integers")
    
    seed_everything(seed)
    # Step 1: Init data and transforms
    train_transforms = None
    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor()])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])
    # Step 2: Load the Data Module
    data_module = get_data_module(train_transforms, val_image_transforms, val_target_transforms, dataset_name, data_dir, batch_size, num_workers)
    num_classes = data_module.num_classes
    ignore_index = data_module.ignore_index

    # Step 3: Replace the -1 number of clusters to the number of classes
    num_clusters_kmeans_miou = [num_classes if x == -1 else x for x in num_clusters_kmeans_miou]

    # Init method
    spatial_res = input_size / patch_size
    assert spatial_res.is_integer()
    model = Overclustering(
        model, 
        ftr_extr_fn,
        patch_size=patch_size,
        num_classes=num_classes,
        spatial_res=int(spatial_res),
        embed_dim=embed_dim,
        kmeans_seeds=kmeans_seeds,
        val_downsample_masks=val_downsample_masks,
        val_iters=val_iters,
        num_clusters_kmeans=num_clusters_kmeans_miou,
        size_masks=size_masks,
        ignore_index=ignore_index,
    )

    # Step 4: Init the trainer
    trainer = Trainer(
        logger=pl_logger,
        devices=n_devices,
        accelerator=accelerator,
        fast_dev_run=False,
        detect_anomaly=False,
    )
    # Step 5: Validate the model
    trainer.validate(model, datamodule=data_module)