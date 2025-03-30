import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms.functional import InterpolationMode

from typing import Tuple, Union, Callable

from semegm.supervised_segmentation.ls_module import LinearSegmentation
from semegm.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
from semegm.data import get_data_module

def ls_train(model:Union[torch.nn.Module, pl.LightningModule], ftr_extr_fn: Callable[[Union[torch.nn.Module, pl.LightningModule], torch.Tensor], torch.Tensor], 
                        patch_size:int, dataset_name:str, data_dir:str, embed_dim:int, input_size:int=448, seed:int=42, batch_size:int=32, num_workers:int=4, 
                        val_iters=None, lr:float=0.01, max_epochs:int=25, drop_at=20, decay_rate:float=0.1, checkpoint_dir='./', save_top_k=3, 
                        pl_logger=None, n_devices=1, accelerator='cuda', train_mask_size=100, val_mask_size=100, resume_ckpt_path=None) -> ModelCheckpoint:
    seed_everything(seed)
    restart = resume_ckpt_path is not None
    # Step 1: Create the Transformations needed for the training and validation datasets
    train_transforms = Compose([
        RandomResizedCrop(size=input_size, scale=(0.8, 1.)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])
    # Step 2: Load the Data Module
    data_module = get_data_module(train_transforms, val_image_transforms, val_target_transforms, dataset_name,
                                  data_dir, batch_size, num_workers)

    num_classes = data_module.num_classes
    ignore_index = data_module.ignore_index
    # Step 3: Init Method
    spatial_res = input_size / patch_size
    assert spatial_res.is_integer()
    model = LinearSegmentation(
        model=model,
        ftr_extr_fn=ftr_extr_fn,
        embed_dim=embed_dim,
        num_classes=num_classes,
        lr=lr,
        input_size=input_size,
        spatial_res=int(spatial_res),
        val_iters=val_iters,
        decay_rate=decay_rate,
        drop_at=drop_at,
        ignore_index=ignore_index,
        train_mask_size=train_mask_size,
        val_mask_size=val_mask_size
    )

    # Step 4: Init checkpoint callback storing top k heads
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='miou_val',
        filename='ckp-{epoch:02d}-{miou_val:.4f}',
        save_top_k=save_top_k,
        mode='max',
        verbose=True,
    )

    # Step 5: Init trainer and start training head
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=pl_logger,
        max_epochs=max_epochs,
        devices=n_devices,
        accelerator=accelerator, 
        fast_dev_run=False,
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        detect_anomaly=False,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt_path if restart else None)
    # Step 6: Load best checkpoint
    state_dict = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    # Step 7: Validate the model
    trainer.validate(model=model, datamodule=data_module)

    return checkpoint_callback


def ls_test(model:Union[torch.nn.Module, pl.LightningModule], 
                        ftr_extr_fn: Callable[[Union[torch.nn.Module, pl.LightningModule], torch.Tensor], torch.Tensor],
                        patch_size:int, dataset_name:str, data_dir:str, embed_dim:int, ckpt_path:str, 
                        input_size:int=448, seed:int=42, batch_size:int=32, num_workers:int=4, 
                        val_iters=None, pl_logger=None, n_devices=1, accelerator='cuda', val_mask_size=100):
    train_mask_size = val_mask_size
    seed_everything(seed)
    # Step 1: Create the Transformations needed for the training and validation datasets
    train_transforms = None # No Train Transform here

    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])
    # Step 2: Load the Data Module
    data_module = get_data_module(train_transforms, val_image_transforms, val_target_transforms, dataset_name,
                                  data_dir, batch_size, num_workers)

    num_classes = data_module.num_classes
    ignore_index = data_module.ignore_index
    # Step 3: Load the chekpoint and create the model
    spatial_res = input_size / patch_size
    assert spatial_res.is_integer()
    #  Create a 'blank' model from scratch and load the weights
    model = LinearSegmentation(
        model=model,
        ftr_extr_fn=ftr_extr_fn,
        embed_dim=embed_dim,
        num_classes=num_classes,
        lr=0.0,
        input_size=input_size,
        spatial_res=int(spatial_res),
        val_iters=val_iters,
        decay_rate=0,
        drop_at=0,
        ignore_index=ignore_index,
        train_mask_size=train_mask_size,
        val_mask_size=val_mask_size,
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)

    # Step 4: Init trainer and start validation
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=pl_logger,
        max_epochs=0,
        devices=n_devices,
        accelerator=accelerator, 
        fast_dev_run=False,
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        detect_anomaly=False,
        callbacks=None
    )
    trainer.validate(model, datamodule=data_module)


if __name__ == "__main__":
    import torch
    # Change to vit_base_patch8_224() if you want to use our larger model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    ftr_extr_fn = lambda model, imgs: model.get_intermediate_layers(imgs)[0][:, 1:]
    data_dir="/home/atuin/v115be/v115be14/data/pascal_voc_aug"
    dataset_name="voc"
    patch_size=16
    ls_train(model, ftr_extr_fn, patch_size, dataset_name, data_dir=data_dir, embed_dim=384, batch_size=128, num_workers=12, max_epochs=3, drop_at=2)
    # ls_test(model = model, ftr_extr_fn=ftr_extr_fn, patch_size=patch_size, dataset_name=dataset_name, data_dir=data_dir, embed_dim=384, ckpt_path="/home/hpc/v115be/v115be14/workspace/simple-semantic-segmentation/ckp-epoch=02-miou_val=0.4535.ckpt")