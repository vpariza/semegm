import os
import random
from semegm.data.VOCdevkit.vocdata import VOCDataModule
from semegm.data.coco.coco_data_module import CocoDataModule
from semegm.transforms import SepTransforms


from semegm.data.cityscapes.cityscapes_data import CityscapesDataModule
from semegm.data.ade20k.ade20kdata import Ade20kDataModule
from semegm.data.vaihingen.vaihingen_data import Vaihingen2DDataModule
from semegm.data.VOCdevkit.vocdata import VOCDataModule
from semegm.data.coco.coco_data_module import CocoDataModule

def get_data_module(train_transforms, val_image_transforms, val_target_transforms, dataset_name, data_dir, batch_size, num_workers):
    """
    Retrieve the Data Module
    """
    if dataset_name == "voc":
        data_module = VOCDataModule(batch_size=batch_size,
                                    return_masks=True,
                                    num_workers=num_workers,
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    drop_last=True,
                                    val_image_transform=val_image_transforms,
                                    val_target_transform=val_target_transforms)
        data_module.num_classes = 21
        data_module.ignore_index = 255
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["thing", "stuff"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index = 255
        file_list = os.listdir(os.path.join(data_dir, "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "images", "val2017"))
        random.shuffle(file_list_val)
        # sample 10% of train images
        random.shuffle(file_list)
        file_list = file_list[:int(len(file_list)*0.1)]
        print(f"sampled {len(file_list)} COCO images for training")

        data_module = CocoDataModule(batch_size=batch_size,
                                     num_workers=num_workers,
                                     file_list=file_list,
                                     data_dir=data_dir,
                                     file_list_val=file_list_val,
                                     mask_type=mask_type,
                                     train_transforms=train_transforms,
                                     val_transforms=val_image_transforms,
                                     val_target_transforms=val_target_transforms)
        data_module.num_classes = num_classes
        data_module.ignore_index = ignore_index
    elif dataset_name == "ade20k":
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = Ade20kDataModule(data_dir,
                                        train_transforms=train_transforms,
                                        val_transforms=val_transforms,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        batch_size=batch_size)
        data_module.num_classes = 151
        data_module.ignore_index = 0
    elif dataset_name == "cityscapes":
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = CityscapesDataModule(root=data_dir,
                                           train_transforms=train_transforms,
                                           val_transforms=val_transforms,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           batch_size=batch_size)
        data_module.num_classes = 19
        data_module.ignore_index = 255
    elif dataset_name == "vaihingen":
        val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = Vaihingen2DDataModule(root=data_dir,
                                           train_transforms=train_transforms,
                                           val_transforms=val_transforms,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           batch_size=batch_size)
        data_module.num_classes = 6
        data_module.ignore_index = 0
    else:
        raise ValueError(f"{dataset_name} not supported")
    

    return data_module