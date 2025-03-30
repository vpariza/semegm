### Description
This repository is a reproduction repository that implements a few simple yet useful semantic segmentation evaluations for Dense Vision Encoders (vision encoders returing an embedding for every patch of an image).

Author
* Valentinos Pariza

At the **University of Technology Nuremberg (UTN)**


### Notes
* For any questions/issues etc. please open a github issue on this repository.
* If you find this repository useful, please consider starring and citing.

### Usage

#### Example on how to Evaluate dino with Overclustering Unsupervised Semantic Segmentation
* The following code uses kmeans to cluster patch embeddings clusters and then tries to compare them with the segmentation maps from the ground truth.
```python
import torch
from semegm.unsupervised_segmentation.overclustering import overclustering
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
ftr_extr_fn = lambda model, imgs: model.get_intermediate_layers(imgs)[0][:, 1:]
data_dir="<PATH TO DIRECTORY WHERE THE PASCAL VOC IS>"
dataset_name="voc"
patch_size=16
embed_dim=384 # The shape of the patch embeddings.
overclustering(model = model, ftr_extr_fn=ftr_extr_fn, patch_size=patch_size, dataset_name=dataset_name, data_dir=data_dir, embed_dim=embed_dim)
```

#### Example on how to Evaluate dino with Linear Supervised Semantic Segmentation
* The following code trains a linear layer to map patch embeddings to patch semantic labels, via supervised semantic segmentatio training.
```python
import torch
from semegm.supervised_segmentation.linear_segmentation import ls_train
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
ftr_extr_fn = lambda model, imgs: model.get_intermediate_layers(imgs)[0][:, 1:]
data_dir="<PATH TO DIRECTORY WHERE THE PASCAL VOC IS>"
dataset_name="voc"
patch_size=16
embed_dim=384 # The shape of the patch embeddings.
ls_train(model = model, ftr_extr_fn=ftr_extr_fn, patch_size=patch_size, dataset_name=dataset_name, data_dir=data_dir, embed_dim=embed_dim)
```

#### Example on how to Evaluate a head trained with Linear Supervised Semantic Segmentation with dino
* The following code trains a linear layer to map patch embeddings to patch semantic labels, via supervised semantic segmentatio training.
```python
import torch
from semegm.supervised_segmentation.linear_segmentation import ls_train
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
ftr_extr_fn = lambda model, imgs: model.get_intermediate_layers(imgs)[0][:, 1:]
data_dir="<PATH TO DIRECTORY WHERE THE PASCAL VOC IS>"
dataset_name="voc"
patch_size=16
embed_dim=384 # The shape of the patch embeddings.
path_to_head="<PATH TO CHECKPOINT WHERE THE LS HEAD IS STORED>"
ls_test(model = model, ftr_extr_fn=ftr_extr_fn, patch_size=patch_size, dataset_name=dataset_name, data_dir=data_dir, embed_dim=embed_dim, ckpt_path=path_to_head)
```

###  Setup
This is the section describing what is required to use our library.

#### Python Libraries
Please refer to the [Installation Guide](./INSTALLATION.md).

* If you want to install the library to have access everywhere when using a python environment, then you can do so:
```bash
cd semegm
pip install . # or for editing and using: `pip install -e .`
```

#### Dataset Setup
Please refer to the [Dataset Setup Guide](./DATASET.md).

## Contributors

| n  | Username |
| ------------- | ------------- |
| 1  | [@vpariza](https://github.com/vpariza)  |

## Citations
If you find this repo helpful, please consider citing these works:

The original paper that introduced the unsupervised segmentation evaluation:
```
@misc{ziegler2022selfsupervisedlearningobjectparts,
      title={Self-Supervised Learning of Object Parts for Semantic Segmentation}, 
      author={Adrian Ziegler and Yuki M. Asano},
      year={2022},
      eprint={2204.13101},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.13101}, 
}
```

Our work and repository:
```
@misc{pariza2025semegm,
      author = {Pariza, Valentinos},
      month = {3},
      title = {Simple Semantic Segmentation for Vision Encoders},
      url = {},
      year = {2025}
}
```