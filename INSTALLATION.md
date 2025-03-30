Installation instructions Posentangle
## Step 1: Create a python environment conda or a python virtual environment
```bash
conda create -n "neco" python=3.11 ipython
```

## Step 2: Activate the conda environment
```bash
conda activate neco
```
or 
```bash
source activate neco
```

## Step 3: Update pip
```bash
python -m pip install --upgrade pip
```

## Step 4:  Install pytorch
Please  choose the appropriate cuda drivers for the specific version: https://pytorch.org/get-started/previous-versions/#v251
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
## Step 5:  Install  Pytorch Lightning
We use pytorch lightning of `2.5.0.post`.
But you can use whatever is compatible with the specific torch version: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
```bash
pip install lightning==2.5.0.post
```

## Step 6: Install Some other needed libraries
```bash
pip install faiss-cpu==1.10.0
pip install joblib==1.4.2
pip install scipy==1.15.2
pip install matplotlib==3.10.1
```