# Fast and Accurate 3D Healthy Tissue Inpainting

This repository provides the code to the paper "Fast and Accurate 3D Healthy Tissue Inpainting".

## Setup

The following steps are required to setup the environment:


```
conda create --name wavediff_inp python=3.8

conda activate wavediff_inp

pip install -r requirements.txt

```

## WDDGAN3D
### Training

Set MODE=train in the file WDDGAN3D/run.sh and adapt the arguments to match your setup. Particularly important:

--num_timesteps = time steps T used in diffusion model

--datadir = absolute path to the training data

--local_rank = ID of the GPU to use

To start the training, run

```
bash run.sh
```

while in the folder WDDGAN3D.

### Sampling

Set MODE=test in the file WDDGAN3D/run.sh and adapt the arguments to match your setup. Particularly important:

--num_timesteps = time steps T used in diffusion model

--checkpoint_path = absolute path to the model weights (.pth file)

--datadir = absolute path to the training data

In the file WDDGAN3D/test_wddgan.py, change os.environ["CUDA_VISIBLE_DEVICES"] = "0" to the ID of the GPU you want to use for sampling.

To start the sampling, run

```
bash run.sh
```

while in the folder WDDGAN3D.

## GO3D
### Training

Set MODE=train in the file GO3D/run.sh and adapt the arguments to match your setup. Particularly important:

--num_timesteps = time steps T used in diffusion model

--datadir = absolute path to the training data

--local_rank = ID of the GPU to use

To start the training, run

```
bash run.sh
```

while in the folder GO3D.

### Sampling

Set MODE=test in the file GO3D/run.sh and adapt the arguments to match your setup. Particularly important:

--num_timesteps = time steps T used in diffusion model

--checkpoint_path = absolute path to the model weights (.pth file)

--datadir = absolute path to the training data

In the file GO3D/test_wddgan.py, change os.environ["CUDA_VISIBLE_DEVICES"] = "0" to the ID of the GPU you want to use for sampling.

To start the sampling, run

```
bash run.sh
```

while in the folder GO3D.

## WDM3D
### Training

In the file WDM3D/run.sh, set the GPU to the ID of the GPU you want to use. Set MODE='train' and DATA_MODE='train' and adapt the arguments as you wish. Particularly important:

COMMON --diffusion_steps = time steps T used in diffusion model

TRAIN --data_dir = absolute path to the training data

To start the training, run

```
bash run.sh
```

while in the folder WDM3D.


### Sampling

In the file WDM3D/run.sh, set the GPU to the ID of the GPU you want to use. Set MODE='sample' and DATA_MODE='test' and adapt the arguments as you wish. Particularly important:

Set the BATCH_SIZE in the model you want to use to 1.

COMMON --diffusion_steps = time steps T used in diffusion model

SAMPLE --data_dir = absolute path to the testing data

SAMPLE --model_path = absolute path to the model weights (.pt file)

To start the sampling, run

```
bash run.sh
```

while in the folder WDM3D.

## Acknowledgements

Thanks to Phung et al. (https://github.com/VinAIResearch/WaveDiff/tree/main) and Friedrich et al. (https://github.com/pfriedri/wdm-3d) for releasing their code for WDDGAN and WDM3D, respectively.
