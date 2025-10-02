# fastWDM3D: Fast and Accurate 3D Healthy Tissue Inpainting

This repository provides the code to the paper ["*fastWDM3D*: Fast and Accurate 3D Healthy Tissue Inpainting"](https://link.springer.com/chapter/10.1007/978-3-032-05472-2_17) 🧠🖌️, which was presented at the DGM4MICCAI workshop at MICCAI 2025 and published as part of the workshop proceedings.

**Paper Abstract**: Healthy tissue inpainting has many applications, for instance, generating pseudo-healthy baselines for tumor growth models or simplifying image registration. In prior editions of the *BraTS Local Synthesis of Healthy Brain Tissue via Inpainting Challenge*, denoising diffusion probabilistic models (DDPMs) demonstrated qualitatively convincing results but suffered from low sampling speed. To mitigate this limitation, we present a modified 3D wavelet diffusion model (*WDM3D*), denoted as *fastWDM3D*. Our proposed model employs a variance-preserving noise schedule and reconstruction losses over the full image as well as over the masked area only. Using *fastWDM3D* with only two time steps we achieved a SSIM of 0.8571, a MSE of 0.0079, and a PSNR of 22.26 on the *BraTS* inpainting test set. The 3D inpainting process took only 1.81 s per image. Compared to other DDPMs used for healthy brain tissue inpainting, our model is up to ~800 times faster but still achieves superior performance metrics. Our proposed method, *fastWDM3D*, represents a promising approach for fast and accurate healthy tissue inpainting.

***

If you use our work, please **cite** our paper (and consider to star this repository 😃):
```bibtex
@inproceedings{durrer2025fastwdm3d,
  title={fastWDM3D: Fast and Accurate 3D Healthy Tissue Inpainting},
  author={Durrer, Alicia and Bieder, Florentin and Friedrich, Paul and Menze, Bjoern and Cattin, Philippe C and Kofler, Florian},
  booktitle={MICCAI Workshop on Deep Generative Models},
  pages={171--181},
  year={2025},
  organization={Springer}
}
```

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

## fastWDM3D
### Training

In the file WDM3D/run.sh, set the GPU to the ID of the GPU you want to use. Set MODE='train' and DATA_MODE='train' and adapt the arguments as you wish. Particularly important:

COMMON --diffusion_steps = time steps T used in diffusion model

TRAIN --data_dir = absolute path to the training data

To start the training, run

```
bash run.sh
```

while in the folder WDM3D.

### Pretrained Model

The weights of our best performing model, the *fastWDM3D* trained with T=2 for 120000 iterations, are released on [HuggingFace](https://huggingface.co/AliciaDurrer/fastWDM3D). For more implementation details please have a look at the paper.

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

Thanks to Phung et al. and Friedrich et al. for releasing their code for [WDDGAN](https://github.com/VinAIResearch/WaveDiff/tree/main) and [WDM3D](https://github.com/pfriedri/wdm-3d), respectively.
