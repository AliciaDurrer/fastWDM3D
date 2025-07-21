# fastWDM3D: Fast and Accurate 3D Healthy Tissue Inpainting

This repository provides the code to the paper ["fastWDM3D: Fast and Accurate 3D Healthy Tissue Inpainting"](https://arxiv.org/abs/2507.13146) 🧠🖌️.

**Paper Abstract**: Healthy tissue inpainting has significant applications, including the generation of pseudo-healthy baselines for tumor growth models and the facilitation of image registration. In previous editions of the BraTS Local Synthesis of Healthy Brain Tissue via Inpainting Challenge, denoising diffusion probabilistic models (DDPMs) demonstrated qualitatively convincing results but suffered from low sampling speed. To mitigate this limitation, we adapted a 2D image generation approach, combining DDPMs with generative adversarial networks (GANs) and employing a variance-preserving noise schedule, for the task of 3D inpainting. Our experiments showed that the variance-preserving noise schedule and the selected reconstruction losses can be effectively utilized for high-quality 3D inpainting in a few time steps without requiring adversarial training. We applied our findings to a different architecture, a 3D wavelet diffusion model (WDM3D) that does not include a GAN component. The resulting model, denoted as fastWDM3D, obtained a SSIM of 0.8571, a MSE of 0.0079, and a PSNR of 22.26 on the BraTS inpainting test set. Remarkably, it achieved these scores using only two time steps, completing the 3D inpainting process in 1.81 s per image. When compared to other DDPMs used for healthy brain tissue inpainting, our model is up to 800 x faster while still achieving superior performance metrics. Our proposed method, fastWDM3D, represents a promising approach for fast and accurate healthy tissue inpainting.

***

If you use our work, please **cite** our paper (and consider to star this repository 😃):
```bibtex
@article{durrer2025fastwdm3d,
  title={fastWDM3D: Fast and Accurate 3D Healthy Tissue Inpainting},
  author={Durrer, Alicia and Bieder, Florentin and Friedrich, Paul and Menze, Bjoern and Cattin, Philippe C and Kofler, Florian},
  journal={arXiv preprint arXiv:2507.13146},
  year={2025}
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
