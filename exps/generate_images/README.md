# Generate images with Tiramisu

## Step #1: download diffusion model checkpoints

CelebA-HQ: https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX

ImageNet: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt and https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt

LSUN-Bedroom: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt

We also need the pretrained VQ models:

CelebA-HQ: download from https://k00.fr/2xkmielf

ImageNet: download from https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/ (rename the folder as vq-f16-imagenet or adjust the path in the config file `configs/imagenet.yaml`)

LSUN-Bedroom: download from https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip

## Step #2: train the PCs

Follow the instructions in `exps/lvd_training` to train the PCs.

Alternatively, download pretrained PCs from https://drive.google.com/drive/folders/19JLV1FUDl440g6sm4DpW-V5KjGbN9Adv?usp=sharing.

## Step #3: adjust the dataset paths in the config files

The config files are in the folder `configs/`.

## Step #4: run the algorithm

```
CUDA_VISIBLE_DEVICES=0,1 python entry.py --config_file configs/celeba.yaml --outdir images/ --n_samples 1 --algorithm Tiramisu --mask_type single_vert_strip
```

Masks: top, left, expand, small_eye, single_vert_strip, single_horiz_strip

