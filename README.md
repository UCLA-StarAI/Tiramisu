# Tiramisu

This is the official implementation of "[Image Inpainting via Tractable Steering of Diffusion Models](https://arxiv.org/abs/2401.03349.pdf)". 

## Usage

Follow the readme instructions in `exps/generate_images` to generate inpainted images. The Probabilistic Circuits required by the generation process can either be downloaded from [here](https://drive.google.com/drive/folders/19JLV1FUDl440g6sm4DpW-V5KjGbN9Adv?usp=sharing) or trained following the instructions in `exps/lvd_training`.

## Structure of the datasets

CelebA:

```
CelebA
├── train256
│   ├── 11900.jpg
│   ├── 11901.jpg
│   ├── ...
├── val256
│   ├── 12291.jpg
│   ├── 12297.jpg
│   ├── ...
```

ImageNet:

```
ImageNet
├── train
│   ├── filelist.txt
│   ├── n01697457
│   │   ├── n01697457_11482.JPEG
│   │   ├── n01697457_11492.JPEG
│   │   ├── ...
│   ├── n01698640
│   │   ├── ...
│   ├── ...
├── val
│   ├── filelist.txt
│   ├── n01698640
│   │   ├── ILSVRC2012_val_00000090.JPEG
│   │   ├── ILSVRC2012_val_00001338.JPEG
│   │   ├── ...
│   ├── n01704323
│   │   ├── ...
│   ├── ...
```

LSUN-Bedroom:

```
LSUN
├── bedrooms
│   ├── train
│   │   ├── 0
│   │   │   ├── ...
│   │   ├── 1
│   │   │   ├── ...
│   ├── val
│   │   ├── 0
│   │   │   ├── ...
│   │   ├── 1
│   │   │   ├── ...
```
