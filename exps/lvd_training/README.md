# Train a PC with LVD

Follow the below steps to train a PC with LVD. We currently support three datasets: CelebA-HQ, ImageNet, and LSUN-Bedroom. For other datasets, please replace the checkpoints accordingly.

## Step #1: extract latent embeddings from pretrained models

Download the approprate pretrained models and place them under `models/`.

CelebA-HQ: download from https://k00.fr/2xkmielf

ImageNet: download from https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/

LSUN-Bedroom: download from https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip

After downloading the models, modify the link to the dataset in the configuration file. For completeness, we also provide the respective config files in `configs/models/`. They can be used to replace the config files in the above checkpoints.

```
python extract_latents.py --config "models/2021-04-23T18-11-19_celebahq_transformer/"
```

## Step #2: train the teacher model

Use the same `--config` as in the previous step.

For `--latent-stage-config`, use the following:

CelebA-HQ: configs/latent_stage_models/mlm_medium_256_1024.yaml

ImageNet: configs/latent_stage_models/mlm_medium_256_16384.yaml

LSUN-Bedroom: configs/latent_stage_models/mlm_medium_256_16384.yaml

```
python train_lvd_teacher.py --config "models/2021-04-23T18-11-19_celebahq_transformer/" --latent-stage-config "configs/latent_stage_models/mlm_medium_256_1024.yaml"
```

## Step #3: train the student PC

Use teh same `--config` and `--latent-stage-config` as above.

For `--lvd-config`, use the following:

CelebA-HQ: configs/lvd_structures/img_pd_1024_no_tie.yaml

ImageNet: configs/lvd_structures/img_pd_1024_no_tie_4.yaml

LSUN-Bedroom: configs/lvd_structures/img_pd_1024_no_tie_4.yaml

```
python train_student_pc.py --gpus 0,1,2,3 --max-kmeans-num-samples 40000 --num-samples-per-partition 20000 --config "models/2021-04-23T18-11-19_celebahq_transformer/" --latent-stage-config "configs/mlm_medium_256_16384.yaml" --lvd-config "configs/lvd_structures/img_pd_1024_no_tie.yaml"
```

Then finetune the PC using the same configuration files:

```
python finetune_pc.py --config "models/2021-04-23T18-11-19_celebahq_transformer/" --latent-stage-config "configs/mlm_medium_256_16384.yaml" --lvd-config "configs/lvd_structures/img_pd_1024_no_tie.yaml"
```

The finetuned PC will be stored at "outputs/lvd/{full_model_name}/{lvd_name}/lvd_finetuned_pc.jpc"
