# Score-based Membership Inference on Diffusion Models
[arXiv]() | [BibTeX](#bibtex)

<p align="center">
<img src=figures/plots_density_score.png />
</p>

This repository contains the **official implementation** of the paper *"Score-based Membership Inference on Diffusion Models?"*.  
It provides the complete codebase for reproducing all experiments associated with the proposed **Simple Attack (SimA)** method.


<p align="center">
<img src=figures/MIA_main_v_2.png />
</p>

  
## Requirements
Please refer to **README.md** of different model sub-folders.

## Experiment Details

| Model | Member | Held-out | Pre-trained | Fine-tuned | Splits | Resolution | Cond. |
|-------|--------|----------|-------------|------------|--------|------------|-------|
| [DDPM – CIFAR-10](#ddpm-cifar-10) | CIFAR-10 | CIFAR-10 | No | – | 25k/25k | 32 | – |
| [DDPM – CIFAR-100](#ddpm-cifar-100) | CIFAR-100 | CIFAR-100 | No | – | 25k/25k | 32 | – |
| [DDPM – STL10-U](#ddpm-stl10-u) | STL10-U | STL10-U | No | – | 50k/50k | 32 | – |
| [DDPM – CelebA](#ddpm-celeba) | CelebA | CelebA | No | – | 30k/30k | 32 | – |
| [Guided Diffusion – ImageNet-1k](#guided-diffusion-imagenet-1k) | ImageNet-1k | ImageNetV2 | Yes | No | 3k/3k | 256 | class |
| [Latent Diffusion Model – ImageNet-1k](#latent-diffusion-model-imagenet-1k) | ImageNet-1k | ImageNetV2 | Yes | No | 3k/3k | 256 | class |
| [Stable Diffusion V1-4 – Pokémon](#sd-v1-4-pokemon) | Pokémon | Pokémon | Yes | Yes | 416/417 | 512 | text |
| [Stable Diffusion V1-4 – COCO2017-Val](#sd-v1-4-coco2017-val) | COCO2017-Val | COCO2017-Val | Yes | Yes | 2.5k/2.5k | 512 | text |
| [Stable Diffusion V1-4 – Flickr30k](#sd-v1-4-flickr30k) | Flickr30k | Flickr30k | Yes | Yes | 10k/10k | 512 | text |
| [Stable Diffusion V1-5 – LAION-Aesthetics v2 5+ / LAION-2B-MultiTranslated](#sd-v1-5-laion-aesthetics-v2-5-laion-2b-multitranslated) | LAION-Aesthetics v2 5+ | LAION-2B-MultiTranslated | Yes | No | 2.5k/2.5k | 512 | text |
| [Stable Diffusion V1-5 – LAION-Aesthetics v2 5+ / COCO2017-Val](#sd-v1-5-laion-aesthetics-v2-5-coco2017-val) | LAION-Aesthetics v2 5+ | COCO2017-Val | Yes | No | 2.5k/2.5k | 512 | text |

---
All the experiment scripts can be found in [script.sh](https://github.com/username/repo/blob/main/scripts/run_experiment.sh). For all experiments, please choose ``--attacker`` in ``[SimA, SecMi, PIA, PFAMI, Loss]``

## DDPM <a href="https://github.com/w86763777/pytorch-ddpm" title="View GitHub source">🔗</a>
Please download all dataset splits and checkpoints [here]().
### <a id="ddpm-cifar-10"></a>CIFAR-10
#### Train
```
python /DDPM/main.py --dataset_root /path/to/dataset_root --dataset CIFAR10 --logdir /path/to/CIFAR10/logs/ --parallel
```

#### Attack
```
python DDPM/attack.py --checkpoint  /path/to/CIFAR10/cifar10_ckpt.pt --dataset_root /path/to/dataset_root --dataset CIFAR10 --img_size 32 --attacker SimA --attack_num 20 --interval 10
```


### <a id="ddpm-cifar-100"></a>CIFAR-100
#### Train
```
python DDPM/main.py --dataset_root /path/to/dataset_root --dataset CIFAR100 --logdir /path/to/CIFAR100/logs/ --parallel
```
#### Attack
```
python DDPM/attack.py --checkpoint  /path/to/CIFAR100/cifar100_ckpt.pt --dataset_root /path/to/dataset_root --dataset CIFAR100 --img_size 32 --attacker SimA --attack_num 20 --interval 10
```
  

### <a id="ddpm-stl10-u"></a>STL10-U
#### Train
```
python DDPM/main.py --dataset_root /path/to/dataset_root --dataset STL10-U --logdir /path/to/STL10-U/logs/ --batch_size 1024 --parallel
```
#### Attack
```
python DDPM/attack.py --checkpoint  /path/to/STL10-U/logs/stl10u_ckpt.pt --dataset_root /path/to/dataset_root --dataset STL10-U --img_size 32 --attacker SimA --attack_num 20 --interval 10
```

### <a id="ddpm-celeba"></a>CelebA
#### Train
```
python DDPM/main.py --dataset_root /path/to/dataset_root --dataset CELEBA --logdir /path/to/CELEBA/logs/ --batch_size 1024 --parallel
```
#### Attack
```
python DDPM/attack.py \
    --checkpoint  /path/to/CELEBA/logs/celeba_ckpt.pt \
    --dataset_root /path/to/dataset_root \
    --dataset CELEBA \
    --img_size 32 \
    --attacker SimA \
    --attack_num 20 \
    --interval 10
```
## Guided Diffusion <a href="https://github.com/openai/guided-diffusion" title="View GitHub source">🔗</a>
### <a id="guided-diffusion-imagenet-1k"></a> Member/ Held-out: [ImageNet-1K](https://www.kaggle.com/datasets/sautkin/imagenet1k0)/ [ImageNetV2](https://github.com/modestyachts/ImageNetV2)
#### Attack
```
python guided-diffusion/INv2_attack.py --IMN1k /path/to/imagenet-1k/train --IMNv2 /path/to/IMAGENETv2/ImageNetV2-matched-frequency --ckpt_path /path/to/IMAGENET1K/256x256_diffusion_uncond.pt --cond=False --attacker SimA
```
## Latent Diffusion Model <a href="https://github.com/CompVis/latent-diffusion" title="View GitHub source">🔗</a>
### <a id="latent-diffusion-model-imagenet-1k"></a> Member/ Held-out: [ImageNet-1K](https://www.kaggle.com/datasets/sautkin/imagenet1k0)/ [ImageNetV2](https://github.com/modestyachts/ImageNetV2)
#### Attack
```
python latent-diffusion/INv2_attack.py --IMN1k /path/to/imagenet-1k/train --IMNv2 /path/to/IMAGENETv2/ImageNetV2-matched-frequency  --ckpt_path /path/to/imagenet1k_ldm.ckpt --config_path latent-diffusion/configs/latent-diffusion/cin256-v2.yaml --cond=False --attacker SimA
```


## Stable Diffusion V1-4 <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" title="View HuggingFace source">🔗</a>

Please download all dataset splits [here]() and email me for the checkpoints


### <a id="sd-v1-4-pokemon"></a> Pokémon
#### Fine-tune
```
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/path/to/POKEMON/pokemon_blip_splits" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/path/to/POKEMON/logs/sd-pokemon-model" 
```
#### Attack
```
cd diffusers
python -m src.mia.attack --attacker SimA --dataset pokemon --ckpt-path /path/to/POKEMON/logs/sd-pokemon-model/
```
**--unconditional**: for unconditional attack

### <a id="sd-v1-4-coco2017-val"></a> COCO2017-Val
#### Fine-tune
```
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/path/to/MSCOCO/coco2017_val_splits" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=50000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/path/to/MSCOCO/logs/sd-mscoco-model" \
  --caption_column="captions"
```
#### Attack
```
cd diffusers
python -m src.mia.attack --attacker SimA --dataset coco --ckpt-path /path/to/MSCOCO/logs/sd-mscoco-model/
```
**--unconditional**: for unconditional attack


### <a id="sd-v1-4-flickr30k"></a>Flickr30k
#### Fine-tune
```
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/path/to/FLICKR/flickr30k_splits" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=200000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/path/to/FLICKR/logs/sd-flickr-model" \
  --caption_column="caption"
```
#### Attack
```
cd diffusers
python -m src.mia.attack --attacker SimA --dataset flickr --ckpt-path /path/to/FLICKR/logs/sd-flickr-model/
```
**--unconditional**: for unconditional attack



## Stable Diffusion V1-5 <a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5" title="View HuggingFace source">🔗</a>

### <a id="sd-v1-5-laion-aesthetics-v2-5-laion-2b-multitranslated"></a>Member/ Held-out: LAION-Aesthetics v2 5+ / LAION-2B-MultiTranslated
#### Attack
```
cd diffusers
python -m src.mia.attack --attacker SimA --dataset laion-aesthetic_coco   --ckpt-path runwayml/stable-diffusion-v1-5 
```
**--unconditional**: for unconditional attack


### <a id="sd-v1-5-laion-aesthetics-v2-5-coco2017-val"></a>Member/ Held-out: LAION-Aesthetics v2 5+ / COCO2017-Val
#### Attack
```
cd diffusers
python -m src.mia.attack --attacker SimA --dataset laion-aesthetic_laion-multitrans   --ckpt-path runwayml/stable-diffusion-v1-5
```
**--unconditional**: for unconditional attack





## BibTeX

```



```


