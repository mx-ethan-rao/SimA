#------------------------------------***DDPM*****-----------------------------------------------------
#-------CIFAR-10-----------------------------------------------
# Train
CUDA_VISIBLE_DEVICES=4,5,6,7 python /DDPM/main.py \
    --dataset_root /banana/ethan/MIA_data \
    --dataset CIFAR10 \
    --logdir /banana/ethan/MIA_data/CIFAR10/logs/ \
    --parallel

# Attack
python DDPM/attack.py \
    --checkpoint  /banana/ethan/MIA_data/CIFAR10/cifar10_ckpt.pt \
    --dataset_root /banana/ethan/MIA_data \
    --dataset CIFAR10 \
    --img_size 32 \
    --attacker SimA \
    --attack_num 30 \
    --interval 20

#-------CIFAR-100-----------------------------------------------
# Train
CUDA_VISIBLE_DEVICES=4,5,6,7 python DDPM/main.py \
    --dataset_root /banana/ethan/MIA_data \
    --dataset CIFAR100 \
    --logdir /banana/ethan/MIA_data/CIFAR100/logs/ \
    --parallel

# Attack
CUDA_VISIBLE_DEVICES=0 python DDPM/attack.py \
    --checkpoint  /banana/ethan/MIA_data/CIFAR100/cifar100_ckpt.pt \
    --dataset_root /banana/ethan/MIA_data \
    --dataset CIFAR100 \
    --img_size 32 \
    --attacker SimA \
    --attack_num 30 \
    --interval 20

#-------STL-10-U-----------------------------------------------
# Train
CUDA_VISIBLE_DEVICES=4,5,6,7 python DDPM/main.py \
    --dataset_root /banana/ethan/MIA_data \
    --dataset STL10-U \
    --logdir /banana/ethan/MIA_data/STL10-U/logs/ \
    --batch_size 1024 \
    --parallel


# Attack
python DDPM/attack.py \
    --checkpoint  /banana/ethan/MIA_data/STL10-U/logs/ckpt-step180000.pt \
    --dataset_root /banana/ethan/MIA_data \
    --dataset STL10-U \
    --img_size 32 \
    --attacker SimA \
    --attack_num 30 \
    --interval 20

#-------CELEBA-----------------------------------------------
# Train
CUDA_VISIBLE_DEVICES=4,5,6,7 python DDPM/main.py \
    --dataset_root /banana/ethan/MIA_data \
    --dataset CELEBA \
    --logdir /banana/ethan/MIA_data/CELEBA/logs/ \
    --batch_size 1024 \
    --parallel

# Attack
CUDA_VISIBLE_DEVICES=3 python DDPM/attack.py \
    --checkpoint  /banana/ethan/MIA_data/CELEBA/logs/ckpt-step60000.pt \
    --dataset_root /banana/ethan/MIA_data \
    --dataset CELEBA \
    --img_size 32 \
    --attacker SimA \
    --attack_num 30 \
    --interval 20

#------------------------------------***Guided Diffusion*****-----------------------------------------------------
#-------IMAGENETv2--------------------------------
CUDA_VISIBLE_DEVICES=7 python guided-diffusion/INv2_attack.py \
    --IMN1k /data/mingxing/imagenet-1k/data_ori/train \
    --IMNv2 /banana/ethan/MIA_data/IMAGENETv2/ImageNetV2-matched-frequency \
    --ckpt_path /banana/ethan/MIA_data/IMAGENET1K/256x256_diffusion_uncond.pt \
    --cond=False \
    --attacker SimA

#------------------------------------***Stable Diffusion*****-----------------------------------------------------

# Fine-tune
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/banana/ethan/MIA_data/POKEMON/pokemon_blip_splits" \
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
  --output_dir="/banana/ethan/MIA_data/POKEMON/logs/sd-pokemon-model" 


accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/banana/ethan/MIA_data/MSCOCO/coco2017_val_splits" \
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
  --output_dir="/banana/ethan/MIA_data/MSCOCO/logs/sd-mscoco-model" \
  --caption_column="captions"

accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="/banana/ethan/MIA_data/FLICKR/flickr30k_splits" \
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
  --output_dir="/banana/ethan/MIA_data/FLICKR/logs/sd-flickr-model" \
  --caption_column="caption"


# Attack
CUDA_VISIBLE_DEVICES=5 python -m src.mia.attack --attacker SimA     --dataset pokemon     --ckpt-path /banana/ethan/MIA_data/POKEMON/logs/sd-pokemon-model/
CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack --attacker SimA     --dataset pokemon     --ckpt-path /banana/ethan/MIA_data/POKEMON/logs/sd-pokemon-model/ --unconditional

CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack --attacker SimA     --dataset coco   --ckpt-path /banana/ethan/MIA_data/MSCOCO/logs/sd-mscoco-model/
CUDA_VISIBLE_DEVICES=4 python -m src.mia.attack --attacker SimA     --dataset coco   --ckpt-path /banana/ethan/MIA_data/MSCOCO/logs/sd-mscoco-model/ --unconditional

CUDA_VISIBLE_DEVICES=7 python -m src.mia.attack --attacker SimA     --dataset flickr     --ckpt-path /banana/ethan/MIA_data/FLICKR/logs/sd-flickr-model/
CUDA_VISIBLE_DEVICES=7 python -m src.mia.attack --attacker SimA     --dataset flickr     --ckpt-path /banana/ethan/MIA_data/FLICKR/logs/sd-flickr-model/ --unconditional

CUDA_VISIBLE_DEVICES=1 python -m src.mia.attack --attacker SimA     --dataset laion-aesthetic_coco   --ckpt-path runwayml/stable-diffusion-v1-5 
CUDA_VISIBLE_DEVICES=1 python -m src.mia.attack --attacker SimA     --dataset laion-aesthetic_coco   --ckpt-path runwayml/stable-diffusion-v1-5 --unconditional

CUDA_VISIBLE_DEVICES=6 python -m src.mia.attack --attacker SimA     --dataset laion-aesthetic_laion-multitrans   --ckpt-path runwayml/stable-diffusion-v1-5
CUDA_VISIBLE_DEVICES=6 python -m src.mia.attack --attacker SimA     --dataset laion-aesthetic_laion-multitrans   --ckpt-path runwayml/stable-diffusion-v1-5 --unconditional