
import tqdm
from sklearn import metrics
from datasets import load_from_disk
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision.datasets import CocoDetection
import os
from typing import Iterable, Callable, Optional, Any, Tuple, List
from omegaconf import OmegaConf
import argparse
from torchmetrics.classification import BinaryAUROC, BinaryROC


def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
            # for unknown caption
            # captions.append('None')
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
            # for unknown caption
            # captions.append('None')
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def preprocess_train(examples):
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [transform(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}



def load_pokemon_datasets():
    ds = load_from_disk('/banana/ethan/MIA_data/POKEMON/pokemon_blip_splits')
    train_dataset = ds['train'].with_transform(preprocess_train)
    test_dataset = ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader

def load_flickr_datasets():
    ds = load_from_disk('/banana/ethan/MIA_data/FLICKR/flickr30k_splits/')
    train_dataset = ds['train'].with_transform(preprocess_train)
    test_dataset = ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader

def load_coco_datasets():
    ds = load_from_disk('/banana/ethan/MIA_data/MSCOCO/coco2017_val_splits')
    train_dataset = ds['train'].with_transform(preprocess_train)
    test_dataset = ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader

def load_laion_datasets():
    train_ds = load_from_disk('/banana/ethan/MIA_data/LAION5k/laion_aesthetic_v2_5plus_2500_clean')
    test_ds = load_from_disk('/banana/ethan/MIA_data/LAION5k/laion2B_multi_ascii_v25_2500_clean')
    train_dataset = train_ds.with_transform(preprocess_train)
    test_dataset = test_ds.with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_laion2_datasets():
    train_ds = load_from_disk('/banana/ethan/MIA_data/LAION5k/laion_aesthetic_v2_5plus_2500_clean')
    test_ds = load_from_disk('/banana/ethan/MIA_data/MSCOCO/coco2017_val_splits')
    train_dataset = train_ds.with_transform(preprocess_train)
    test_dataset = test_ds['test'].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def perform_attack(pipe, dataloader, attacker, unconditional, prefix='member'):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores = []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        # Convert images to latent space
        pixel_values = batch["pixel_values"].to(weight_dtype)
        pixel_values = pixel_values.cuda()
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215
        # Get the text embedding for conditioning
        input_ids = batch["input_ids"].cuda()
        encoder_hidden_states =  None if unconditional else text_encoder(input_ids)[0]

        if attacker == 'SimA':
            score = \
                pipe.__sima_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'PIA':
            score = \
                pipe.__pia_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'SecMI':
            score = \
                pipe.__secmi_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'Loss':
            score = \
                pipe.__loss_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'PFAMI':
            score = \
                pipe.__pfami_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        elif attacker == 'Epsilon':
            score = \
                pipe.__epsilon_call__(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)
        else:
            raise NotImplementedError
        scores.append(score)
        # mean_l2 += score
        # print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

    return torch.concat(scores, dim=-1)


def main(args):
    if args.dataset == 'pokemon':
        _, _, train_loader, test_loader = load_pokemon_datasets()
    elif args.dataset == 'coco':
        _, _, train_loader, test_loader = load_coco_datasets()
    elif args.dataset == 'flickr':
        _, _, train_loader, test_loader = load_flickr_datasets()
    elif args.dataset == 'laion-aesthetic_laion-multitrans':
        _, _, train_loader, test_loader = load_laion_datasets()
    elif args.dataset == 'laion-aesthetic_coco':
        _, _, train_loader, test_loader = load_laion2_datasets()
        # _, _, train_loader, test_loader = load_laion2_datasets()
        
    else:
        raise NotImplementedError

    pipe = load_pipeline(args.ckpt_path, args.device)

    member_scores = perform_attack(pipe, train_loader, args.attacker, args.unconditional)
    if args.dataset == 'laion-aesthetic_coco':
        global caption_column
        caption_column = dataset_name_mapping['coco'][1]
    nonmember_scores = perform_attack(pipe, test_loader, args.attacker, args.unconditional)

    # ⑤ metrics
    auc_mtr, roc_mtr = BinaryAUROC().to(args.device), BinaryROC().to(args.device)
    auroc_k, tpr1_k, asr_k = [], [], []

    for k in range(member_scores.size(0)):
        m, n = member_scores[k], nonmember_scores[k]
        scale = torch.max(m.max(), n.max()); m, n = m/scale, n/scale
        scores = torch.cat([m, n])
        labels = torch.cat([torch.zeros_like(m), torch.ones_like(n)]).long()

        auroc = auc_mtr(scores, labels).item()
        fpr, tpr, _ = roc_mtr(scores, labels)
        idx = (fpr < 0.01).sum() - 1
        tpr_at1 = tpr[idx].item()
        asr = ((tpr + 1 - fpr) / 2).max().item()

        auroc_k.append(auroc); tpr1_k.append(tpr_at1); asr_k.append(asr)
        auc_mtr.reset(); roc_mtr.reset()

    print('AUROC  per-step :', auroc_k)
    print('TPR@1% per-step :', tpr1_k)
    print('ASR     per-step:', asr_k)
    print('\nBest over K steps')
    print(f'  AUROC  = {max(auroc_k):.4f}')
    print(f'  ASR    = {max(asr_k):.4f}')
    print(f'  TPR@1% = {max(tpr1_k):.4f}')




def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pokemon', choices=['pokemon', 'coco', 'flickr', 'laion-aesthetic_laion-multitrans', 'laion-aesthetic_coco'])
    # parser.add_argument('--dataset-root', default='/banana/ethan/MIA_data/SecMI-LDM_data/datasets', type=str)
    parser.add_argument('--attacker', default='sima', choices=['SimA', 'SecMI', 'PIA', 'Loss', 'PFAMI', 'Epsilon'])
    parser.add_argument('--caption_column', default='text', type=str)
    parser.add_argument('--image_column', default='image', type=str)
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help=(
            "Whether to turn on unconditional generative mode"
        ),
    )
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='/banana/ethan/MIA_data/POKEMON/logs/sd-pokemon-model')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()


    dataset_name_mapping = {
        "pokemon": ("image", "text"),
        "coco": ("image", "captions"),
        "flickr": ("image", "caption"),
        "laion-aesthetic_laion-multitrans": ("image", "text"),
        "laion-aesthetic_coco": ("image", "text")
    }

    image_column, caption_column= dataset_name_mapping[args.dataset]

    # image.save("astronaut_rides_horse.png")
    # ckpt_path = "/banana/ethan/MIA_data/POKEMON/sd-pokemon-checkpoint"
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    # args.ckpt_path = ckpt_path

    tokenizer = CLIPTokenizer.from_pretrained(
        args.ckpt_path, subfolder="tokenizer", revision=None
    )
    # tokenizer = tokenizer.cuda()

    text_encoder = CLIPTextModel.from_pretrained(
        args.ckpt_path, subfolder="text_encoder", revision=None
    )
    text_encoder = text_encoder.to(args.device)

    vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    vae = vae.to(args.device)

    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt_path, subfolder="unet", revision=None
    )
    unet = unet.to(args.device)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    fix_seed(args.seed)

    main(args)
