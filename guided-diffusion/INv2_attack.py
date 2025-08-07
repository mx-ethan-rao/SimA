import time, math
from pathlib import Path
import random
from collections import defaultdict
import warnings
import logging
from rich.logging import RichHandler
from rich.progress import track
from itertools import chain

import torch
from typing import Type, Dict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from absl import app, flags
import numpy as np
import components
from torchmetrics.classification import BinaryAUROC, BinaryROC

# Guided‑diffusion
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from guided_diffusion.gaussian_diffusion import ModelMeanType

# ImageNet‑v2 helper (auto‑download & correct labels)
from imagenetv2_pytorch import ImageNetV2Dataset



FLAGS = flags.FLAGS
flags.DEFINE_string('IMN1k', '/data/mingxing/imagenet-1k/data_ori/train', help='ImageNet-1k path')
flags.DEFINE_string('IMNv2', '/banana/ethan/MIA_data/IMAGENETv2/ImageNetV2-matched-frequency', help='ImageNetv2 path')
flags.DEFINE_string('ckpt_path', '/banana/ethan/MIA_data/IMAGENET1K/256x256_diffusion.pt', help='Checkpoint path')
flags.DEFINE_bool('cond', True, help='Class-Conditional')
flags.DEFINE_string('attacker', 'SimA', help='Attack Method')
flags.DEFINE_integer('interval', 10, help='Interval between attacks')
flags.DEFINE_integer('attack_num', 20, help='Number of attacks')

flags.DEFINE_integer('image_size', 256, help='image size')
flags.DEFINE_integer('batch_size', 64, help='batch size')
flags.DEFINE_integer('num_workers', 8, help='Number of workers')
flags.DEFINE_integer('seed', 2025, help='Seed')


# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        out = self.model(xt, t, condition) if condition is not None else self.model(xt, t)
        return out[:, :3]

# ----------------------------------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False



def sample_per_class(dataset, n, rng):
    buckets = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        buckets[label].append(idx)
    subset = []
    for lbl, bucket in buckets.items():
        if len(bucket) < n:
            raise RuntimeError(f'class {lbl} has only {len(bucket)} images')
        rng.shuffle(bucket)
        subset.extend(bucket[:n])
    rng.shuffle(subset)
    return subset


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "SecMI": components.SecMI,
    "PIA": components.PIA,
    "Loss": components.Loss,
    "PIAN": components.PIAN,
    "SimA": components.SimA,
    "PFAMI": components.PFAMI,
    "Epsilon": components.Epsilon,
}



def get_model(ckpt):
    cfg = model_and_diffusion_defaults()
    cfg.update(
        {
            "image_size":          256,
            "class_cond":          FLAGS.cond,      # ← one-line change
            "diffusion_steps":     1000,
            "noise_schedule":      "linear",
            "learn_sigma":         True,

            "num_channels":        256,
            "num_head_channels":   64,
            "num_res_blocks":      2,

            "attention_resolutions": "32,16,8",
            "resblock_updown":     True,
            "use_scale_shift_norm":True,

            "use_fp16":            False,           # keep FP32 for MIA
            "timestep_respacing":  "",
        }
    )

    # ------------------------------------------------------------------
    # 2)  Instantiate the UNet + Diffusion objects
    # ------------------------------------------------------------------
    model, diffusion = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().requires_grad_(False)

    # ------------- multi-GPU wrapper -------------
    if torch.cuda.device_count() > 1:
        print(f"✓ using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    return model, diffusion




@torch.no_grad()
def attack():
    seed_all(FLAGS.seed)


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    model, diffusion = get_model(FLAGS.ckpt_path)
    DEVICE = next(model.parameters()).device


    logger.info("loading dataset...")
    rng = random.Random(FLAGS.seed)
    tx = transforms.Compose([
        transforms.Resize(FLAGS.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(FLAGS.image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    imn1k = datasets.ImageFolder(str(FLAGS.IMN1k), transform=tx)

    imnv2 = datasets.ImageFolder(str(FLAGS.IMNv2), transform=tx)

    idx_imn1k = sample_per_class(imn1k, n=3, rng=rng)
    idx_imnv2 = sample_per_class(imnv2, n=3, rng=rng)

    member_loader = DataLoader(Subset(imn1k, idx_imn1k), batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
                        shuffle=False, pin_memory=True)
    # member = DataLoader(imn1k, batch_size=batch_size, num_workers=num_workers,
    #                     shuffle=False, pin_memory=True)

    held_out_loader = DataLoader(Subset(imnv2, idx_imnv2), batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
                        shuffle=False, pin_memory=True)
    
    logger.info(f"Member subset: {len(idx_imn1k)} images; Held-out: {len(idx_imnv2)} images")
    
    attacker = attackers[FLAGS.attacker](
        diffusion, FLAGS.interval, FLAGS.attack_num, EpsGetter(model), lambda x: x * 2 - 1)

    logger.info("attack start...")
    members, nonmembers = [], []
    for (member, member_label), (nonmember, nonmember_label) in track(zip(member_loader, chain(*([held_out_loader]))), total=len(held_out_loader)):
        member, nonmember = member.to(DEVICE), nonmember.to(DEVICE)
        member_label, nonmember_label = member_label.to(DEVICE), nonmember_label.to(DEVICE)

        members.append(attacker(member, condition=member_label if FLAGS.cond else None))
        nonmembers.append(attacker(nonmember, condition=nonmember_label if FLAGS.cond else None))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

    member = members[0]
    nonmember = nonmembers[0]
    metric_auc = BinaryAUROC().cuda()
    metric_roc = BinaryROC().cuda()

    auroc_k, tpr1_k, asr_k = [], [], []

    # K = member.shape[0]  (one slice per timestep t = k·interval)
    for k in range(member.shape[0]):                           
        m = member[k]                 # shape [N]  — members at timestep k
        n = nonmember[k]              # shape [N]  — non-members at timestep k

        # -------- channel-wise rescale to [0,1] --------
        scale = torch.max(m.max(), n.max())
        m, n = m / scale, n / scale

        # -------- assemble score & label tensors --------
        scores  = torch.cat([m, n]).cuda()
        labels  = torch.cat([torch.zeros_like(m),
                            torch.ones_like(n)]).long().cuda()

        # -------- AUROC --------
        auc = metric_auc(scores, labels).item()

        # -------- ROC curve --------
        fpr, tpr, _ = metric_roc(scores, labels)

        # TPR at 1 % FPR
        idx = (fpr < 0.01).sum() - 1
        tpr_at1 = tpr[idx].item()

        # ASR = best accuracy
        acc = ((tpr + 1 - fpr) / 2).max().item()

        auroc_k.append(auc)
        tpr1_k.append(tpr_at1)
        asr_k.append(acc)

        metric_auc.reset(); metric_roc.reset()          # reuse the metric objects

    # ---- summary ----
    print(f"AUROC per-timestep  : {auroc_k}")
    print(f"TPR@1%FPR per-tstep : {tpr1_k}")
    print(f"ASR   per-timestep  : {asr_k}")

    print("\nBest over all timesteps")
    print(f"  AUROC  = {max(auroc_k):.4f}")
    print(f"  ASR    = {max(asr_k):.4f}")
    print(f"  TPR@1% = {max(tpr1_k):.4f}")


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    attack()

if __name__ == '__main__':
    # fire.Fire(main)
    app.run(main)


