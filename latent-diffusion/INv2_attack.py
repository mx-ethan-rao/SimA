"""
INv2_attack.py – Membership‑Inference benchmark adapted for
CompVis Latent‑Diffusion (LDM) checkpoints.

Example
-------
python INv2_attack.py \
  --IMN1k /path/to/ILSVRC2012/train \
  --IMNv2 /path/to/imagenetv2-matched-frequency \
  --ckpt_path  /path/to/ldm_imagenet256_cond.ckpt \
  --config_path configs/latent-diffusion/ldm-imagenet256.yaml \
  --attacker SimA --cond True
"""

# from __future__ import annotations
import logging, random, warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Type

import numpy as np
import torch
from absl import app, flags
from omegaconf import OmegaConf
from rich.logging import RichHandler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryROC

import components                                  # local utilities / attackers
from ldm.util import instantiate_from_config       # ≡ CompVis repo

# --------------------------------------------------------------------------- #
#                                  Flags                                      #
# --------------------------------------------------------------------------- #
FLAGS = flags.FLAGS
flags.DEFINE_string('IMN1k', '/data/mingxing/imagenet-1k/data_ori/train',
                    'ImageNet‑1K training root')
flags.DEFINE_string('IMNv2', '/banana/ethan/MIA_data/IMAGENETv2/ImageNetV2-matched-frequency',
                    'ImageNet‑v2 (matched‑frequency) root')
flags.DEFINE_string('ckpt_path', '/banana/ethan/MIA_data/IMAGENETv2/imagenet1k_ldm.ckpt',
                    'LDM checkpoint')
flags.DEFINE_string('config_path',
                    'configs/latent-diffusion/cin256-v2.yaml',
                    'YAML config that matches the checkpoint')
flags.DEFINE_bool  ('cond', False,
                    'Class‑conditional (False → unconditional)')
flags.DEFINE_string('attacker', 'SimA',
                    'Attack: SecMI | PFAMI | PIA | PIAN | Loss | SimA | Epsilon')
flags.DEFINE_integer('interval',    10, help='DDIM step interval')
flags.DEFINE_integer('attack_num',  20, help='# reverse steps')
flags.DEFINE_integer('image_size', 256, help='Resize / crop size')
flags.DEFINE_integer('batch_size',  64, help='Mini‑batch size')
flags.DEFINE_integer('num_workers',  8, help='# dataloader workers')
flags.DEFINE_integer('seed',      2025, help='Random seed')


# --------------------------------------------------------------------------- #
#                              Helper routines                                #
# --------------------------------------------------------------------------- #
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


@torch.no_grad()
def encode_to_latent(model, img: torch.Tensor) -> torch.Tensor:
    """x₀ ∈ [−1,1] → z₀ (latent space)."""
    posterior = model.encode_first_stage(img)
    return model.get_first_stage_encoding(posterior)


def sample_per_class(dataset: datasets.ImageFolder, n: int, rng: random.Random):
    """Return *balanced* indices (n images / class)."""
    buckets = defaultdict(list)
    for idx, (_, lbl) in enumerate(dataset.samples):
        buckets[lbl].append(idx)

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


# --------------------------------------------------------------------------- #
#                         ε‑predictor adapter (LDM)                           #
# --------------------------------------------------------------------------- #
class LDMEpsGetter(components.EpsGetter):
    """Bridges the UNet inside an LDM with the attacker API."""
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, t: int = None):
        # assert t is not None, 'timestep `t` must be supplied'
        # print(t.shape)
        # t_batch = torch.ones([xt.shape[0]], device=xt.device).long() * t
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

        # if condition is None:
        #     self.model.model.conditioning_key = None
        #     return self.model.apply_model(xt, t_batch, condition)
        # # print(self.model.apply_model(xt, t_batch, condition).shape)
        return self.model.apply_model(xt, t_batch, condition)


# --------------------------------------------------------------------------- #
#                             Model & scheduler                               #
# --------------------------------------------------------------------------- #
def load_ldm(config_path: str, ckpt_path: str):
    """Load LDM checkpoint + wrap its diffusion scheduler."""
    cfg   = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)

    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    model.eval().requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # diffusion = SchedulerWrapper(model.model)
    return model


# --------------------------------------------------------------------------- #
#                                  Attack                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attack():
    seed_all(FLAGS.seed)

    # ---------- logging ----------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info('Loading model …')
    model = load_ldm(FLAGS.config_path, FLAGS.ckpt_path)
    DEVICE = next(model.parameters()).device

    # ---------- datasets ----------
    logger.info('Loading datasets …')
    tx = transforms.Compose([
        transforms.Resize(FLAGS.image_size, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(FLAGS.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),          # → [−1,1]
    ])
    imn1k = datasets.ImageFolder(str(FLAGS.IMN1k), transform=tx)
    imnv2 = datasets.ImageFolder(str(FLAGS.IMNv2), transform=tx)

    rng = random.Random(FLAGS.seed)
    idx_imn1k = sample_per_class(imn1k, n=3, rng=rng)      # ≈ 3 k imgs
    idx_imnv2 = sample_per_class(imnv2, n=3, rng=rng)

    member_loader  = DataLoader(Subset(imn1k, idx_imn1k),
                                batch_size=FLAGS.batch_size,
                                shuffle=False, pin_memory=True,
                                num_workers=FLAGS.num_workers)
    heldout_loader = DataLoader(Subset(imnv2, idx_imnv2),
                                batch_size=FLAGS.batch_size,
                                shuffle=False, pin_memory=True,
                                num_workers=FLAGS.num_workers)

    logger.info(f'Member subset: {len(idx_imn1k)}  |  Held‑out: {len(idx_imnv2)}')

    eps_getter = LDMEpsGetter(model)
    attacker   = attackers[FLAGS.attacker](
        model, FLAGS.interval, FLAGS.attack_num, eps_getter,
        normalize=None                                    # latents already ~N(0, 1)
    )

    # ---------- main loop ----------
    logger.info('Running attack …')
    members, nonmembers = [], []
    for (m_img, m_lbl), (h_img, h_lbl) in tqdm(zip(member_loader, heldout_loader),
                                               total=len(heldout_loader)):
        m_img, h_img = m_img.to(DEVICE), h_img.to(DEVICE)
        m_lbl, h_lbl = m_lbl.to(DEVICE), h_lbl.to(DEVICE)

        # -> latent space
        m_z = encode_to_latent(model, m_img)
        h_z = encode_to_latent(model, h_img)

        if FLAGS.cond:
            cond_m = model.get_learned_conditioning({model.cond_stage_key: m_lbl})
            cond_h = model.get_learned_conditioning({model.cond_stage_key: h_lbl})
        else:
            m_lbl = torch.arange(1000).to(DEVICE)
            cond_m = model.get_learned_conditioning({model.cond_stage_key: m_lbl})
            cond_m = cond_m.mean(0).unsqueeze(0).repeat(m_img.shape[0], 1, 1)
            cond_h = cond_m

        members.append(attacker(m_z, condition=cond_m))
        nonmembers.append(attacker(h_z, condition=cond_h))
        
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


def _main(argv):                 # absl entry‑point
    warnings.filterwarnings('ignore', category=FutureWarning)
    attack()


if __name__ == '__main__':
    app.run(_main)
