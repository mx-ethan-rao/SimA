import numpy as np, random, torch, logging


import torch
from rich.progress import track
import logging
import warnings
from rich.logging import RichHandler
import components
from typing import Type, Dict
from itertools import chain
from model import UNet
from dataset_utils import load_member_data
from torchmetrics.classification import BinaryAUROC, BinaryROC
from absl import app, flags


# def get_FLAGS():

#     def FLAGS(x): return x
#     FLAGS.T = 1000
#     FLAGS.ch = 128
#     FLAGS.ch_mult = [1, 2, 2, 2]
#     FLAGS.attn = [1]
#     FLAGS.num_res_blocks = 2
#     FLAGS.dropout = 0.1
#     FLAGS.beta_1 = 0.0001
#     FLAGS.beta_T = 0.02

#     return FLAGS

FLAGS = flags.FLAGS
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')

flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_string('dataset', 'CIFAR10', help='dataset')
flags.DEFINE_string('attacker', 'SimA', help='Attack model')
flags.DEFINE_string('dataset_root', '', help='data set')
flags.DEFINE_string('checkpoint', '', help='Checkpoint to load')
flags.DEFINE_integer('interval', 20, help='Interval between attacks')
flags.DEFINE_integer('attack_num', 5, help='Number of attacks')
flags.DEFINE_integer('seed', 2025, help='Seed')



def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False



def get_model(ckpt, WA=True):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


class EpsGetter(components.EpsGetter):
    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        t = torch.ones([xt.shape[0]], device=xt.device).long() * t
        return self.model(xt, t=t)


attackers: Dict[str, Type[components.DDIMAttacker]] = {
    "SecMI": components.SecMI,
    "PIA": components.PIA,
    "Loss": components.Loss,
    "PIAN": components.PIAN,
    "SimA": components.SimA,
    "PFAMI": components.PFAMI,
    "Epsilon": components.Epsilon,
}


DEVICE = 'cuda'


@torch.no_grad()
def attack():
    seed_all(FLAGS.seed)

    # FLAGS = get_FLAGS()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())

    logger.info("loading model...")
    model = get_model(FLAGS.checkpoint, WA = True).to(DEVICE)
    model.eval()

    logger.info("loading dataset...")
    if FLAGS.dataset in ['CIFAR10', 'CIFAR100', 'STL10-U', 'CELEBA', 'CIFAR101']:

        _, _, train_loader, test_loader = load_member_data(dataset_root=FLAGS.dataset_root, dataset_name=FLAGS.dataset, batch_size=64,
                                                           shuffle=False, randaugment=False)

    attacker = attackers[FLAGS.attacker](
        torch.from_numpy(np.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T)).to(DEVICE), FLAGS.interval, FLAGS.attack_num, EpsGetter(model), lambda x: x * 2 - 1)

    logger.info("attack start...")
    members, nonmembers = [], []
    for member, nonmember in track(zip(train_loader, chain(*([test_loader]))), total=len(test_loader)):
        member, nonmember = member[0].to(DEVICE), nonmember[0].to(DEVICE)

        members.append(attacker(member))
        nonmembers.append(attacker(nonmember))

        members = [torch.cat(members, dim=-1)]
        nonmembers = [torch.cat(nonmembers, dim=-1)]

    member = members[0]
    nonmember = nonmembers[0]

    # auroc = [BinaryAUROC().cuda()(torch.cat([member[i] / max([member[i].max().item(), nonmember[i].max().item()]), nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()).item() for i in range(member.shape[0])]
    # tpr_fpr = [BinaryROC().cuda()(torch.cat([1 - nonmember[i] / max([member[i].max().item(), nonmember[i].max().item()]), 1 - member[i] / max([member[i].max().item(), nonmember[i].max().item()])]), torch.cat([torch.zeros(member.shape[1]).long(), torch.ones(nonmember.shape[1]).long()]).cuda()) for i in range(member.shape[0])]
    # tpr_fpr_1 = [i[1][(i[0] < 0.01).sum() - 1].item() for i in tpr_fpr]
    # cp_auroc = auroc[:]
    # cp_auroc.sort(reverse=True)
    # cp_tpr_fpr_1 = tpr_fpr_1[:]
    # cp_tpr_fpr_1.sort(reverse=True)
    # print('auc', auroc)
    # print('tpr @ 1% fpr', cp_tpr_fpr_1)
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
