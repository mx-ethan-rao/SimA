import os.path
import sys

# sys.path.append('.')

import torch
import torchvision.datasets
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Subset


# from measures import ssim
# from score import fid


class ReconsDataset(torch.utils.data.Dataset):

    def __init__(self, data, transforms=None):
        super(ReconsDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image


class MIASTL10(torchvision.datasets.STL10):

    def __init__(self, idxs, **kwargs):
        super(MIASTL10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASTL10, self).__getitem__(item)


class MIACelebA(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.img_dir = os.path.join(root, "img_align_celeba")
        self.attr = pd.read_csv(os.path.join(root, "list_attr_celeba.csv"))
        self.part = pd.read_csv(os.path.join(root, "list_eval_partition.csv"))
        self.transform = transform
        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split_idx = split_map[split]
        if split_idx is not None:
            ids = self.part[self.part["partition"] == split_idx]["image_id"]
            self.attr = self.attr[self.attr["image_id"].isin(ids)]
        self.files = self.attr["image_id"].tolist()
        self.attrs = self.attr.drop(columns=["image_id"]).astype("int32").values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        target = torch.tensor(self.attrs[idx])
        if self.transform:
            img = self.transform(img)
        return img, target


class MIASVHN(torchvision.datasets.SVHN):

    def __init__(self, idxs, **kwargs):
        super(MIASVHN, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASVHN, self).__getitem__(item)


class MIACIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR10, self).__getitem__(item)
    
class MIACIFAR101(Dataset):
    def __init__(self, root, transform=None):
        self.data = np.load(os.path.join(root, "cifar10.1_v6_data.npy"))
        self.labels = np.load(os.path.join(root, "cifar10.1_v6_labels.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


class MIACIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR100, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR100, self).__getitem__(item)


class MIAImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, idxs, **kwargs):
        super(MIAImageFolder, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIAImageFolder, self).__getitem__(item)


def load_member_data(dataset_root, dataset_name, batch_size=128, shuffle=False, randaugment=False):
    if dataset_name.upper() == 'CIFAR10':
        splits = np.load(os.path.join(dataset_root, 'CIFAR10', 'CIFAR10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=5),
                                                         torchvision.transforms.ToTensor()])
        else:
            transforms = torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor()])
        member_set = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'CIFAR10'), train=True,
                                transform=transforms, download=True)
        nonmember_set = MIACIFAR10(nonmember_idxs, root=os.path.join(dataset_root, 'CIFAR10'), train=True,
                                   transform=transforms, download=True)
    elif dataset_name.upper() == 'CIFAR100':
        splits = np.load(os.path.join(dataset_root, 'CIFAR100', 'CIFAR100_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        member_set = MIACIFAR100(member_idxs, root=os.path.join(dataset_root, 'CIFAR100'), train=True,
                                 transform=transforms, download=True)
        nonmember_set = MIACIFAR100(nonmember_idxs, root=os.path.join(dataset_root, 'CIFAR100'), train=True,
                                    transform=transforms, download=True)
    elif dataset_name.upper() == 'CIFAR101':
        splits = np.load(os.path.join(dataset_root, 'CIFAR10', 'CIFAR10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=5),
                                                         torchvision.transforms.ToTensor()])
        else:
            transforms = torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor()])
        member_set = MIACIFAR10(member_idxs[:2000], root=os.path.join(dataset_root, 'CIFAR10'), train=True,
                                transform=transforms, download=True)
        nonmember_set = MIACIFAR101(root='/banana/ethan/MIA_data/CIFAR101/datasets', transform=transforms)
    elif dataset_name.upper() == 'CELEBA':
        splits = np.load(os.path.join(dataset_root, 'CELEBA', 'CELEBA_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        # member_set = MIACelebA(member_idxs, root=os.path.join(dataset_root, 'CELEBA'), split='train',
        #                        transform=transforms, download=False)
        # nonmember_set = MIACelebA(nonmember_idxs, root=os.path.join(dataset_root, 'CELEBA'), split='train',
        #                           transform=transforms, download=False)
        datasets = MIACelebA(root=os.path.join(dataset_root, 'CELEBA','celeba'), split="train", transform=transforms)
        member_set = Subset(datasets, member_idxs)
        nonmember_set = Subset(datasets, nonmember_idxs)
    elif dataset_name.upper() == 'STL10':
        splits = np.load(os.path.join(dataset_root, 'STL10', 'STL10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'STL10'), split='train',
                              download=True, transform=transforms)
        nonmember_set = MIASTL10(nonmember_idxs, root=os.path.join(dataset_root, 'STL10'), split='train',
                                 download=True, transform=transforms)
    elif dataset_name.upper() == 'STL10-U':
        splits = np.load(os.path.join(dataset_root, 'STL10-U', 'STL10-U_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'STL10-U'), split='unlabeled',
                              download=True, transform=transforms)
        nonmember_set = MIASTL10(nonmember_idxs, root=os.path.join(dataset_root, 'STL10-U'), split='unlabeled',
                                 download=True, transform=transforms)
    else:
        raise NotImplemented

    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, shuffle=shuffle)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, shuffle=shuffle)
    return member_set, nonmember_set, member_loader, nonmember_loader
