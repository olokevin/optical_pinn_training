"""
Description:
Author:
Date: 2021-10-24 16:39:50
LastEditors:
LastEditTime: 2021-10-24 16:39:51
"""
import os
import pickle
import sys

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from pyutils.config import configs

def transform_default(img_height, img_width):
    return transforms.Compose([transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])

def resize_crop(img_height, img_width):
    return transforms.Compose(
              [transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC), 
                transforms.CenterCrop((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            )


def get_dataset(dataset, img_height, img_width, dataset_dir="./data", transform=None):
    if transform == "resize_crop":
        _transform = resize_crop(img_height, img_width)
    else:
        _transform = transform_default(img_height, img_width)
    
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            dataset_dir,
            train=True,
            download=True,
            transform=_transform,
        )

        validation_dataset = datasets.MNIST(
            dataset_dir,
            train=False,
            transform=_transform,
        )
    elif dataset == "fashionmnist":
        train_dataset = datasets.FashionMNIST(
            dataset_dir,
            train=True,
            download=True,
            transform=_transform,
        )

        validation_dataset = datasets.FashionMNIST(
            dataset_dir,
            train=False,
            transform=_transform,
        )
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=_transform,
        )

        validation_dataset = datasets.CIFAR10(
            dataset_dir,
            train=False,
            transform=_transform,
        )
    elif dataset == 'imagenet':
        train_dataset = datasets.ImageNet(
            dataset_dir, 
            split='train',
            transform=_transform,
        )
        validation_dataset = datasets.ImageNet(
            dataset_dir, 
            split='val',
            transform=_transform,
        )
    elif dataset == "vowel4_4":
        train_dataset = VowelRecog(os.path.join(dataset_dir, "vowel4_4/processed"), mode="train")
        validation_dataset = VowelRecog(os.path.join(dataset_dir, "vowel4_4/processed"), mode="test")

    return train_dataset, validation_dataset


class VowelRecog(torch.utils.data.Dataset):
    def __init__(self, path, mode="train"):
        self.path = path
        assert os.path.exists(path)
        assert mode in ["train", "test"]
        self.data, self.labels = self.load(mode=mode)

    def load(self, mode="train"):
        with open(f"{self.path}/{mode}.pt", "rb") as f:
            data, labels = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            # data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        return data, labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

