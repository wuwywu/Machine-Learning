# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/4/3
# User      : WuY
# File      : CelebA.py
# description : CelebA 数据集的 torch 版本

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# 下载困难可以从网上下载
## 在相应的位置包含文件("img_align_celeba.zip", "list_attr_celeba.txt", "identity_CelebA.txt", "list_bbox_celeba.txt", "list_eval_partition.txt", "list_landmarks_align_celeba.txt")

data_path = './data/CelebA'
trainset = datasets.CelebA(root=data_path,
                        split='train', 
                        download=True, 
                        transform=transforms.ToTensor())

validset = datasets.CelebA(root=data_path,
                        split='valid', 
                        download=True, 
                        transform=transforms.ToTensor())

testset = datasets.CelebA(root=data_path, 
                        split='test', 
                        download=True, 
                        transform=transforms.ToTensor())


# 载入数据集
train_data_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

valid_data_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

test_data_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

