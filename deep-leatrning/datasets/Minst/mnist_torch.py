# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/3/31
# User      : WuY
# File      : mnist_torch.py
# description : MNIST 数据集的 torch 版本

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 设置数据集
trainset = datasets.MNIST(root='./data',
                        train=True, 
                        download=True, 
                        transform=transforms.ToTensor())

testset = datasets.MNIST(root='./data', 
                         train=False, 
                         download=True, 
                         transform=transforms.ToTensor())

# 载入数据集
train_data_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=64,
        shuffle=True,
        drop_last=True)

test_data_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=64,
        shuffle=False,
        drop_last=False)

if __name__=="__main__":
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")

    images, labels = next(iter(train_data_loader))	# images：Tensor(64,1,28,28)、labels：Tensor(64,)

    img = torchvision.utils.make_grid(images)	# 把64张图片拼接为1张图片

    # pytorch网络输入图像的格式为（C, H, W)，而numpy中的图像的shape为（H,W,C）。故需要变换通道才能有效输出
    img = img.numpy().transpose(1, 2, 0)

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    print(labels)
    plt.imshow(img)
    plt.show()

