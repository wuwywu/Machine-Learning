# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/4/2
# User      : WuY
# File      : CIFAR10.py
# description : CIFAR10 数据集的 torch 版本

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 设置数据集的类别
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# 设置数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
])

data_path = './data/CIFAR10'
trainset = datasets.CIFAR10(root=data_path,
                        train=True, 
                        download=True, 
                        transform=transforms.ToTensor())

testset = datasets.CIFAR10(root=data_path, 
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

    # 展示数据集
    fig = plt.figure(figsize=(8,3))
    num_classes = 10
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        ax.set_title(class_names[i])
        img = next(img for img, label in trainset if label == i)
        plt.imshow(img.permute(1, 2, 0).numpy())  # permute(1, 2, 0) 将通道维移到最后
    plt.show()
    