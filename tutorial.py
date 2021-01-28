import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import argparse
import torchvision
from torchvision import datasets, models, transforms
import utils_pytorch
import resnet_model
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./data/cifar-100-python', type=str)
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])

trainset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=True, transform=transform_test)
evalset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=False, transform=transform_test)


class CustomDataset(Dataset, x_data, y_data): 
  def __init__(self):
    self.x_data
    self.y_data

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y