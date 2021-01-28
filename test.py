#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch import nn
import numpy as np
from PIL import Image
from copy import deepcopy
import example
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import argparse

arr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
arr1=arr[0:10]
# print(arr1)
for i in range(4):
    # print(arr[i*10:(i+1)*10])
    arr1=arr[i*10:(i+1)*10]
    print(arr1)
exit()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--dataset_dir', default='./data/cifar-100-python', type=str)
parser.add_argument('--OOD_dir', default='./data/SVHN', type=str)
parser.add_argument('--num_classes', default=100, type=int)
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

svhn_data = torchvision.datasets.SVHN(root=args.OOD_dir, download=True, transform=transform_train)
svhn_num = svhn_data.data.shape[0]
svhn_data_copy = svhn_data.data
svhn_labels_copy = svhn_data.labels



idx = torch.randperm(svhn_num)

svhn_data_sub = svhn_data_copy[idx]
svhn_labels_sub = svhn_labels_copy[idx]

map_svhn_data = svhn_data_sub[0:20]
# map_svhn_data = map_svhn_data.reshape(10000, 32, 32, 3)
map_svhn_labels = svhn_labels_sub[0:20]

X_svhn_sub = torch.tensor(map_svhn_data, dtype=torch.float32)
print(X_svhn_sub.shape, X_svhn_sub.size)
X_svhn_sub = X_svhn_sub.permute(0,3,2,1)
print(X_svhn_sub.shape, X_svhn_sub.size)
