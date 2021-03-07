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
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import argparse
import copy
import sys


data_dir = 'tiny-imagenet-200/'
num_workers = {'train' : 100,'val'   : 0,'test'  : 0}
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['train', 'val','test']}
trainloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
                  for x in ['train']}
evalloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=False, num_workers=num_workers[x])
                  for x in ['val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}