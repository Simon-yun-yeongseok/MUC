#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *
 
def compute_accuracy_WI(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)            
            outputs = tg_model(inputs, side_fc=False)            
            outputs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_Version1(tg_model, evalloader, nb_cl, nclassifier, iteration):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            outputs = tg_model(inputs, side_fc=True)
            #outputs = F.softmax(outputs, dim=1)
            real_classes = int(outputs.size(1)/nclassifier)
            nstep = iteration+1
            outputs_sum = torch.zeros(outputs.size(0), real_classes).to(device)
            ##
            for i in range(nstep):
                start = nb_cl*nclassifier*i
                for j in range(nclassifier):
                    end = start+nb_cl
                    outputs_sum[:, i*nb_cl:(i+1)*nb_cl] += outputs[:, start:end]
                    start = end
            outputs_sum = F.softmax(outputs_sum, dim=1)
            _, predicted = outputs_sum.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100. * correct / total

    return cnn_acc


class TEST:
    def __init__(self, epochs=200, val_epoch=10, num_classes=100, nb_cl=20):
        self.epochs = epochs
        self.val_epoch = val_epoch
        self.num_classes = num_classes
        self.nb_cl = nb_cl
        
    def compute_accuracy_test(self, tg_model, evalloader, start_class, end_class):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        correct = np.zeros((int(self.epochs/self.val_epoch), int(self.num_classes/self.nb_cl)))
        total = 0
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            outputs = tg_model(inputs)
            for i in range(int(self.epochs/self.val_epoch)):
                outputs_eval = outputs[:, 20*i:20*(i+1)]                            
                outputs_eval = F.softmax(outputs_eval, dim=1)
            print(outputs_eval)    
            exit()
                # _, predicted1 = outputs_eval.max()
                # for j in range(int(self.num_classes/self.nb_cl)):
                #     correct[i][j] = predicted1.eq(targets).sum().item()
        return correct
