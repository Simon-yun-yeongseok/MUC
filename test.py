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

# def make_eval_model(iteration):
#     eval_models = []
#     for i in range (iteration):
#         eval_models.append("Eval_model_"+str(i+1))
#     print(eval_models)
#     for name in eval_models:
#         globals()[name] = [x for x in range(3)]
#     for name in eval_models:
#         for i in range (iteration):
#             print(name, "=", globals()[i])
    
a1 = 1
a2 = 2
b1 = 5
b2 = 6
    
if a1 >= a2:
    b1 = b2
else:
    b2 = b1+5
    

print(a1, a2, b1, b2)

# sum = 0
# for i in range(5):
#     globals()['var{}'.format(i)] = i
#     sum += globals()['var{}'.format(i)]
#     print(sum)
# for i in range(5):
#     print(globals()['var{}'.format(i)])

# iteration = 0
# for j in range(5):
#     iteration +=1
#     for i in range(iteration, iteration+1):
#         globals()['eval_model_{}.format(i)'] = tg_model
#         eval_model_i = i+3
#         print(eval_model_i)
# for i in range(5):
#     print(globals()['var{}'.format(i)])