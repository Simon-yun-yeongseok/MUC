import torch.nn as nn
import math
import torch
import pdb
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import copy
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=40, side_classifier=3, is_sub_f=False):
        self.inplanes = 16
        self.outplanes = 1 # gmpark
        self.is_sub_f = is_sub_f
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.module_list = nn.ModuleList()

        if self.is_sub_f:
            self.sub_f = nn.Sequential(
                nn.Conv2d(64, self.outplanes, kernel_size=1, stride=1, bias=False),
                # nn.BatchNorm2d(self.outplanes),
                # nn.ReLU(inplace=True)
                # nn.Sigmoid()
            )
            # Ver.1 - Concat
            # self.fc = nn.Linear(64 * block.expansion + self.outplanes, num_classes)
            # self.fc_side = nn.Linear(64 * block.expansion + self.outplanes, num_classes*side_classifier)
            # Ver.2 - Replace, Ver.3 - Masking
            self.fc = nn.Linear(64 * block.expansion, num_classes)
            self.fc_side = nn.Linear(64 * block.expansion, num_classes*side_classifier)  # original
            # Ver.4 - AvgPool
            # self.module_list.append(nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),)
            # self.module_list.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            #                         nn.Conv2d(64, 64, 1, bias=False),
            #                         nn.BatchNorm2d(64),
            #                         nn.ReLU()
            #                         ))
            # for i, m in enumerate(self.module_list):
            #     if i == 0:
            #         m.weight.data.normal_(0, 0.01)
            #     else:
            #         for n in m:
            #             if isinstance(n, nn.Conv2d):
            #                 n.weight.data.normal_(0, 0.01)
            #             elif isinstance(n, nn.BatchNorm2d):
            #                 n.weight.data.fill_(1)
            #                 n.bias.data.zero_()                        
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)
            self.fc_side = nn.Linear(64 * block.expansion, num_classes*side_classifier)  # original
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, side_fc=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        
        if self.is_sub_f:
            sf = self.sub_f(x)
            sf = sf.view(sf.size(0), -1)  # [Batch, self.outplanes]

        x = x.view(x.size(0), -1)     # [Batch, 64]

        if self.is_sub_f:
            # Ver1. concatenate #
            # x = torch.cat((x, sf), dim=1)  # [Batch, 64 + self.outplanes]

            # # Ver2. replace the one that has max diff #
            for batch in range(x.size(0)):
                diff = (x[batch] - sf[batch].repeat(x[batch].size(0))).data.tolist()
                max_ind = diff.index(max(diff))
                x[batch, max_ind] = Variable(sf[batch])
            
            # Ver3. Masking
            # xm = copy.deepcopy(x)
            # for batch in range(x.size(0)):
            #     x[batch] = Variable(x[batch] * torch.sigmoid(sf[batch]).repeat(x[batch].size(0)))
            # Ver4. Avg-pool
            # out1 = self.module_list[0](x)
            # out2 = self.module_list[1](x)
            # x = out1 + out2               # [Batch, 64, 1, 1]
            # x = x.view(x.size(0), -1)     # [Batch, 64]

        if side_fc is False:
            x = self.fc(x)  # [Batch, 64 + self.outplanes] -> [Batch, num_classes]
            # xm = self.fc(xm)
        else:
            x = self.fc_side(x)

        return x

def resnet20(pretrained=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model

def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model
   
def resnet56(pretrained=False, **kwargs):
    n = 9
    model = ResNet(Bottleneck, [n, n, n], **kwargs)
    return model
