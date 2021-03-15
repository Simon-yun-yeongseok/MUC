import torch.nn as nn
import math
import torch
import pdb
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)   ##0315

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
    
    def __init__(self, block, layers, num_classes=40, side_classifier = 3):
        self.inplanes = 16
        self.outplanes = 2 # gmpark
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)

        self.sub_f = nn.Sequential(
            nn.Conv2d(64, self.outplanes, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(self.outplanes),
            # nn.ReLU(inplace=True)
        )
        self.sub_Avg = nn.AvgPool2d(1, stride=1)

        # self.fc = nn.Linear(64, num_classes)
        # self.fc_side = nn.Linear(64, num_classes*side_classifier)  # original
        self.fc = nn.Linear(64 * block.expansion + self.outplanes, num_classes)        # version 1
        self.fc3 = nn.Linear(1, num_classes)                                       # version 3

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
        sf = self.sub_f(x)
        sub_Avg = self.sub_Avg(x)     # [Batch, 64]

        x = x.view(x.size(0), -1)     # [Batch, 64]
        sf = sf.view(sf.size(0), -1)  # [Batch, self.outplanes]
        sub_Avg = sub_Avg.view(sub_Avg.size(0), -1)  # [Batch, self.outplanes]

        ## Ver1. concatenate #
        x = torch.cat((x, sf), dim=1)  # [Batch, 64 + self.outplanes]

        # Ver2. replace the one that has max diff #
        # for batch in range(x.size(0)):
        #     replace = int(x[batch].size(0) / self.outplanes)    # yunys
        #     diff = (x[batch] - sf[batch].repeat(replace)).data.tolist()
        #     max_ind = int(diff.index(max(diff)) / self.outplanes)
        #     # min_ind = int(diff.index(min(diff)) / self.outplanes)
        #     x[batch, max_ind] = Variable(sf[batch])

        # Ver3. set parallel the conv & Avgpool #
        # pdb.set_trace()
        # sub_Avg = self.fc(sub_Avg)  # [Batch, 64] -> [Batch, num_classes]
        # sf = self.fc3(sf)           # [Batch,  1] -> [Batch, num_classes]
        x = self.fc(x)              # [Batch, 64] -> [Batch, num_classes]
        # x = x + sf + sub_Avg

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
