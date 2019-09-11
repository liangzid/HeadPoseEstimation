'''
BifurcateNet is for the new result of head-pose-estimation.
          liangzid
        2019.8
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class BifurcateNet(nn.Module):
    # BifurcateNet is self-designed for the following reasons:
    # 1. higher accurate rate. 2. lower consume of computing.
    # to achieve it, we use ResNet34 to learn the global feature while using ConvNet to leanrn
    # the self-feature (three angles).finally we use FullyConneted the classification.

    def __init__(self,block,layers,num_bins):
        self.inplanes=64    #plane in num
        super(BifurcateNet,self).__init__()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.RuLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.yaw_conv=self._perFeatureBlock()
        self.pitch_conv=self._perFeatureBlock()
        self.roll_conv=self._perFeatureBlock()

        self.yaw_fc = nn.Linear(512, num_bins)
        self.pitch_fc = nn.Linear(512, num_bins)
        self.roll_fc = nn.Linear(512, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # ResNet34
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # extract per feature
        x1=self.yaw_conv(x)
        x2=self.pitch_conv(x)
        x3=self.roll_conv(x)

        x1 = x1.view(x.size(0), -1)
        x2 = x1.view(x.size(0), -1)
        x3 = x1.view(x.size(0), -1)

        x1=self.yaw_fc(x1)
        x2=self.pitch_fc(x2)
        x3=self.roll_fc(x3)

        return x1,x2,x3 

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

    def _perFeatureBlock(self):
        '''
        to extracted each feature of the EulerAngles.Just Copy it.
        '''
        Block=BasicBlock(512,512)
        return Block

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
