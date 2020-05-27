import sys
import os

from collections import OrderedDict

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True
import functools

BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36), norm=BatchNorm2d):
        super(ASPPModule, self).__init__()

        # self.conv1 = nn.Sequential(OrderedDict([
        #     ('pool', nn.AdaptiveAvgPool2d((1, 1))),
        #     ('conv', nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False)),
        #     ('bn', BatchNorm2d(inner_features)),
        # ]))
        # self.conv2 = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False)),
        #     ('bn', BatchNorm2d(inner_features)),
        # ]))
        # self.conv3 = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False)),
        #     ('bn', BatchNorm2d(inner_features)),
        # ]))
        # self.conv4 = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False)),
        #     ('bn', BatchNorm2d(inner_features)),
        # ]))
        # self.conv5 = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False)),
        #     ('bn', BatchNorm2d(inner_features)),
        # ]))

        # self.bottleneck = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False)),
        #     ('bn', BatchNorm2d(out_features)),
        #     ('dropout', nn.Dropout2d(0.1)),
        # ]))

        conv1_list = [
            nn.AdaptiveAvgPool2d((1, 1)),
        ]
        if isinstance(norm, BatchNorm2d):
            conv1_list += [
                nn.Conv2d(features, inner_features, kernel_size=1,
                          padding=0, dilation=1, bias=False),
                norm(inner_features)
            ]
        else:
            conv1_list += [
                nn.Conv2d(features, inner_features, kernel_size=1,
                          padding=0, dilation=1, bias=True),
            ]

        self.conv1 = nn.Sequential(*conv1_list)
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   norm(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   norm(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   norm(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm(out_features),
            nn.Dropout2d(0.1)
            )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(
            h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, is_bn_freezed=True):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

        # self.head = nn.Sequential(OrderedDict([
        #     ('aspp', ASPPModule(2048)),
        #     ('conv', nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)),
        # ]))

        # self.dsn = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)),
        #     ('bn', BatchNorm2d(512)),
        #     ('drop', nn.Dropout2d(0.1)),
        #     ('conv2', nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)),
        # ]))
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self._is_bn_freezed = is_bn_freezed
        if self._is_bn_freezed:
            norm = nn.InstanceNorm2d

        self.head = nn.Sequential(
            ASPPModule(2048, norm=norm),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            norm(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample=nn.Sequential(OrderedDict([
            #     ('conv', nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)),
            #     ('bn', BatchNorm2d(planes * block.expansion, affine=affine_par)),
            # ]))
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []

        def generate_multi_grid(index, grids): return grids[index % len(
            grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation,
                            downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)
        return [x, x_dsn]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
        self._is_bn_freezed = True

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(ResNet, self).train(mode)
        if self._is_bn_freezed:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduce=True, size_average=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, size_average=size_average, reduce=reduce
        )
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4


def get_deeplabV3(num_classes, model_path_prefix=None, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

    if pretrained:
        saved_state_dict = torch.load(os.path.join(
            model_path_prefix, 'resnet101-imagenet.pth'
        ))
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            # if not i_parts[1]=='layer5':
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

        model.load_state_dict(new_params)
    return model


if __name__ == "__main__":
    net = DeeplabV3(21, '/home/users/liangchen.song/data/models/')
    # for name, _ in net.named_parameters():
    #     print(name)
    # for name, module in net.named_modules():
    #     print(name, isinstance(module, BatchNorm2d))
    #     for param in module.parameters():
    #         param.requires_grad = False
