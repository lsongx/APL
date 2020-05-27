import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                ),
            )

        for m in self.stages.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
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


class IBNResNetDeepLab(nn.Module):

    def __init__(self, block, layers, pyramids, n_classes=19):
        scale = 64
        self.inplanes = scale
        super(IBNResNetDeepLab, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2)
        # # 8 down
        # self.layer3 = self._make_layer(block, scale*4, layers[2], dilation=2)
        # self.layer4 = self._make_layer(block, scale*8, layers[3], dilation=4)
        # self.aspp = _ASPPModule(2048, n_classes, pyramids)
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # 16 down
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], dilation=2)
        self.aspp = _ASPPModule(2048, n_classes, pyramids)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_style=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        if return_style:
            return l1, l2
        x = self.layer3(l2)
        x = self.layer4(x)

        x = self.aspp(x)
        x = self.upsample(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.eval()


def resnet101_ibn_a_deeplab(model_path_prefix=None, n_classes=19):
    model = IBNResNetDeepLab(
        Bottleneck, layers=[3, 4, 23, 3], pyramids=[6, 12, 18, 24], n_classes=19
    )
    if model_path_prefix is None:
        return model
    else:
        model = model.cuda()
        model_state_dict = torch.load(
            os.path.join(model_path_prefix, 'resnet101_ibn_a.pth')
        )
        model_state_dict = model_state_dict['state_dict']
        state_name_list = list(model_state_dict.keys())
        for key in state_name_list:
            if 'fc' in key:
                del model_state_dict[key]
                continue
            new_key = key.replace('module.', '')
            model_state_dict[new_key] = model_state_dict.pop(key)
        model.load_state_dict(model_state_dict, strict=False)
        return model


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def get_seg_optimizer(deeplab_model, opt):
    seg_optimizer = torch.optim.SGD(
        params=[
            {
                "params": get_params(deeplab_model, key="1x"),
                "lr": opt.learning_rate,
                "weight_decay": opt.weight_decay,
            },
            {
                "params": get_params(deeplab_model, key="10x"),
                "lr": 10 * opt.learning_rate,
                "weight_decay": opt.weight_decay,
            },
            {
                "params": get_params(deeplab_model, key="20x"),
                "lr": 20 * opt.learning_rate,
                "weight_decay": 0.0,
            },
        ],
        momentum=opt.momentum,
    )
    return seg_optimizer


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    optimizer.param_groups[1]["lr"] = 10 * new_lr
    optimizer.param_groups[2]["lr"] = 20 * new_lr
