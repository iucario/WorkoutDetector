# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

from __future__ import absolute_import, division, print_function

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, normal_
from workoutdetector.models import TSN, get_scheduler
from einops import rearrange


def create_model(num_class: int,
                 num_segments: int = 8,
                 base_model: str = 'resnet50',
                 checkpoint: str = None,
                 consensus_type='avg',
                 dropout: float = 0.5,
                 partial_bn: bool = False,
                 fc_lr5: bool = False) -> nn.Module:

    model = TSN(num_class=num_class,
                num_segments=num_segments,
                new_length=1,
                backbone_fn=tdn_net,
                base_model=base_model,
                consensus_type=consensus_type,
                dropout=dropout,
                partial_bn=partial_bn,
                fc_lr5=fc_lr5)

    if not checkpoint:
        return model
    print(("=> fine-tuning from '{}'".format(checkpoint)))
    sd = torch.load(checkpoint)
    sd = sd['state_dict']
    fc_layer_weight = list(sd.keys())[-2]
    fc_layer_bias = list(sd.keys())[-1]
    model_dict = model.state_dict()
    replace_dict = []
    for k, v in sd.items():
        if k not in model_dict and k.replace('.net', '') in model_dict:
            print('=> Load after remove .net: ', k)
            replace_dict.append((k, k.replace('.net', '')))
    for k, v in model_dict.items():
        if k not in sd and k.replace('.net', '') in sd:
            print('=> Load after adding .net: ', k)
            replace_dict.append((k.replace('.net', ''), k))

    for k, k_new in replace_dict:
        sd[k_new] = sd.pop(k)
    keys1 = set(list(sd.keys()))
    keys2 = set(list(model_dict.keys()))
    set_diff = (keys1 - keys2) | (keys2 - keys1)
    print('#### Notice: keys that failed to load: {}'.format(set_diff))
    # print(model_dict.keys())
    if sd[fc_layer_weight].shape != model_dict['new_fc.weight'].shape:
        print('=> New dataset, do not load fc weights')
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
    model_dict.update(sd)
    model.load_state_dict(model_dict)
    return model


def get_optimizer(
    model, epochs, lr, momentum, weight_decay, n_iter_per_epoch, warmup_epoch,
    lr_scheduler, lr_decay_rate, lr_steps, warmup_multiplier
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    policies = model.get_optim_policies()

    optimizer = torch.optim.SGD(policies,
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    scheduler = get_scheduler(optimizer, n_iter_per_epoch, lr_scheduler, lr_decay_rate,
                              warmup_epoch, lr_steps, epochs, warmup_multiplier)
    return optimizer, scheduler


class TDN_Net(nn.Module):

    def __init__(self,
                 resnet_model,
                 resnet_model1,
                 alpha: float = 0.5,
                 beta: float = 0.5):
        super(TDN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        # implement conv1_5 and inflate weight
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         dilation=1,
                                         ceil_mode=False)
        # only need this one layer?
        self.resnext_layer1 = nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = list(resnet_model.children())[8]
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        """Only diff information is used in the model.
        num_segments does not matter. I don't understand.
        """

        # original: [batch, num_seg, 5, 3, 224, 224]
        # Reshaped to: x = original.view((-1, 3*5) + original.size()[-2:])
        # It's the same as: x = rearrange(original, 'b s d c h w -> (b s) (d c) h w')
        x1, x2, x3, x4, x5 = [x[:, i:i + 3, ...] for i in range(0, 13, 3)]
        t = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4],
                      1)  # [batch*num_seg, 12, 224, 224]
        # After avg_diff: [batch*num_seg, 12, 112, 112]
        x_c5 = self.conv1_5(self.avg_diff(t.view(-1, 12, x2.size()[2], x2.size()[3])))
        # x_c5 [batch*num_seg, 64, 56, 56]
        x_diff = self.maxpool_diff(1.0 / 1.0 * x_c5)  # [batch*num_seg, 64, 28, 28]

        temp_out_diff1 = x_diff
        x_diff = self.resnext_layer1(x_diff)

        x = self.conv1(x3)  # x3 is the center frame
        x = self.bn1(x)
        x = self.relu(x)
        #fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.alpha * x + self.beta * temp_out_diff1
        #fusion layer2
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.alpha * x + self.beta * x_diff

        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)  # [batch*num_seg, 2048, 1, 1]
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def tdn_net(base_model=None, num_segments=8, pretrained=True):
    if ("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if (num_segments == 8):
        model = TDN_Net(resnet_model, resnet_model1, alpha=0.5, beta=0.5)
    else:
        model = TDN_Net(resnet_model, resnet_model1, alpha=0.75, beta=0.25)
    return model


class mSEModule(nn.Module):

    def __init__(self, channel, n_segment=8, index=1):
        super(mSEModule, self).__init__()
        self.channel = channel
        self.reduction = 16
        self.n_segment = n_segment
        self.stride = 2**(index - 1)
        self.conv1 = nn.Conv2d(in_channels=self.channel,
                               out_channels=self.channel // self.reduction,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel // self.reduction,
                               out_channels=self.channel // self.reduction,
                               kernel_size=3,
                               padding=1,
                               groups=self.channel // self.reduction,
                               bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_forward = nn.Sigmoid()

        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2,
                                               stride=2)  #nn.AdaptiveMaxPool2d(1)
        self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_backward = nn.Sigmoid()

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(in_channels=self.channel // self.reduction,
                               out_channels=self.channel,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv3_smallscale2 = nn.Conv2d(in_channels=self.channel // self.reduction,
                                           out_channels=self.channel // self.reduction,
                                           padding=1,
                                           kernel_size=3,
                                           bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

        self.conv3_smallscale4 = nn.Conv2d(in_channels=self.channel // self.reduction,
                                           out_channels=self.channel // self.reduction,
                                           padding=1,
                                           kernel_size=3,
                                           bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel // self.reduction)

    def spatial_pool(self, x):
        nt, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(nt, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(nt, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.view(nt, 1, height, width)
        return context_mask

    def forward(self, x):
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view(
            (-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w

        t_fea_forward, _ = reshape_bottleneck.split([self.n_segment - 1, 1],
                                                    dim=1)  # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split([1, self.n_segment - 1],
                                                     dim=1)  # n, t-1, c//r, h, w

        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view(
            (-1, self.n_segment) + conv_bottleneck.size()[1:])  # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split(
            [1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        tPlusone_fea_backward, _ = reshape_conv_bottleneck.split(
            [self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward  # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward  # n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(diff_fea_forward,
                                          self.pad1_forward,
                                          mode="constant",
                                          value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view(
            (-1,) + diff_fea_pluszero_forward.size()[2:])  #nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(diff_fea_backward,
                                           self.pad1_backward,
                                           mode="constant",
                                           value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view(
            (-1,) + diff_fea_pluszero_backward.size()[2:])  #nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(
            diff_fea_pluszero_forward)  # nt, c//r, 1, 1
        y_backward_smallscale2 = self.avg_pool_backward2(
            diff_fea_pluszero_backward)  # nt, c//r, 1, 1

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(
            self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(
            self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(
            self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(
            self.conv3_smallscale4(y_backward_smallscale4))

        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2,
                                              diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2,
                                               diff_fea_pluszero_backward.size()[2:])

        y_forward = self.bn3(
            self.conv3(1.0 / 3.0 * diff_fea_pluszero_forward +
                       1.0 / 3.0 * y_forward_smallscale2 +
                       1.0 / 3.0 * y_forward_smallscale4))  # nt, c, 1, 1
        y_backward = self.bn3(
            self.conv3(1.0 / 3.0 * diff_fea_pluszero_backward +
                       1.0 / 3.0 * y_backward_smallscale2 +
                       1.0 / 3.0 * y_backward_smallscale4))  # nt, c, 1, 1

        y_forward = self.sigmoid_forward(y_forward) - 0.5
        y_backward = self.sigmoid_backward(y_backward) - 0.5

        y = 0.5 * y_forward + 0.5 * y_backward
        output = x + x * y
        return output


class ShiftModule(nn.Module):

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div * self.fold,
                              self.fold_div * self.fold,
                              kernel_size=3,
                              padding=1,
                              groups=self.fold_div * self.fold,
                              bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1  # shift left
            self.conv.weight.data[self.fold:2 * self.fold, 0, 0] = 1  # shift right
            if 2 * self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1  # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1  # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1)  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2)  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


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

    def __init__(self,
                 num_segments: int,
                 inplanes: int,
                 planes: int,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class BottleneckShift(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BottleneckShift, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments
        self.mse = mSEModule(planes, n_segment=num_segments, index=1)
        self.shift = ShiftModule(planes, n_segment=num_segments, n_div=8, mode='shift')

        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.mse(out)
        out = self.shift(out)

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


class FBResNet(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000):
        self.inplanes = 64

        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments
        super(FBResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments, Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments, block, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_segments, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


model_urls = {
    'fbresnet18': 'https://data.lip6.fr/cadene/pretrainedmodels/resnet18-5c106cde.pth',
    'fbresnet50': 'https://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
    'fbresnet101': 'https://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
}


def fbresnet50(num_segments=8, pretrained=False, num_classes=1000):
    ckpt = 'checkpoints/resnet50-19c8e357.pth'
    url = 'https://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth'
    model = FBResNet(num_segments, BottleneckShift, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['fbresnet50']), strict=False)
    return model


def fbresnet101(num_segments, pretrained=False, num_classes=1000):
    url = 'https://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
    ckpt = "checkpoints/resnet101-5d3b4d8f.pth"
    model = FBResNet(num_segments,
                     BottleneckShift, [3, 4, 23, 3],
                     num_classes=num_classes)
    if pretrained:
        model.load_state_dict(ckpt, strict=False)
    return model


if __name__ == '__main__':
    device = 'cpu'
    ckpt_path = 'checkpoints/tdn_sthv2_r50_8x1x1.pth'
    batch = 4
    num_class = 10
    num_diff = 5
    num_seg = 8
    dummy_x = torch.randn(batch, num_seg, num_diff, 3, 224, 224)
    model = create_model(num_class=num_class,
                         num_segments=num_seg,
                         base_model='resnet50',
                         checkpoint=ckpt_path)
    model = model.to(device)
    model.eval()
    y = model(dummy_x.to(device))
    print(y.shape)
    assert y.shape == (batch, num_class)
