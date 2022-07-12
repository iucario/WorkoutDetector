# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn
from typing import Callable

import torch
from torch import nn
from torch.nn.init import constant_, normal_


class TSNHead(nn.Module):
    """Class head for TSN.
    Modified from mmaction/models/heads/tsn_head.py

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 consensus_type='avg',
                 dropout_ratio=0.4,
                 init_std=0.01):

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.consensus = ConsensusModule(consensus_type)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        module = self.fc_cls
        mean = 0
        std = self.init_std
        bias = 0
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


class TSN(nn.Module):
    """Only for TDN"""

    def __init__(self,
                 num_class: int,
                 num_segments: int,
                 backbone_fn: Callable,
                 num_frames: int = 5,
                 base_model: str = 'resnet50',
                 consensus_type: str = 'avg',
                 dropout: float = 0.8,
                 init_std: float = 0.01,
                 partial_bn=True,
                 fc_lr5=False):
        super(TSN, self).__init__()
        assert 'resnet' in base_model, ValueError(f'Unknown base model: {base_model}')
        self.num_class = num_class
        self.num_segments = num_segments
        self.backbone_fn = backbone_fn  # tdn_net()
        self.reshape = True
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.base_model = base_model
        self.fc_lr5 = fc_lr5  # fine_tuning for UCF/HMDB
        self.init_std = init_std
        self._prepare_base_model(base_model, num_segments)
        self._prepare_tsn()
        self.consensus = ConsensusModule(consensus_type)
        self.modality = 'RGB'
        self.new_length = num_frames
        self.before_softmax = True
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self):

        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, self.num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, self.num_class)

        if self.new_fc is None:
            normal_(
                getattr(self.base_model, self.base_model.last_layer_name).weight, 0,
                self.init_std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, self.init_std)
                constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self, base_model, num_segments):
        self.base_model = self.backbone_fn(base_model, num_segments)
        self.base_model.last_layer_name = 'fc'
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        inorm = []
        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".
                        format(type(m)))

        if self.fc_lr5:  # fine_tuning for UCF/HMDB
            return [
                {
                    'params': first_conv_weight,
                    'lr_mult': 5 if self.modality == 'Flow' else 1,
                    'decay_mult': 1,
                    'name': "first_conv_weight"
                },
                {
                    'params': first_conv_bias,
                    'lr_mult': 10 if self.modality == 'Flow' else 2,
                    'decay_mult': 0,
                    'name': "first_conv_bias"
                },
                {
                    'params': normal_weight,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "normal_weight"
                },
                {
                    'params': normal_bias,
                    'lr_mult': 2,
                    'decay_mult': 0,
                    'name': "normal_bias"
                },
                {
                    'params': bn,
                    'lr_mult': 1,
                    'decay_mult': 0,
                    'name': "BN scale/shift"
                },
                {
                    'params': custom_ops,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "custom_ops"
                },
                {
                    'params': lr5_weight,
                    'lr_mult': 5,
                    'decay_mult': 1,
                    'name': "lr5_weight"
                },
                {
                    'params': lr10_bias,
                    'lr_mult': 10,
                    'decay_mult': 0,
                    'name': "lr10_bias"
                },
            ]
        else:  # default
            return [
                {
                    'params': first_conv_weight,
                    'lr_mult': 5 if self.modality == 'Flow' else 1,
                    'decay_mult': 1,
                    'name': "first_conv_weight"
                },
                {
                    'params': first_conv_bias,
                    'lr_mult': 10 if self.modality == 'Flow' else 2,
                    'decay_mult': 0,
                    'name': "first_conv_bias"
                },
                {
                    'params': normal_weight,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "normal_weight"
                },
                {
                    'params': normal_bias,
                    'lr_mult': 2,
                    'decay_mult': 0,
                    'name': "normal_bias"
                },
                {
                    'params': bn,
                    'lr_mult': 1,
                    'decay_mult': 0,
                    'name': "BN scale/shift"
                },
                {
                    'params': custom_ops,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "custom_ops"
                },
            ]

    def forward(self, input, reshape=True):
        if reshape:
            sample_len = 3 * self.new_length
            base_out = self.base_model(input.reshape((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output.squeeze(1)


class ConsensusModule(nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class SegmentConsensus(nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output
