# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# Modified by: me

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, normal_
import torchvision
from workoutdetector.models.build import MODEL_REGISTRY


class TemporalShift(nn.Module):

    def __init__(self,
                 net: nn.Module,
                 n_segment: int = 3,
                 n_div: int = 8,
                 inplace: bool = False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            out = InplaceShift.apply(x, fold)
            # raise NotImplementedError
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold:2 * fold]
        input.data[:, :, fold:2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold:2 * fold]
        grad_output.data[:, :, fold:2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):

    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net: nn.Module,
                        n_segment: int,
                        n_div=8,
                        place='blockres',
                        temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0

    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            for j, seg in enumerate(n_segment_list, 1):
                blocks = list(getattr(net, f'layer{j}').children())
                # print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=seg, n_div=n_div)

                setattr(net, f'layer{j}', nn.Sequential(*(blocks)))

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
            for j, seg in enumerate(n_segment_list, 1):
                blocks = list(getattr(net, f'layer{j}').children())
                # print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1,
                                                        n_segment=seg,
                                                        n_div=n_div)
                setattr(net, f'layer{j}', nn.Sequential(*(blocks)))
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        # print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


class Identity(torch.nn.Module):

    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

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


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


# @MODEL_REGISTRY.register()
class TSM(nn.Module):
    """TSN with temporal shift module
    Input shape: (batch_size*num_segments, channel, height, width)
    
    Args:
        num_class (int): number of classes
        num_segments (int): number of segments, default 8
        base_model (str): base model name, default 'resnet50'
        consensus_type (str): 'avg' or 'identity', default 'avg'.
        before_softmax (bool): If True, output raw score, else softmax. Default True.
        dropout (float): dropout rate, default 0.5
        img_feature_dim (int): I don't think it's used.
        partial_bn (bool): use partial bn or not, default True
        print_spec (bool): print out the spec of the model, default False
        is_shift (bool): use temporal shift module or not, default True
        shift_div (int): the number of folds of temporal shift, default 8
        shift_place (str): 'block' or 'blockres', default 'blockres'
        fc_lr5 (bool): If True, multiply nn.Linear layers' weights by 5, 
        and multiply nn.Linear layers' bias by 10. default False
        temporal_pool (bool): use temporal pooling or not, default False
        non_local (bool): use non-local module or not, default False
    """

    def __init__(self,
                 num_class,
                 num_segments=8,
                 base_model='resnet50',
                 consensus_type='avg',
                 before_softmax=True,
                 dropout=0.5,
                 img_feature_dim=256,
                 partial_bn=True,
                 is_shift=True,
                 shift_div=8,
                 shift_place='blockres',
                 fc_lr5=False,
                 non_local=False):
        super(TSM, self).__init__()
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim
        self.temporal_pool = False
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model = base_model
        self.fc_lr5 = fc_lr5
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")
        std = 0.001
        self._prepare_base_model(base_model)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        feature_dim = self.base_model.fc.in_features
        self.fc = nn.Linear(feature_dim, num_class)
        normal_(self.fc.weight, 0, std)
        constant_(self.fc.bias, 0)
        self.consensus = ConsensusModule(consensus_type)
        self.base_model = nn.Sequential(
            OrderedDict(list(self.base_model.named_children())[:-2]))

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        self.fc = nn.Linear(feature_dim, num_class)
        init_std = 0.001
        normal_(self.fc.weight, 0, init_std)
        constant_(self.fc.bias, 0)

    def _prepare_base_model(self, base_model: str):
        # print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            if self.is_shift:
                # print('Adding temporal shift...')
                make_temporal_shift(self.base_model,
                                    self.num_segments,
                                    n_div=self.shift_div,
                                    place=self.shift_place,
                                    temporal_pool=self.temporal_pool)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(TSM, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
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
                        f"New atomic module type: {type(m)}. Need to give it a learning policy"
                    )

        return [
            {
                'params': first_conv_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "first_conv_weight"
            },
            {
                'params': first_conv_bias,
                'lr_mult': 2,
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
            # for fc
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

    def forward(self, x):
        o = self.base_model(x)
        o = self.avgpool(o)
        o = self.dropout(o)
        o = self.fc(o.view(o.size(0), -1))
        if self.is_shift and self.temporal_pool:
            o = o.view((-1, self.num_segments // 2) + o.size()[1:])
        else:
            o = o.view((-1, self.num_segments) + o.size()[1:])
        output = self.consensus(o)
        return output.squeeze(1)


def create_model(num_class: int = 2,
                 num_segments: int = 8,
                 base_model: str = 'resnet50',
                 checkpoint: str = None,
                 device: str = None,
                 fc_lr5: bool = True,
                 is_shift: bool = True,
                 shift_div: int = 8,
                 shift_place: str = 'blockres',
                 consensus_type: str = 'avg',
                 img_feature_dim: int = 256,
                 non_local: bool = False,
                 **kwargs) -> nn.Module:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert consensus_type in ['avg', 'identity']
    assert shift_place in ['blockres', 'block']

    model = TSM(num_class=num_class,
                num_segments=num_segments,
                base_model=base_model,
                consensus_type=consensus_type,
                img_feature_dim=img_feature_dim,
                is_shift=is_shift,
                shift_div=shift_div,
                shift_place=shift_place,
                fc_lr5=fc_lr5,
                non_local=non_local)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt['state_dict']
        fc_layer_weight = list(state_dict.keys())[-2]
        fc_layer_bias = list(state_dict.keys())[-1]
        dim_feature = state_dict[fc_layer_weight].shape
        if dim_feature[0] == num_class:
            state_dict['module.fc.weight'] = state_dict[fc_layer_weight]
            state_dict['module.fc.bias'] = state_dict[fc_layer_bias]

        del state_dict[fc_layer_weight]
        del state_dict[fc_layer_bias]
        base_dict = OrderedDict(
            ('.'.join(k.split('.')[1:]), v) for k, v in state_dict.items())
        # replace_dict = {
        #     'base_model.classifier.weight': 'new_fc.weight',
        #     'base_model.classifier.bias': 'new_fc.bias',
        # }
        # for k, v in replace_dict.items():
        #     if k in base_dict:
        #         base_dict[v] = base_dict.pop(k)

        model.load_state_dict(base_dict, strict=False)

    model.to(device)
    return model


if __name__ == '__main__':
    num_class = 2
    num_seg = 8
    model = create_model(num_class, num_seg)
    print(model)

    model.eval()
    x = torch.randn(2 * 8, 3, 224, 224)
    y = model(x.cuda())
    print(y.size())

    model.train()
    x = torch.randn(2 * 8, 3, 224, 224)
    y = model(x.cuda())
    print(y)

    # checkpoint
    ckpt_path = 'checkpoints/finetune/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth'
    pretrained = create_model(2, 8, 'resnet50', checkpoint=ckpt_path)
    # print(pretrained)

    state_dict = torch.load(ckpt_path).get('state_dict')
    base_dict = OrderedDict(
        ('.'.join(k.split('.')[1:]), v) for k, v in state_dict.items())

    # check weights
    for k, v in pretrained.state_dict().items():
        if k in base_dict:
            assert torch.allclose(v, base_dict[k]), f"{k} not equal"
        else:
            print(k, v.shape, f"{k} is not in base_dict")
