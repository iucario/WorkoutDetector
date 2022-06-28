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
from .build import MODEL_REGISTRY


class TemporalShift(nn.Module):

    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
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


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    # print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                # print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                # print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                # print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1,
                                                        n_segment=this_segment,
                                                        n_div=n_div)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
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


@MODEL_REGISTRY.register()
class TSM(nn.Module):
    """TSN with temporal shift module
    Input shape: (batch_size*num_segments, channel, height, width)
    
    Args:
        num_class (int): number of classes
        num_segments (int): number of segments, default 8
        modality (str): modality of input data, default 'RGB'
        base_model (str): base model name, default 'resnet50'
        consensus_type (str): 'avg' or 'identity', default 'avg'.
        dropout (float): dropout rate, default 0.5
        img_feature_dim (int): I don't think it's used.
        crop_num (int): number of crops from one image, default 1
        partial_bn (bool): use partial bn or not, default True
        print_spec (bool): print out the spec of the model, default False
        is_shift (bool): use temporal shift module or not, default True
        shift_div (int): the number of folds of temporal shift, default 8
        shift_place (str): 'block' or 'blockres', default 'blockres'
        fc_lr5 (bool): I don't know what it is.
        temporal_pool (bool): use temporal pooling or not, default False
        non_local (bool): use non-local module or not, default False
    """

    def __init__(self,
                 num_class,
                 num_segments=8,
                 modality='RGB',
                 base_model='resnet50',
                 consensus_type='avg',
                 before_softmax=True,
                 dropout=0.5,
                 img_feature_dim=256,
                 crop_num=1,
                 partial_bn=True,
                 print_spec=False,
                 is_shift=True,
                 shift_div=8,
                 shift_place='blockres',
                 fc_lr5=False,
                 temporal_pool=False,
                 non_local=False):
        super(TSM, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = 1
        if print_spec:
            print(f"Initializing TSN with base model: {base_model}.",
                  "TSN Configurations:",
                  f"input_modality:     {self.modality}",
                  f"num_segments:       {self.num_segments}",
                  f"new_length:         {self.new_length}",
                  f"consensus_module:   {consensus_type}",
                  f"dropout_ratio:      {self.dropout}",
                  f"img_feature_dim:    {self.img_feature_dim}",
                  sep='\n')

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class: int) -> int:
        feature_dim = getattr(self.base_model,
                              self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name,
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(
                getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

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

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

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
                        "New atomic module type: {}. Need to give it a learning policy".
                        format(type(m)))

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

    def head(self, base_out):
        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) +
                                         base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def forward(self, input_x, no_reshape=False):
        base_out = self.forward_features(input_x, no_reshape)
        return self.head(base_out)

    def forward_features(self, input_x, reshape=False):
        if reshape:
            sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length
            base_out = self.base_model(
                input_x.view((-1, sample_len) + input_x.size()[-2:]))
        else:
            base_out = self.base_model(input_x)
        return base_out

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


def create_model(num_class: int = 2,
                 num_seg: int = 8,
                 base_model: str = 'resnet50',
                 pretrained: bool = False,
                 ckpt: str = None,
                 device: str = None) -> nn.Module:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TSM(num_class=num_class,
                num_segments=num_seg,
                base_model=base_model,
                consensus_type='avg',
                img_feature_dim=512,
                is_shift=True,
                shift_div=8,
                shift_place='blockres',
                fc_lr5=False,
                temporal_pool=False,
                non_local=False)
    if pretrained:
        checkpoint = torch.load(ckpt, map_location=device)
        state_dict = checkpoint['state_dict']
        dim_feature = state_dict['module.new_fc.weight'].shape
        if dim_feature[0] != num_class:
            state_dict['module.new_fc.weight'] = torch.zeros(num_class, dim_feature[1]).cuda()
            state_dict['module.new_fc.bias'] = torch.zeros(num_class).cuda()
        base_dict = OrderedDict(
            ('.'.join(k.split('.')[1:]), v) for k, v in state_dict.items())
        # replace_dict = {
        #     'base_model.classifier.weight': 'new_fc.weight',
        #     'base_model.classifier.bias': 'new_fc.bias',
        # }
        # for k, v in replace_dict.items():
        #     if k in base_dict:
        #         base_dict[v] = base_dict.pop(k)

        model.load_state_dict(base_dict)

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

    from workoutdetector.datasets import DebugDataset
    dataset = DebugDataset(2, 8, size=100)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    model.cuda()

    for _ in range(20):
        for x, y in loader:
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(-1, 3, 224, 224)
            y_pred = model(x.cuda())
            loss = loss_fn(y_pred.cpu(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())