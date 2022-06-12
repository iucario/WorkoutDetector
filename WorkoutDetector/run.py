import torch
import torch.nn as nn

import mmaction
from mmaction.models.backbones import ResNetTSM
from mmaction.models.heads import TSMHead

backbone = ResNetTSM(depth=18)

dummpy_input = torch.randn(24, 3, 224, 224)
print(backbone(dummpy_input).shape)

head = TSMHead(num_classes=2, in_channels=512, num_segments=8)
print(head)

o = head(backbone(dummpy_input), num_segs=1)
print(o.shape) # [batch_size/num_segs, num_classes]