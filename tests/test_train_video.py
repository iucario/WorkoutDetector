import torch
from workoutdetector.train_video import VideoModel


def test_VideoModel():
    model = VideoModel(model_name='x3d_s', model_num_class=12)
    x1 = torch.randn(2, 3, 8, 224, 224) # slow pathway
    x2 = torch.randn(2, 3, 32, 224, 224) # fast pathway
    x3 = torch.randn(2, 3, 13, 224, 224) # x3d_s
    y = model(x3)
    assert y.shape == torch.Size([2, 12])
