import torch
import PIL.Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt


class DebugDataset(torch.utils.data.Dataset):
    """Generates pure color videos or images for debugging
    
    Args:
        num_class (int): number of classes
        num_segments (int): number of segments for video dataset
        size (int): number of videos or images to generate
    
    Returns:
        Tuple[Tensor, int], Tensor of shape (num_segments, 3, 224, 224), label
    """

    def __init__(self,
                 num_class: int = 2,
                 num_segments: int = 8,
                 size: int = 100) -> None:
        self.num_class = num_class
        self.num_segments = num_segments
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx) -> tuple:
        label = idx % self.num_class
        x = torch.full((self.num_segments, 3, 224, 224),
                       1 / (label + 1),
                       dtype=torch.float32)
        return x, label


if __name__ == '__main__':

    dataset = DebugDataset(num_class=10, size=20)
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(x.shape, y)