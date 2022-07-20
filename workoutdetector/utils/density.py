"""
0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 

Guassian density function applied to repetition instances.
    \mu = (start_time + end_time) / 2
    \sigma = end_time - start_time

Inputs:
    Continuous 8 frames, [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]

Outputs:
    A float value between 0 and 1.

Model:
    A action recognizer + FC layer.

Loss function:
    Mean squared error. Compares with the density function at i+4.
    Formally, the loss is: `nn.MSE(model(i), density(i+4))`

Inference:
    Sum the predicted density at each frame.

Why this method:
    Reduces a loooot of engineering. And is suitable for online inference.
"""

from typing import Any, Callable, List
import torch
from torch import nn, Tensor
from scipy.stats import norm


def density_fn(start: int, end: int) -> Any:
    mid = (end + start) / 2
    sigma = (end - start) / 6
    dist = norm(loc=mid, scale=sigma)
    return dist

def create_label(reps: List[int]) -> Tensor:
    """Create normalized label.

    Args:
        reps: A list of repetition starts and ends. `[start_1, end_1, start_2, end_2, ...]`
    Returns:
        Tensor, shape [1, len(reps)]
    """
    labels = [0] * len(reps)
    for s, e in zip(reps[::2], reps[1::2]):
        dist = density_fn(s, e)
        for i in range(s, e + 1):
            labels[i] = dist.pdf(i)
    return torch.tensor(labels)


def main():
    pass