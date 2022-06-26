import torch
from torch.utils.data import DataLoader
from workoutdetector.models import TSM
from workoutdetector.datasets import DebugDataset
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from einops import rearrange

def test_TSM():
    model = TSM(2, 8, base_model='resnet18', img_feature_dim=512)
    model.eval()
    i = torch.randn(4 * 8, 3, 224, 224)
    y = model(i)
    assert y.shape == (4, 2), y.shape

    dataset = DebugDataset(2, 8, 20)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    EPOCHS = 3
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.cuda()
    model.train()
    for _ in range(EPOCHS):
        for x, y in loader:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            assert x.shape == (2*8, 3, 224, 224)
            y_pred = model(x.cuda())
            loss = loss_fn(y_pred.cpu(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss, y_pred.argmax(dim=1))

    model.eval()
    correct = 0
    for x, y in loader:
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        y_pred = model(x.cuda())
        correct += (y_pred.cpu().argmax(dim=1) == y).sum().item()

    acc = correct / len(loader.dataset)
    assert acc > 0.5, f"Accuracy {acc} is too low"