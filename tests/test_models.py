from collections import OrderedDict
import random
import sys
import torch
from torch.utils.data import DataLoader
from workoutdetector.models.tsm import create_model
from workoutdetector.datasets import DebugDataset, Pipeline
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import optim
from einops import rearrange
import pandas as pd
import os
from torchvision.io import read_video


class Test_TSM:

    model = create_model(4, 8, 'resnet18', checkpoint=None, device='cuda')
    model.eval()
    ckpt_path = 'checkpoints/finetune/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth'
    k400_path = 'checkpoints/finetune/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth'

    def test_train(self):
        model = self.model
        i = torch.randn(4 * 8, 3, 224, 224)
        y = model(i.cuda())
        assert y.shape == (4, 4), y.shape

        dataset = DebugDataset(num_class=4, num_segments=8, size=100)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        EPOCHS = 10
        loss_fn = CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        model.cuda()
        model.train()
        for _ in range(EPOCHS):
            for x, y in loader:
                x = rearrange(x, 'b t c h w -> (b t) c h w')
                assert x.shape == (2 * 8, 3, 224, 224)
                y_pred = model(x.cuda())
                loss = loss_fn(y_pred.cpu(), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(loss.item(), y_pred.argmax(dim=1))

        model.eval()
        correct = 0
        for x, y in loader:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            y_pred = model(x.cuda())
            print(y_pred.argmax(dim=1), y)
            correct += (y_pred.cpu().argmax(dim=1) == y).sum().item()

        acc = correct / len(loader.dataset)
        assert acc > 0.5, f"Accuracy {acc} is too low"

    def test_finetune(self):
        num_class = 2
        batch = 4
        pretrained = create_model(num_class,
                                  8,
                                  'resnet50',
                                  checkpoint=self.ckpt_path,
                                  device='cpu')
        pretrained.eval()
        x = torch.randn(batch * 8, 3, 224, 224)
        y = pretrained(x)
        assert y.shape == (batch, num_class), \
            f"y.shape = {y.shape}. Expected {(batch, num_class)}"

        # check weights
        state_dict = torch.load(self.ckpt_path,
                                map_location=torch.device('cpu')).get('state_dict')
        base_dict = OrderedDict(
            ('.'.join(k.split('.')[1:]), v) for k, v in state_dict.items())
        for k, v in pretrained.state_dict().items():
            if k in base_dict:
                assert torch.allclose(v, base_dict[k]), f"{k} not equal"
            else:
                sys.stderr.write(k, v.shape, f"{k} is not in base_dict\n")

    @torch.no_grad()
    def test_k400(self):
        """Test accuracy of trained model on Kinetics400 subset Countix"""

        num_samples = 50
        model = create_model(400, 8, 'resnet50', checkpoint=self.k400_path)
        model.eval()
        label_df = pd.read_csv('datasets/kinetics400/kinetics_400_labels.csv')
        data_root = 'data/Countix/videos/train'
        data_df = pd.read_csv('datasets/Countix/countix_train.csv')
        video_list = os.listdir(data_root)
        video_ids = random.sample(video_list, num_samples)
        P = Pipeline()
        acc = 0
        for video_id in video_ids:
            gt_label = data_df.loc[data_df['video_id'] == video_id.split('.')[0],
                                   'class'].values[0]
            video = read_video(os.path.join(data_root, video_id))[0]
            inp = P.transform_read_video(video)
            out = model(inp.cuda()).cpu()
            top5 = torch.topk(out, 5)[1].tolist()[0]
            labels = [label_df.iloc[i, 1] for i in top5]
            #softmax
            label = labels[0]
            assert out.shape == (1, 400), out.shape
            if not label == gt_label:
                sys.stderr.write(f"Prediction: {label} != {gt_label}\n")
            acc += 1 if label == gt_label else 0
        assert acc / num_samples > 0.5, f"Accuracy {acc} is too low"
