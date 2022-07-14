import json

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from fvcore.common.config import CfgNode
from tqdm import tqdm
from workoutdetector.datasets import build_dataset, build_test_transform
from workoutdetector.trainer import LitModel
from pytorch_lightning import LightningModule


def eval_onnx(output_file: str) -> None:
    """Evaluate the classification model. And save results to JSON file."""

    cfg = CfgNode(yaml.safe_load(open('workoutdetector/configs/defaults.yaml')))
    cfg.merge_from_file('workoutdetector/configs/repcount_12_tsm.yaml')
    print(cfg.data)
    train_ds = build_dataset(cfg.data, 'train')
    val_ds = build_dataset(cfg.data, 'val')
    test_ds = build_dataset(cfg.data, 'test')
    print(len(train_ds), len(val_ds), len(test_ds))
    print(train_ds[0][0].shape)

    checkpoint = 'checkpoints/repcount-12/rep_12_20220705_220720.onnx'
    ort_session = onnxruntime.InferenceSession(checkpoint,
                                               providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    classes_1 = [
        'situp 1', 'situp 2', 'push_up 1', 'push_up 2', 'pull_up 1', 'pull_up 2',
        'jump_jack 1', 'jump_jack 2', 'squat 1', 'squat 2', 'front_raise 1',
        'front_raise 2', 'all'
    ]
    classes_2 = [
        'situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise', 'all_6'
    ]
    result = dict()

    for split in ('train', 'val', 'test'):
        class_acc_1 = [0] * 12
        class_total_1 = [0] * 12
        class_acc_2 = [0] * 6
        class_total_2 = [0] * 6
        config = cfg.clone().data
        config.test.anno = config.get(split).anno  # use test transform
        ds = build_dataset(config, 'test')
        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0)
        for i, (x, y) in tqdm(enumerate(loader), total=len(ds)):
            assert x.shape == (1, 8, 3, 224, 224)
            out = ort_session.run(None, {input_name: x.numpy()})
            pred = out[0].argmax()

            # 12 class correct
            if pred == y[0]:
                class_acc_1[y[0]] += 1
            class_total_1[y[0]] += 1

            # 6 class correct
            if pred // 2 == y[0] // 2:
                class_acc_2[y[0] // 2] += 1
            class_total_2[y[0] // 2] += 1

        # 12 class acc
        l1 = [x / y for x, y in zip(class_acc_1, class_total_1)]
        l1.append(sum(class_acc_1) / sum(class_total_1))
        result[split] = dict(zip(classes_1, l1),
                             class_acc=class_acc_1,
                             class_total=class_total_1)
        # 6 class acc
        l2 = [x / y for x, y in zip(class_acc_2, class_total_2)]
        l2.append(sum(class_acc_2) / sum(class_total_2))
        result[split].update(
            dict(zip(classes_2, l2), class_acc_2=class_acc_2,
                 class_total_2=class_total_2))
    print(result)
    with open(output_file, 'a') as f:
        json.dump(result, f)


@torch.no_grad()
def eval_torch(output_file: str) -> None:
    cfg = CfgNode(yaml.safe_load(open('workoutdetector/configs/defaults.yaml')))
    cfg.merge_from_file('workoutdetector/configs/tsm.yaml')
    print(cfg.data)

    device = 'cuda:0'
    checkpoint = 'checkpoints/best-val-acc=0.975-epoch=15-20220711-220219.ckpt'
    model = LitModel.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    print(model(torch.randn(8, 3, 224, 224).to(device)))
    # trainer = Trainer(devices=0, enable_progress_bar=False)

    classes = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise', 'all']
    result = dict()
    outfile = open(output_file, 'a')
    sizes = {'train': 1075, 'val': 195, 'test': 231}

    for split in ('train', 'val', 'test'):
        class_acc = [0] * 6
        class_total = [0] * 6
        config = cfg.data.clone()
        config.test.anno = config.get(split).anno  # use test transform
        ds = build_dataset(config, 'test')
        assert len(
            ds) == sizes[split], f'length of {split} = {len(ds)} should be {sizes[split]}'

        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0)
        for i, (x, y) in tqdm(enumerate(loader), total=len(ds)):
            x = x.squeeze(0)
            assert x.shape == (8, 3, 224, 224)
            out = model(x.to(device))
            assert out.shape == (1, 6)
            pred = out[0].argmax()
            if pred == y[0]:
                class_acc[y[0]] += 1
            class_total[y[0]] += 1
        l = [x / y for x, y in zip(class_acc, class_total)]
        l.append(sum(class_acc) / sum(class_total))
        print(f'class correct: {class_acc}, class total: {class_total}')
        assert len(l) == len(classes) == 7
        result[split] = dict(zip(classes, l),
                             class_acc=class_acc,
                             class_total=class_total)
        json.dump(result, outfile)
    outfile.close()
    print(result)


if __name__ == '__main__':
    eval_onnx('repcount_12_tsm_acc.json')
    # eval_torch('repcount_6_tsm_acc.json')
