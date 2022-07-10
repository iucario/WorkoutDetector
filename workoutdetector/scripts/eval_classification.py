import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime
from fvcore.common.config import CfgNode
import yaml
from workoutdetector.datasets import build_dataset, build_test_transform
import json
from tqdm import tqdm

def main(output_file: str) -> None:
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
    onnx_model = onnx.load(checkpoint)
    ort_session = onnxruntime.InferenceSession(checkpoint,
                                               providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    classes = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise', 'all']
    class_acc = [0] * 12
    class_total = [0] * 12
    result = dict()

    for split in ('train', 'val', 'test'):
        ds = build_dataset(cfg.data, split)
        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0)
        for i, (x, y) in tqdm(enumerate(loader), len(ds)):
            assert x.shape == (1, 8, 3, 224, 224)
            pred = ort_session.run(None, {input_name: x.numpy()})
            if pred[0][0] == y[0]:
                class_acc[y[0]] += 1
        l = [x / y for x, y in zip(class_acc, class_total)]
        l.append(sum(class_acc) / sum(class_total))
        result[split] = dict(zip(classes, l))
    print(result)
    with open(output_file, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    out = 'repcount_12_tsm_acc.json'
    main(out)