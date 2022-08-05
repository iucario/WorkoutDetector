import json

import torch
import yaml
from fvcore.common.config import CfgNode
from tqdm import tqdm
from workoutdet.trainer import LitModel

classes_1 = [
    'situp 1', 'situp 2', 'push_up 1', 'push_up 2', 'pull_up 1', 'pull_up 2',
    'jump_jack 1', 'jump_jack 2', 'squat 1', 'squat 2', 'front_raise 1', 'front_raise 2',
    'all'
]
classes_2 = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise', 'all_6']


def create_dataloader(cfg: CfgNode, dataset: str) -> torch.utils.data.DataLoader:
    """Create dataloader for the given dataset."""
    assert dataset in ('train', 'val', 'test')
    config = cfg.clone().data
    config.test.anno = config.get(dataset).anno  # use test transform
    ds = build_dataset(config, dataset)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    return loader


def calculate_acc(class_acc_1: list, class_total_1: list, class_acc_2: list,
                  class_total_2: list) -> dict:
    result = dict()
    # 12 class acc
    l1 = [x / y for x, y in zip(class_acc_1, class_total_1)]
    l1.append(sum(class_acc_1) / sum(class_total_1))
    result = dict(zip(classes_1, l1), class_acc=class_acc_1, class_total=class_total_1)
    # 6 class acc
    l2 = [x / y for x, y in zip(class_acc_2, class_total_2)]
    l2.append(sum(class_acc_2) / sum(class_total_2))
    result.update(
        dict(zip(classes_2, l2), class_acc_2=class_acc_2, class_total_2=class_total_2))
    return result


@torch.no_grad()
def eval_torch_12(checkpoint: str, output_file: str) -> None:
    cfg = CfgNode(yaml.safe_load(open('workoutdet/repcount.yaml')))
    print(cfg.data)

    device = 'cuda:0'
    model = LitModel.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    print(model(torch.randn(8, 3, 224, 224).to(device)))
    # trainer = Trainer(devices=0, enable_progress_bar=False)
    result = dict()
    for split in ('train', 'val', 'test'):
        class_acc_1 = [0] * 12
        class_total_1 = [0] * 12
        class_acc_2 = [0] * 6
        class_total_2 = [0] * 6
        loader = create_dataloader(cfg, split)
        for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
            pred = model(x.to(device)).argmax(dim=1).cpu()
            # 12 class correct
            if pred == y:
                class_acc_1[y] += 1
            class_total_1[y] += 1

            # 6 class correct
            if pred // 2 == y // 2:
                class_acc_2[y // 2] += 1
            class_total_2[y // 2] += 1

        d = calculate_acc(class_acc_1, class_total_1, class_acc_2, class_total_2)
        result[split] = d
    print(result)
    with open(output_file, 'a') as f:
        json.dump(result, f)


if __name__ == '__main__':
    eval_torch_12(
        'checkpoints/repcount-12/best-val-acc=0.923-epoch=10-20220720-151025.ckpt',
        'relabel_12_situp_tsm.json')
