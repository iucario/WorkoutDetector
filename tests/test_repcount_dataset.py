import os
import random
import torch
import pandas as pd
from workoutdetector.datasets import RepcountHelper, RepcountRecognitionDataset
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

SPLITS = ['train', 'val', 'test']
ACTIONS = [
    'situp', 'push_up', 'pull_up', 'bench_pressing', 'jump_jack', 'squat', 'front_raise'
]


class TestRepcountHelper:
    """Test RepcountHelper"""
    DATA_ROOT = os.path.join(PROJ_ROOT, 'data/RepCount')
    helper = RepcountHelper(DATA_ROOT, REPCOUNT_ANNO_PATH)
    all_ = helper.get_rep_data(split=SPLITS, action=ACTIONS)

    def test_data(self):
        name = 'stu4_57.mp4'
        class_ = 'situp'
        assert name in self.all_.keys(), f'{name} not in data'
        assert 'stu2_48.mp4' in self.all_.keys(), f'stu2_48.mp4 not in data'
        assert self.all_['stu2_48.mp4'].class_ == 'situp', f'stu2_48.mp4 mislabled class_'

    def test_get_rep_data(self):
        rand_action = random.choice(ACTIONS)
        rand_split = random.choice(SPLITS)

        random_dict = self.helper.get_rep_data(split=[rand_split], action=[rand_action])
        random_item = random.choice(list(random_dict.values()))
        assert os.path.exists(random_item.video_path)
        assert os.path.isdir(random_item.frames_path)

        train = self.helper.get_rep_data(split=['train'], action=[rand_action])
        val = self.helper.get_rep_data(split=['val'], action=[rand_action])
        test = self.helper.get_rep_data(split=['test'], action=[rand_action])

        all_action = self.helper.get_rep_data(split=SPLITS, action=ACTIONS)
        rand_split_all = self.helper.get_rep_data(split=[rand_split], action=ACTIONS)
        split_num = 0
        for action in ACTIONS:
            split_num += len(self.helper.get_rep_data(split=[rand_split],
                                                      action=[action]))

        assert len(rand_split_all) == split_num

        TRAIN_TOTAL = 602
        VAL_TOTAL = 110
        TEST_TOTAL = 115
        assert len(self.helper.get_rep_data(split=['train'],
                                            action=ACTIONS)) == TRAIN_TOTAL
        assert len(self.helper.get_rep_data(split=['val'], action=ACTIONS)) == VAL_TOTAL
        assert len(self.helper.get_rep_data(split=['test'], action=ACTIONS)) == TEST_TOTAL
        # test rep order: start < end
        reps = random_item.reps
        for start, end in zip(reps[::2], reps[1::2]):
            assert start < end, f'{random_item.video_name},' \
                    f'{reps[i]} should be less than {reps[i+1]}'
        # test rep order: end_i <= end_i+1
        for i in range(1, len(reps) // 2, 2):
            assert reps[i] <= reps[i+1], f'{random_item.video_name},' \
                    f'{reps[i]} should be no larger than {reps[i+1]}'

    def test_RepcountHelper_eval_count(self):
        for sp in SPLITS:
            for act in ACTIONS:
                data = self.helper.get_rep_data(split=[sp], action=[act])
                # randomly +1 or -1 rep to true rep data
                count_list = [x.count for x in data.values()]

                preds = dict()
                true_mae = 0.0
                true_obo = 1
                random_diffs = [random.choice([-1, 1]) for _ in range(len(count_list))]
                for i, (name, val) in enumerate(data.items()):
                    preds[name] = val.count + random_diffs[i]
                    if val.count > 0:
                        true_mae += abs(val.count - preds[name]) / val.count
                true_mae /= len(preds)
                # evaluate
                mae, obo, _ = self.helper.eval_count(preds, [sp], [act])
                assert mae == true_mae, f'sp={sp} act={act}, mae={mae}, true_mae={true_mae}'
                assert obo == true_obo, f'sp={sp} act={act}, obo={obo}, true_obo={true_obo}'


def test_RepcountRecognitionDataset():
    DATA_ROOT = os.path.join(PROJ_ROOT, 'data/RepCount')
    actions = ['push_up', 'situp', 'squat', 'jump_jack', 'pull_up']
    for split in ['train', 'val', 'test']:
        dataset = RepcountRecognitionDataset(DATA_ROOT,
                                             split=split,
                                             actions=actions,
                                             num_segments=8)
        for i in range(len(dataset)):
            x, y = dataset[i]
            assert x.shape[:2] == torch.Size([3, 8])
            assert 0 <= y < len(actions)
