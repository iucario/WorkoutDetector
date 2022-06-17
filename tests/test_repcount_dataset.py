import os
import random

import pandas as pd
from WorkoutDetector.datasets import RepcountHelper
from WorkoutDetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

SPLITS = ['train', 'val', 'test']
ACTIONS = [
    'situp', 'push_up', 'pull_up', 'bench_pressing', 'jump_jack', 'squat', 'front_raise'
]
DATA_ROOT = os.path.join(PROJ_ROOT, 'data/RepCount')


class TestRepcountHelper:
    """Test RepcountHelper"""
    helper = RepcountHelper(DATA_ROOT, REPCOUNT_ANNO_PATH)

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
        all_ = self.helper.get_rep_data(split=SPLITS, action=[rand_action])
        all_action = self.helper.get_rep_data(split=SPLITS, action=ACTIONS)
        rand_split_all = self.helper.get_rep_data(split=[rand_split], action=ACTIONS)
        split_num = 0
        for action in ACTIONS:
            split_num += len(
                self.helper.get_rep_data(split=[rand_split], action=[action]))

        assert len(rand_split_all) == split_num
        assert len(train) + len(val) + len(test) == len(all_)

        TRAIN_TOTAL = 602
        VAL_TOTAL = 110
        TEST_TOTAL = 115
        assert len(self.helper.get_rep_data(split=['train'],
                                            action=ACTIONS)) == TRAIN_TOTAL
        assert len(self.helper.get_rep_data(split=['val'], action=ACTIONS)) == VAL_TOTAL
        assert len(self.helper.get_rep_data(split=['test'], action=ACTIONS)) == TEST_TOTAL
    

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
