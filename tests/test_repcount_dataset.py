import os
import random

import pandas as pd
from WorkoutDetector.datasets import RepcountHelper
from WorkoutDetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH


def test_RepcountHelper():
    """Test RepcountHelper"""
    DATA_ROOT = os.path.join(PROJ_ROOT, 'data/RepCount')
    helper = RepcountHelper(DATA_ROOT, REPCOUNT_ANNO_PATH)
    SPLITS = ['train', 'val', 'test']
    ACTIONS = [
        'situp', 'push_up', 'pull_up', 'bench_pressing', 'jump_jack', 'squat',
        'front_raise'
    ]

    # Test get_rep_data()
    for _ in range(10):
        rand_action = random.choice(ACTIONS)
        rand_split = random.choice(SPLITS)
        train = helper.get_rep_data(split=['train'], action=[rand_action])
        val = helper.get_rep_data(split=['val'], action=[rand_action])
        test = helper.get_rep_data(split=['test'], action=[rand_action])
        all_ = helper.get_rep_data(split=SPLITS, action=[rand_action])
        all_action = helper.get_rep_data(split=SPLITS, action=ACTIONS)
        rand_split_all = helper.get_rep_data(split=[rand_split], action=ACTIONS)
        split_num = 0
        for action in ACTIONS:
            split_num += len(helper.get_rep_data(split=[rand_split], action=[action]))

        assert len(rand_split_all) == split_num
        assert len(train) + len(val) + len(test) == len(all_)

    TRAIN_TOTAL = 602
    VAL_TOTAL = 110
    TEST_TOTAL = 115
    assert len(helper.get_rep_data(split=['train'], action=ACTIONS)) == TRAIN_TOTAL
    assert len(helper.get_rep_data(split=['val'], action=ACTIONS)) == VAL_TOTAL
    assert len(helper.get_rep_data(split=['test'], action=ACTIONS)) == TEST_TOTAL
