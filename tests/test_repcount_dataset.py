from WorkoutDetector.datasets import RepcountHelper
from WorkoutDetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
import pandas as pd
import os
import random

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
        num_split = 0
        for action in ACTIONS:
            num_split += len(helper.get_rep_data(split=rand_split, action=[action]))

        assert len(train) + len(all_) + len(all_action) == num_split
        assert len(train) + len(val) + len(test) == len(all_)



