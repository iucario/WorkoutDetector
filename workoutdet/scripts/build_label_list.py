import os
import os.path as osp
from os.path import join as osj

from workoutdet.utils import get_video_list


def build_with_start(data_root: str, anno_file: str, dst_dir: str) -> None:
    """Creates label files for repetition video classification.
    Example line: `/app/data/RepCount/rawframes/test/train3344 2 12 0`

    Args:
        data_root (str): Directory to RepCount dataset. Same as in RepCountDataset.
        anno_file (str): Path to annotation file.
        dst_dir (str): Path to dir where txt files will be created.

    Note:
        For each action and `all-{split}.txt` file is created.
        Lines in the label files are in the format:
            `{video_path} {start} {length} {label}`
    
    Example::

        >>> anno_file = 'datasets/RepCount/annotation.csv'
        >>> data_root = os.path.expandeuser('data')
        >>> dst_dir = os.path.join(data_root, 'Binary')
        >>> build_with_start(data_root, anno_file, dst_dir)

        project root: /work
        ===> Done! Label files are created in
        /home/user/data/Binary
        ===> First line of all-train.txt:
        RepCount/rawframes/train/train951 7 34 10
        # Creates `{data_root}/Binary/all-train.txt` and `{data_root}/Binary/all-val.txt` and 
        # `all-test.txt`, `situp-train.txt`, etc.
    """

    # pop bench_pressing because there are too many errors in the annotation
    CLASSES = ['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise']
    MAX_REPS = 2  # limit the number of reps per video, because some videos have too many repetitions
    os.makedirs(dst_dir, exist_ok=True)

    for action in CLASSES:

        for split in ['train', 'val', 'test']:
            video_list = get_video_list(anno_file, split, action, max_reps=MAX_REPS)
            # build for single action
            with open(os.path.join(dst_dir, f'{action}-{split}.txt'), 'w') as f:
                for v in video_list:
                    f.write(
                        f'{v["video_path"]} {v["start"]} {v["length"]} {v["label"]}\n')

    # build for all actions
    for split in ['train', 'val', 'test']:
        video_list = get_video_list(anno_file, split, action=None, max_reps=MAX_REPS)
        with open(os.path.join(dst_dir, f'all-{split}.txt'), 'w') as f:
            for v in video_list:
                action = v['class']
                if action == 'bench_pressing':
                    continue
                action_idx = CLASSES.index(action)
                label = v['label'] + 2 * action_idx
                f.write(f'{v["video_path"]} {v["start"]} {v["length"]} {label}\n')

    print(f'===> Done! Label files are created in\n{dst_dir}')
    first_line = open(osp.join(dst_dir, 'all-train.txt')).readline()
    print(f'===> First line of all-train.txt:\n{first_line}')


if __name__ == '__main__':
    anno_file = 'datasets/RepCount/annotation.csv'
    data_root = osp.expanduser('~/data')
    dst_dir = os.path.join(data_root, 'Binary')
    build_with_start(data_root, anno_file, dst_dir)
