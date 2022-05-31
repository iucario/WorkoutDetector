# Check Pytorch installation
from mmcv.parallel import MMDataParallel
from mmaction.datasets import build_dataloader
from mmaction.apis import single_gpu_test
import mmcv
from mmaction.apis import train_model
from mmaction.models import build_model
from mmaction.datasets import build_dataset
from mmcv.runner import set_random_seed
import os.path as osp
from mmcv import Config
import warnings
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmaction
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMAction2 installation
print(mmaction.__version__)

# Check MMCV installation
print(get_compiling_cuda_version())
print(get_compiler_version())

warnings.filterwarnings('ignore')

classes = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack',
    'situp', 'push_up', 'battle_rope', 'exercising_arm', 'lunge',
    'mountain_climber']
BASE_DIR = '/home/umi/projects/WorkoutDetector/'
config = osp.join(BASE_DIR, 'mmaction2/configs/recognition/tsm/tsm_my_config.py')
checkpoint = 'https://download.openmmlab.com/mmaction/recognition/'\
    'tsm/tsm_r50_1x1x16_50e_sthv2_rgb/'\
    'tsm_r50_256h_1x1x16_50e_sthv2_rgb_20210331-0a45549c.pth'

cfg = Config.fromfile(config)

DATASET = 'data/Workouts/rawframes/'
NUM_CLASSES = len(classes)
# Modify dataset type and path
cfg.dataset_type = 'RawframeDataset'
cfg.data_root = DATASET
cfg.data_root_val = DATASET
cfg.ann_file_train = DATASET + 'train.txt'
cfg.ann_file_val = DATASET + 'val.txt'
cfg.ann_file_test = DATASET + 'test.txt'

cfg.data.test.ann_file = cfg.ann_file_test
cfg.data.test.data_prefix = cfg.data_root

cfg.data.train.ann_file = cfg.ann_file_train
cfg.data.train.data_prefix = cfg.data_root

cfg.data.val.ann_file = cfg.ann_file_val
cfg.data.val.data_prefix = cfg.data_root_val

cfg.setdefault('omnisource', False)

cfg.model.cls_head.num_classes = NUM_CLASSES

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.data.videos_per_gpu = max(1, cfg.data.videos_per_gpu // 8)
cfg.optimizer.lr = cfg.optimizer.lr / 8 / 16
cfg.total_epochs = 5
cfg.load_from = checkpoint
# cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')
cfg.checkpoint_config.interval = 5

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Save the best
cfg.evaluation.save_best = 'auto'

cfg.log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook',
        #   init_kwargs=dict(project='RepCount-cleaned',config={**cfg})),
    ])

print(cfg.pretty_text)

# train


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=cfg.get(
    'train_cfg'), test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

## train model
train_model(model, datasets, cfg, distributed=False, validate=True)


# Build a test dataloader
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(
    dataset,
    videos_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader)

eval_config = cfg.evaluation
eval_config.pop('interval')
eval_res = dataset.evaluate(outputs, **eval_config)
