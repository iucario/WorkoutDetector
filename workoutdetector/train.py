import os
from workoutdetector.settings import PROJ_ROOT
from mmcv.parallel import MMDistributedDataParallel
from mmaction.datasets import build_dataloader
from mmaction.apis import single_gpu_test
import mmcv
from mmaction.apis import train_model
from mmaction.models import build_model
from mmaction.datasets import build_dataset
from mmcv.runner import get_dist_info, init_dist, set_random_seed
import os.path as osp
from mmcv import Config
import warnings
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmaction
import torch
import torch.distributed as dist
print(torch.__version__, torch.cuda.is_available())

# Check MMAction2 installation
print(mmaction.__version__)

# Check MMCV installation
print(get_compiling_cuda_version())
print(get_compiler_version())

warnings.filterwarnings('ignore')

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '8'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

CLASSES = [
    'front_raise', 'pull_up', 'squat', 'bench_pressing', 'jumping_jack', 'situp',
    'push_up', 'battle_rope', 'exercising_arm', 'lunge', 'mountain_climber'
]
config = osp.join(PROJ_ROOT, 'workoutdetector/configs/tsm_action_recogition_sthv2.py')

cfg = Config.fromfile(config)
set_random_seed(0, deterministic=True)
init_dist('pytorch', **cfg.dist_params)
_, world_size = get_dist_info()
cfg.gpu_ids = range(world_size)

cfg.log_config = dict(interval=50,
                      hooks=[
                          dict(type='TextLoggerHook'),
                          dict(type='TensorboardLoggerHook'),
                        #   dict(type='WandbLoggerHook',
                        #        init_kwargs=dict(project='RepCount+Countix',
                        #                         config={**cfg}))
                      ])

print(cfg.pretty_text)

# train

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model,
                    train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

## train model
train_model(model, datasets, cfg, distributed=False, validate=True)

# Build a test dataloader
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(dataset,
                               videos_per_gpu=6,
                               workers_per_gpu=2,
                               dist=True,
                               shuffle=False)
model = MMDistributedDataParallel(model, device_ids=list(range(torch.cuda.device_count())))
outputs = single_gpu_test(model, data_loader)

eval_config = cfg.evaluation
eval_config.pop('interval')
eval_res = dataset.evaluate(outputs, **eval_config)
