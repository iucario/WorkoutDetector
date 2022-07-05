# Modified from https://github.com/facebookresearch/slowfast
"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# PyTorch Lightning trainer options.
# ---------------------------------------------------------------------------- #
_C.trainer = CfgNode()

# See pytorch lightning documentation for details.
_C.trainer.default_root_dir = "exp"
_C.trainer.max_epochs = 30
_C.trainer.early_stopping = False
_C.trainer.enable_checkpointing = None
_C.trainer.num_nodes = 1
_C.trainer.devices = 8
_C.trainer.gpus = "auto"
_C.trainer.log_gpu_memory = True
_C.trainer.enable_progress_bar = True
_C.trainer.overfit_batches = 0.0
_C.trainer.check_val_every_n_epoch = 1
_C.trainer.min_epochs = 1
_C.trainer.accelerator = "auto"
_C.trainer.sync_batchnorm = False
_C.trainer.precision = 32
_C.trainer.enable_model_summary = True
_C.trainer.weights_summary = "top"
_C.trainer.num_sanity_val_steps = 2
_C.trainer.resume_from_checkpoint = None
_C.trainer.benchmark = False
_C.trainer.deterministic = True
_C.trainer.auto_lr_find = False
_C.trainer.auto_scale_batch_size = None
_C.trainer.prepare_data_per_node = None
_C.trainer.patience = 10
_C.trainer.fast_dev_run = False

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.optimizer = CfgNode()

# Base learning rate.
_C.optimizer.lr = 0.1

# Optimization method.
_C.optimizer.method = "sgd"

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.lr_schedular.policy = "StepLR"

# Momentum.
_C.optimizer.momentum = 0.9
_C.optimizer.eps = 1e-8

# L2 regularization.
_C.optimizer.weight_decay = 1e-4

# Steps for 'steps_' policies (in epochs).
_C.lr_schedular.step = []

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.data = CfgNode()

# Whether to crop person for test data transform.
_C.aug.person_crop = True

_C.data.dataset_type = "FrameDataset"
_C.data.data_root = "/data/RepCount/rawframes"
_C.data.num_segments = 8
_C.data.filename_tmpl = 'img_{:05}.jpg'
_C.data.batch_size = 2

# If columns == 4, will be read as `{video} {start} {total_frames} {label}`
_C.data.anno_col = 4

_C.data.train.anno = "/data/Binary/all-test.txt"
_C.data.train.data_prefix = None
_C.data.train.anno = "/data/Binary/all-val.txt"
_C.data.train.data_prefix = None
_C.data.train.anno = "/data/Binary/all-test.txt"
_C.data.train.data_prefix = None
_C.data.num_workers = 4

# ---------------------------------------------------------------------------- #
# Logging options.
# ---------------------------------------------------------------------------- #
_C.log = CfgNode()
_C.log.output_dir = "exp"
_C.log.name = "experiment"
_C.log.log_every_n_steps = 20
_C.log.tensorboard.enable = True
_C.log.wandb.enable = False
_C.log.wandb.offline = False
_C.log.wandb.project = "experiment"

# ---------------------------------------------------------------------------- #
# Other options.
# ---------------------------------------------------------------------------- #
_C.seed = 0
_C.train = True
_C.checkpoint = None

# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.mixup = CfgNode()

# Whether to use mixup.
_C.mixup.enable = False

# Mixup alpha.
_C.mixup.alpha = 0.8

# Cutmix alpha.
_C.mixup.cutmix_alpha = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.mixup.rob = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.mixup.switch_prob = 0.5

# Label smoothing.
_C.mixup.label_smooth_value = 0.1

# ---------------------------------------------------------------------------- #
# Time shift module options.
# ---------------------------------------------------------------------------- #
_C.model = CfgNode()
_C.model.num_class = 11
_C.model.num_segments = 8
_C.model.base_model = "resnet50"
_C.model.consensus_type = "avg"
_C.model.img_feature_dim = 256
_C.model.is_shift = True
_C.model.shift_div = 8
_C.model.shift_place = "blockres"
_C.model.fc_lr5 = False
_C.model.temporal_pool = False
_C.model.non_local = False


def assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
