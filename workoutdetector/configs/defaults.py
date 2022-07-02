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
_C.trainer.default_root_dir= "exp"
_C.trainer.max_epochs= 30
_C.trainer.early_stopping= False
_C.trainer.enable_checkpointing= None
_C.trainer.num_nodes= 1
_C.trainer.devices= 8
_C.trainer.gpus= "auto"
_C.trainer.log_gpu_memory= True
_C.trainer.enable_progress_bar= True
_C.trainer.overfit_batches= 0.0
_C.trainer.check_val_every_n_epoch= 1
_C.trainer.min_epochs= 1
_C.trainer.accelerator= "auto"
_C.trainer.sync_batchnorm= False
_C.trainer.precision= 32
_C.trainer.enable_model_summary= True
_C.trainer.weights_summary= "top"
_C.trainer.num_sanity_val_steps= 2
_C.trainer.resume_from_checkpoint= None
_C.trainer.benchmark= False
_C.trainer.deterministic= True
_C.trainer.auto_lr_find= False
_C.trainer.auto_scale_batch_size= None
_C.trainer.prepare_data_per_node= None
_C.trainer.patience= 10
_C.trainer.fast_dev_run= False


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
_C.AUG = CfgNode()


# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7


# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# Directory path of frames.
_C.AVA.FRAME_DIR = "/mnt/fair-flash3-east/ava_trainval_frames.img/"

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.AVA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "cv2"

# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]

# Config for layers' weights and activations visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS = CfgNode()

# If False, skip model visualization.
_C.TENSORBOARD.MODEL_VIS.ENABLE = False

# If False, skip visualizing model weights.
_C.TENSORBOARD.MODEL_VIS.MODEL_WEIGHTS = False

# If False, skip visualizing model activations.
_C.TENSORBOARD.MODEL_VIS.ACTIVATIONS = False

# If False, skip visualizing input videos.
_C.TENSORBOARD.MODEL_VIS.INPUT_VIDEO = False


# List of strings containing data about layer names and their indexing to
# visualize weights and activations for. The indexing is meant for
# choosing a subset of activations outputed by a layer for visualization.
# If indexing is not specified, visualize all activations outputed by the layer.
# For each string, layer name and indexing is separated by whitespaces.
# e.g.: [layer1 1,2;1,2, layer2, layer3 150,151;3,4]; this means for each array `arr`
# along the batch dimension in `layer1`, we take arr[[1, 2], [1, 2]]
_C.TENSORBOARD.MODEL_VIS.LAYER_LIST = []
# Top-k predictions to plot on videos
_C.TENSORBOARD.MODEL_VIS.TOPK_PREDS = 1
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.COLORMAP = "Pastel2"
# Config for visualization video inputs with Grad-CAM.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM = CfgNode()
# Whether to run visualization using Grad-CAM technique.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE = True
# CNN layers to use for Grad-CAM. The number of layers must be equal to
# number of pathway(s).
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST = []
# If True, visualize Grad-CAM using true labels for each instances.
# If False, use the highest predicted class.
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.USE_TRUE_LABEL = False
# Colormap to for text boxes and bounding boxes colors
_C.TENSORBOARD.MODEL_VIS.GRAD_CAM.COLORMAP = "viridis"

# Config for visualization for wrong prediction visualization.
# _C.TENSORBOARD.ENABLE must be True.
_C.TENSORBOARD.WRONG_PRED_VIS = CfgNode()
_C.TENSORBOARD.WRONG_PRED_VIS.ENABLE = False
# Folder tag to origanize model eval videos under.
_C.TENSORBOARD.WRONG_PRED_VIS.TAG = "Incorrectly classified videos."
# Subset of labels to visualize. Only wrong predictions with true labels
# within this subset is visualized.
_C.TENSORBOARD.WRONG_PRED_VIS.SUBSET_PATH = ""


# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CfgNode()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
_C.DEMO.LABEL_FILE_PATH = ""

# Specify a camera device as input. This will be prioritized
# over input video if set.
# If -1, use input video instead.
_C.DEMO.WEBCAM = -1

# Path to input video for demo.
_C.DEMO.INPUT_VIDEO = ""
# Custom width for reading input video data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input video data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Path to Detectron2 object detection model configuration,
# only used for detection tasks.
_C.DEMO.DETECTRON2_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# Path to Detectron2 object detection model pre-trained weights.
_C.DEMO.DETECTRON2_WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# Threshold for choosing predicted bounding boxes by Detectron2.
_C.DEMO.DETECTRON2_THRESH = 0.9
# Number of overlapping frames between 2 consecutive clips.
# Increase this number for more frequent action predictions.
# The number of overlapping frames cannot be larger than
# half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
_C.DEMO.BUFFER_SIZE = 0
# If specified, the visualized outputs will be written this a video file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output video file.
# If not set (-1), use fps rate from input file.
_C.DEMO.OUTPUT_FPS = -1
# Input format from demo video reader ("RGB" or "BGR").
_C.DEMO.INPUT_FORMAT = "BGR"
# Draw visualization frames in [keyframe_idx - CLIP_VIS_SIZE, keyframe_idx + CLIP_VIS_SIZE] inclusively.
_C.DEMO.CLIP_VIS_SIZE = 10
# Number of processes to run video visualizer.
_C.DEMO.NUM_VIS_INSTANCES = 2

# Path to pre-computed predicted boxes
_C.DEMO.PREDS_BOXES = ""
# Whether to run in with multi-threaded video reader.
_C.DEMO.THREAD_ENABLE = False
# Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
# This is used for fast demo speed by reducing the prediction/visualiztion frequency.
# If -1, take the most recent read clip for visualization. This mode is only supported
# if `DEMO.THREAD_ENABLE` is set to True.
_C.DEMO.NUM_CLIPS_SKIP = 0
# Path to ground-truth boxes and labels (optional)
_C.DEMO.GT_BOXES = ""
# The starting second of the video w.r.t bounding boxes file.
_C.DEMO.STARTING_SECOND = 900
# Frames per second of the input video/folder of images.
_C.DEMO.FPS = 30
# Visualize with top-k predictions or predictions above certain threshold(s).
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.
_C.DEMO.COMMON_CLASS_THRES = 0.7
# Theshold for uncommon class names. This will not be
# used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# This is chosen based on distribution of examples in
# each classes in AVA dataset.
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# Slow-motion rate for the visualization. The visualized portions of the
# video will be played `_C.DEMO.SLOWMO` times slower than usual speed.
_C.DEMO.SLOWMO = 1


def assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.optimizer.BASE_LR_SCALE_NUM_SHARDS:
        cfg.optimizer.BASE_LR *= cfg.NUM_SHARDS
        cfg.optimizer.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.optimizer.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
