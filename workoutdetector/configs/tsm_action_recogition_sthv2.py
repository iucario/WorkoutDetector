import os
import time
from workoutdetector.settings import PROJ_ROOT

model = dict(
    type='Recognizer2D',
    backbone=dict(type='ResNetTSM',
                  pretrained='torchvision://resnet50',
                  depth=50,
                  norm_eval=False,
                  shift_div=8),
    cls_head=dict(type='TSMHead',
                  num_classes=11,
                  in_channels=2048,
                  spatial_type='avg',
                  consensus=dict(type='AvgConsensus', dim=1),
                  dropout_ratio=0.5,
                  init_std=0.001,
                  is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

gpu_ids = range(1)

# optimizer
optimizer = dict(type='SGD',
                 constructor='TSMOptimizerConstructor',
                 paramwise_cfg=dict(fc_lr5=True),
                 lr=0.0015 / 8,
                 momentum=0.9,
                 weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 5

# dataset settings
dataset_type = 'RawframeDataset'
data_root = os.path.join(PROJ_ROOT, 'data/Workout/rawframes/')
data_root_val = os.path.join(PROJ_ROOT, 'data/Workout/rawframes/')
data_root_test = os.path.join(PROJ_ROOT, 'data/Workout/rawframes/')
ann_file_train = os.path.join(PROJ_ROOT, 'data/Workout/rawframes/train.txt')
ann_file_val = os.path.join(PROJ_ROOT, 'data/Workout/rawframes/val.txt')
ann_file_test = os.path.join(PROJ_ROOT, 'data/Workout/rawframes/test.txt')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MultiScaleCrop',
         input_size=224,
         scales=(1, 0.875, 0.75, 0.66),
         random_crop=False,
         max_wh_scale_gap=1,
         num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(videos_per_gpu=2,
            workers_per_gpu=1,
            test_dataloader=dict(videos_per_gpu=1),
            train=dict(type=dataset_type,
                       ann_file=ann_file_train,
                       data_prefix=data_root,
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     ann_file=ann_file_val,
                     data_prefix=data_root_val,
                     pipeline=val_pipeline),
            test=dict(type=dataset_type,
                      ann_file=ann_file_val,
                      data_prefix=data_root_val,
                      pipeline=val_pipeline))

# runtime settings
evaluation = dict(interval=1,
                  save_best='auto',
                  metrics=['top_k_accuracy', 'mean_class_accuracy'],
                  topk=(1,))
checkpoint_config = dict(interval=5)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/'\
    'tsm_r50_1x1x8_50e_sthv2_rgb/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth'
resume_from = None
workflow = [
    ('train', 1),
]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
seed = 0
omnisource = False

DATE = time.strftime('%Y%m%d-%H%M%S')
work_dir = os.path.join(PROJ_ROOT, f'log/work_dirs/tsm_action_recogition_sthv2_{DATE}')

log_config = dict(interval=20,
                  hooks=[dict(type='TextLoggerHook'),
                         dict(type='TensorboardLoggerHook')])
