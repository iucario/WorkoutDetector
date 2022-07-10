# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(type='ResNetTSM',
                  pretrained='torchvision://resnet50',
                  depth=50,
                  out_indices=(2, 3),
                  norm_eval=False,
                  shift_div=8),
    neck=dict(type='TPN',
              in_channels=(1024, 2048),
              out_channels=1024,
              spatial_modulation_cfg=dict(in_channels=(1024, 2048), out_channels=2048),
              temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
              upsample_cfg=dict(scale_factor=(1, 1, 1)),
              downsample_cfg=dict(downsample_scale=(1, 1, 1)),
              level_fusion_cfg=dict(in_channels=(1024, 1024),
                                    mid_channels=(1024, 1024),
                                    out_channels=2048,
                                    downsample_scales=((1, 1, 1), (1, 1, 1))),
              aux_head_cfg=dict(out_channels=174, loss_weight=0.5)),
    cls_head=dict(type='TPNHead',
                  num_classes=12,
                  in_channels=2048,
                  spatial_type='avg',
                  consensus=dict(type='AvgConsensus', dim=1),
                  dropout_ratio=0.5,
                  init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob', fcn_test=True))

# dataset settings
dataset_type = 'MultiActionRepCount'
data_root = '/home/root/data'
data_root_train = '/home/root/data'
data_root_val = '/home/root/data'
data_root_test = '/home/root/data'
ann_file_train = '/home/root/data/Binary/all-train.txt'
ann_file_val = '/home/root/data/Binary/all-val.txt'
ann_file_test = '/home/root/data/Binary/all-test.txt'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
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
    dict(type='SampleFrames',
         clip_len=1,
         frame_interval=1,
         num_clips=8,
         twice_sample=True,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(videos_per_gpu=4,
            workers_per_gpu=8,
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
                      ann_file=ann_file_test,
                      data_prefix=data_root_val,
                      pipeline=test_pipeline))

evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005,
                 nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 30

# runtime settings
work_dir = 'log/work_dirs/tpn_tsm_r50_1x1x8_150e_kinetics400_rgb'
checkpoint_config = dict(interval=5)
log_config = dict(interval=20,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='TensorboardLoggerHook'),
                  ])
# runtime settings
dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = ('https://download.openmmlab.com/mmaction/recognition/'
             'tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/'
             'tpn_tsm_r50_1x1x8_150e_sthv1_rgb_20211202-c28ed83f.pth')
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
omnisource = False