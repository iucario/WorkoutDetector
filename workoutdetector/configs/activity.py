# model settings
model = dict(type='PEM',
             pem_feat_dim=32,
             pem_hidden_dim=256,
             pem_u_ratio_m=1,
             pem_u_ratio_l=2,
             pem_high_temporal_iou_threshold=0.6,
             pem_low_temporal_iou_threshold=0.2,
             soft_nms_alpha=0.75,
             soft_nms_low_threshold=0.65,
             soft_nms_high_threshold=0.9,
             post_process_top_k=100)

# optimizer
optimizer = dict(type='Adam', lr=0.01, weight_decay=0.00001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=10)
total_epochs = 20

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = '/home/umi/data/RepCount/rawframes/'
prefix_train = data_root + 'train/'
prefix_val = data_root + 'val/'
prefix_test = data_root + 'test/'
ann_file_train = '/home/umi/data/RepCount/activity/train.json'
ann_file_val = '/home/umi/data/RepCount/activity/val.json'
ann_file_test = '/home/umi/data/RepCount/activity/test.json'

work_dir = 'work_dirs/bsn_400x100_20e_1x16_activitynet_feature/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

test_pipeline = [
    dict(type='LoadProposals',
         top_k=1000,
         pgm_proposals_dir=pgm_proposals_dir,
         pgm_features_dir=pgm_features_dir),
    dict(type='Collect',
         keys=['bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score'],
         meta_name='video_meta',
         meta_keys=[
             'video_name', 'duration_second', 'duration_frame', 'annotations',
             'feature_frame'
         ]),
    dict(type='ToTensor', keys=['bsp_feature'])
]

train_pipeline = [
    dict(type='LoadProposals',
         top_k=500,
         pgm_proposals_dir=pgm_proposals_dir,
         pgm_features_dir=pgm_features_dir),
    dict(type='Collect',
         keys=['bsp_feature', 'reference_temporal_iou'],
         meta_name='video_meta',
         meta_keys=[]),
    dict(type='ToTensor', keys=['bsp_feature', 'reference_temporal_iou']),
    dict(type='ToDataContainer',
         fields=(dict(key='bsp_feature',
                      stack=False), dict(key='reference_temporal_iou', stack=False)))
]

val_pipeline = [
    dict(type='LoadProposals',
         top_k=1000,
         pgm_proposals_dir=pgm_proposals_dir,
         pgm_features_dir=pgm_features_dir),
    dict(type='Collect',
         keys=['bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score'],
         meta_name='video_meta',
         meta_keys=[
             'video_name', 'duration_second', 'duration_frame', 'annotations',
             'feature_frame'
         ]),
    dict(type='ToTensor', keys=['bsp_feature'])
]
data = dict(videos_per_gpu=2,
            workers_per_gpu=2,
            train_dataloader=dict(drop_last=True),
            val_dataloader=dict(videos_per_gpu=1),
            test_dataloader=dict(videos_per_gpu=1),
            test=dict(type=dataset_type,
                      ann_file=ann_file_test,
                      pipeline=test_pipeline,
                      data_prefix=prefix_test),
            val=dict(type=dataset_type,
                     ann_file=ann_file_val,
                     pipeline=val_pipeline,
                     data_prefix=prefix_val),
            train=dict(type=dataset_type,
                       ann_file=ann_file_train,
                       pipeline=train_pipeline,
                       data_prefix=prefix_train))
evaluation = dict(interval=1, metrics=['AR@AN'])

# runtime settings
checkpoint_config = dict(interval=1, filename_tmpl='pem_epoch_{}.pth')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
output_config = dict(out=f'{work_dir}/results.json', output_format='json')

checkpoint_config = dict(interval=1)

# runtime settings
dist_params = dict(backend='gloo')
log_level = 'INFO'
ckpt_tem = ('https://download.openmmlab.com/mmaction/localization/bsn/'
            'bsn_tem_400x100_1x16_20e_activitynet_feature/'
            'bsn_tem_400x100_1x16_20e_activitynet_feature_20200619-cd6accc3.pth')
ckpt_pem = ('https://download.openmmlab.com/mmaction/localization/bsn/'
            'bsn_pem_400x100_1x16_20e_activitynet_feature/'
            'bsn_pem_400x100_1x16_20e_activitynet_feature_20210203-1c27763d.pth')
load_from = ckpt_pem
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
