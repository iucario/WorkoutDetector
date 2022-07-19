#!/usr/bin/bash
ts=$(date +%Y%m%d-%H%M%S)

python workoutdetector/trainer.py \
    --cfg workoutdetector/configs/tsm.yaml \
    trainer.max_epochs 40 \
    optimizer.lr 0.001 \
    lr_scheduler.step 7  \
    data.dataset_type VideoDataset \
    data.data_root $HOME/data/RepCount/action_video \
    data.train.anno $HOME/data/RepCount/action_video/train.txt \
    data.val.anno $HOME/data/RepCount/action_video/val.txt \
    data.test.anno $HOME/data/RepCount/action_video/test.txt \
    timestamp "$ts" \
    seed 42 \
    log.wandb.enable False \
    trainer.fast_dev_run True
