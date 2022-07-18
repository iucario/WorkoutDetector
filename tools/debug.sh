#!/usr/bin/bash
ts=$(date +%Y%m%d-%H%M%S)

python workoutdetector/trainer.py \
    --cfg workoutdetector/configs/repcount_12_tsm.yaml \
    trainer.default_root_dir exp/repcount-12-tsm \
    trainer.fast_dev_run False \
    log.wandb.enable False \
    log.tensorboard.enable False \
    optimizer.method SGD \
    optimizer.lr 0.0015 \
    seed 0 \
    timestamp "$ts" 
