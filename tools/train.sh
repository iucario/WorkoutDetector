#!/usr/bin/bash

python workoutdetector/trainer.py \
    --cfg workoutdetector/configs/tsm.yaml \
    trainer.max_epochs 40 \
    optimizer.lr 0.01 \
    lr_scheduler.step 10  \
    seed 42
