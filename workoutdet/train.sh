#!/usr/bin/bash
ts=$(date +%Y%m%d-%H%M%S)

python workoutdet/trainer.py \
    --cfg workoutdet/repcount.yaml \
    trainer.max_epochs 20 \
    optimizer.lr 0.015 \
    seed 42 \
    timestamp $ts
