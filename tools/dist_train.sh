#!/usr/bin/env bash

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --nproc_per_node=8 \
    --master_port=29500 \
    workoutdetector/train_rep.py \
    --cfg workoutdetector/configs/tpn.py

