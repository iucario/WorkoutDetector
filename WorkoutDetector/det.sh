# Run from directory WorkoutDetector/WorkoutDetector. Or modify path
python webcam_demo.py --camera-id ../data/RepCountA/video/test/stu1_27.mp4 \
    ../mmaction2/configs/recognition/tsm/tsm_my_config.py \
    ../checkpoints/tsm_r50_256h_1x1x16_50e_sthv2_rgb_20220517_best_top1_acc_epoch_58.pth \
    /home/umi/projects/WorkoutDetector/datasets/RepCount/classes.txt
