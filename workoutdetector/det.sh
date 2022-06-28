# Run from directory WorkoutDetector/workoutdetector. Or modify path
python webcam_demo.py --camera-id /home/umi/projects/WorkoutDetector/data/RepCount/videos/test/stu6_68.mp4 \
    ../mmaction2/configs/recognition/tsm/tsm_my_config.py \
    /home/umi/projects/WorkoutDetector/WorkoutDetector/work_dirs/tsm_8_binary_squat_20220607_1956/best_top1_acc_epoch_16.pth \
    /home/umi/projects/WorkoutDetector/binary_label.txt
