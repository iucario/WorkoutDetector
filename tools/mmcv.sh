docker run -it \
    --gpus all \
    --shm-size 32g \
    -w /app \
    -v /home/$USER/repos/mmaction2:/app \
    -v /home/$USER/projects/WorkoutDetector/data:/data \
    --name mmcv \
    mmcv:latest