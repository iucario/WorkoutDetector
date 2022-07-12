docker run -it \
    --gpus all \
    --shm-size 32g \
    --name uniformer \
    -w /app \
    -v /home/$USER/repos/UniFormer:/app \
    -v /home/$USER/projects/WorkoutDetector/data:/data \
    cnstark/pytorch:1.11.0-py3.9.12-cuda11.3.1-devel-ubuntu20.04 \
    mkdir path_to_models && \
    wget "https://api.onedrive.com/v1.0/shares/u!\
        aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW9oVjNIUmYtMzRpX1luMHAwcWxMMkZGM2p1c0E=/root/content" \
        -O path_to_models/uniformer_small_in1k.pth
