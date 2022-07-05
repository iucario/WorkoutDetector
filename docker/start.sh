docker run -it \
    --gpus=all \
    --shm-size=16gb \
    --volume="$PWD:/work" \
    --volume="/home/$USER/data:/home/user/data:ro" \
    -w /work \
    --entrypoint zsh \
    --name devbox \
    my/dev
