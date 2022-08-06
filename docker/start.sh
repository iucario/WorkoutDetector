docker run -it \
    --gpus=all \
    --shm-size=32gb \
    -u $(id -u):$(id -g) \
    --volume="$PWD:/work" \
    --volume="/home/$USER/data:/home/user/data:ro" \
    -w /work \
    --entrypoint zsh \
    my/dev
