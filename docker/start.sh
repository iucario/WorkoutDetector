docker run -it \
    --gpus=all \
    --shm-size=32gb \
    -u $(id -u):$(id -g) \
    -e PROJ_ROOT="/work" \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    --volume="$PWD:/work" \
    --volume="/home/$USER/data:/home/user/data:ro" \
    -w /work \
    --entrypoint zsh \
    --name devbox \
    my/dev
