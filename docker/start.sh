docker run -it \
    --gpus=all \
    --shm-size=16gb \
    --volume="$PWD:/work" \
    -w /work \
    --entrypoint zsh \
    --name devbox \
    my/dev
