docker run -it \
    --gpus=all \
    --shm-size=16gb \
    --volume="$PWD:/work" \
    -w /work \
    --entrypoint zsh \
    --name devbox \
    dev:11.3.1-cudnn8-devel-ubuntu20.04