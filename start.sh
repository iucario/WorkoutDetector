docker run -it \
    --gpus=all \
    --shm-size=4gb \
    --user="$(id -u):$(id -g)" \
    --volume="$PWD:/app" \
    -w /app \
    --entrypoint bash \
    --name ani \
    anibali/pytorch:1.10.2-cuda11.3 