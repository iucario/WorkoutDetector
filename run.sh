docker run --rm -it \
  --gpus=2 \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  work:latest python3 tmp/multi_gpu.py
