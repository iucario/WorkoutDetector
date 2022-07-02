docker run --rm -it \
  --gpus=all \
  --shm-size=16gb \
  --volume="$PWD:/work" \
  dev:latest python3 tmp/multi_gpu.py
