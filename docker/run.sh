docker run --rm -it \
  --gpus=all \
  --shm-size=16gb \
  --volume="$PWD:/work" \
  --volume="/home/$USER/data:/home/user/data:ro" \
  my/dev:latest python3 trainer.py
