# Workout Detector

Detects workouts and counts repetitions in videos.

## Installation

```
git clone https://github.com/iucario/workoutdetector.git
cd WorkoutDetector
git checkout clean
pip install -r requirements.txt
pip install -e .
```

## Docker

Build image for dev environment example:

```
docker built -t workout/dev docker
docker run -it \
    --gpus=all \
    --shm-size=32gb \
    --volume="$PWD:/work" \
    --volume="/home/$USER/data:/home/user/data:ro" \
    -w /work \
    --entrypoint zsh \
    --name devbox \
    workout/dev

sudo apt install vim tree -y
pip install wandb pytest
sudo pip install -e .
```

Run docker example

```
docker run --rm -it \
  --gpus=all \
  --shm-size=32gb \
  --volume="$PWD:/work" \
  --volume="/home/$USER/data:/home/user/data:ro" \
  workout/dev:latest python3 workoutdetector/trainer.py
```

## Repetition counting

Method is naive. The transition of states is counted as one repetition. 
Hidden markov model is used to count repetitions.

### Evaluation

1. Inference videos and save results to a directory. Results of each video will be saved in a JSON file.
   `workoutdetector/predict.py`
   ```python
   ckpt = 'checkpoints/model.onnx'
   model = onnxruntime.InferenceSession(ckpt, providers=['CUDAExecutionProvider'])
   inference_dataset(model, ['train', 'val', 'test'], out_dir='out/tsm_rep_scores', checkpoint=ckpt)
   ```
   Scores of each video are saved in `out/tsm_rep_scores/video.mp4.score.json`.
2. Evaluating mean absolute error and off-by-one accuracy
   `workoutdetector/utils/eval_count.py`
   ```python
   json_dir = 'out/tsm_rep_scores'
   anno_path = 'data/RepCount/annotation.csv'
   out_csv = 'out/tsm_rep_scores.csv'
   main(json_dir, anno_path, out_csv, softmax=True)
   analyze_count(out_csv, out_csv.replace('.csv', '_meta.csv'))
   ```
   Results of every video are saved in `out/tsm_rep_scores.csv`.
   Metrics are saved in `out/tsm_rep_scores_meta.csv`.
3. Visualization
   `notebooks/rep_analysis.ipynb`

## Train an action recognition model

### Extract video frames

Extract frames to `data/{dataset}/rawframes/{split}/{video_name}/img_{frame_id}.jpg`

### Train video with TSM

1. Build label files
   `scripts/build_label_files.py`

   ```
    anno_file = 'datasets/RepCount/annotation.csv'
    data_root = os.path.join(PROJ_ROOT, 'data')
    dst_dir = os.path.join(data_root, 'Binary')
    build_with_start(data_root, anno_file, dst_dir)
   ```

2. Download weights pretrained on SSV2:
   https://hanlab.mit.edu/projects/tsm/models/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth
3. Modify config file
   `configs/repcount_12_tsm.yaml`
4. Train
   `python workoutdetector/trainer.py --cfg workoutdetector/configs/tsm.yaml`

- Configs
  Best weigths are saved in directory `{cfg.trainer.default_root_dir}/checkpoints`,
  in format `best-val-acc={val/acc:.2f}-epoch={epoch:02d}" + f"-{timenow}.ckpt`.
  Modify `callbacks.modelcheckpoint.dirpath` to save weights in a different directory.

  Tensorboard logs are in `{cfg.log.output_dir}/{cfg.log.name}/version_{num}`.

  Wandb logs are in `{cfg.log.output_dir}/wandb`.

## Acknowledgements

This project uses pretrained models from:

- [TSM](https://hanlab.mit.edu/projects/tsm/)
