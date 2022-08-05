# Workout Detector

Detects workouts and counts repetitions in videos.

## Requirements

- Data: [RepCount](https://github.com/SvipRepetitionCounting/TransRAC)
- Weights: [TSM](https://hanlab.mit.edu/projects/tsm/models/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth)

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
   `workoutdet/predict.py`

   ```python
   ckpt = 'checkpoints/model.ckpt'
   out_dir = 'out/tsm_rep_scores'
   main(ckpt, out_dir, stride=1, step=1, rank=0, world_size=1)
   ```

   Scores of each video will be saved in `out/tsm_rep_scores/{video}.stride_1_step_1.json`.

2. Counting repetitions using hidden markov model
   `workoutdet/hmm.py`

   ```python
   anno_path = 'datasets/RepCount/annotation.csv'
   json_dir = 'json'
   hmm_eval(anno_path, json_dir)
   ```
   Will save results to 'hmm_result.csv' and print the results.

3. Or counting repetitions using the naive method
   `workoutdet/evaluate.py`
   ```python
   json_dir = 'out/tsm_rep_scores'
   anno_path = 'datasets/RepCount/annotation.csv'
   out_csv = 'out/tsm_rep_scores.csv'
   main(json_dir, anno_path, out_csv, window=10, stride=1, step=1, threshold=0.5, softmax=True)
   count_stats(out_csv, out_csv.replace('.csv', '_meta.csv'))
   ```
   Results of every video will be saved in `out/tsm_rep_scores.csv`.
   Metrics will be saved in `out/tsm_rep_scores_meta.csv`.

## Train an action recognition model

### Prepare data
1. Extract video frames
   `workoutdet/scripts/extract_frame.py`

2. Build label files
   `workoutdet/scripts/build_label_file.py`

### Train video with TSM

1. Build label files
   `scripts/build_label_files.py`

   ```python
    anno_file = 'datasets/RepCount/annotation.csv'
    data_root = os.path.join(PROJ_ROOT, 'data')
    dst_dir = os.path.join(data_root, 'Binary')
    build_with_start(data_root, anno_file, dst_dir)
   ```

2. Download weights pretrained on Something-Something-v2:
   https://hanlab.mit.edu/projects/tsm/models/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth
3. Modify config file
   `configs/repcount.yaml`
4. Train
   `python workoutdet/trainer.py --cfg workoutdet/repcount.yaml`

- Configs
  Best weigths are saved in directory `{cfg.trainer.default_root_dir}/checkpoints`,
  in format `best-val-acc={val/acc:.2f}-epoch={epoch:02d}" + f"-{timenow}.ckpt`.
  Modify `callbacks.modelcheckpoint.dirpath` to save weights in a different directory.

  Tensorboard logs are in `{cfg.log.output_dir}/{cfg.log.name}/version_{num}`.

  Wandb logs are in `{cfg.log.output_dir}/wandb`.

## Acknowledgements

This project uses pretrained models from:

- [TSM](https://hanlab.mit.edu/projects/tsm/)

Data from:
- [RepCount](https://github.com/SvipRepetitionCounting/TransRAC)