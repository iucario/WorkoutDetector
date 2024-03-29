# Workout Detector

- [x] Clean and process datasets
- [x] Action recognition
- [ ] Train on more datasets
- [x] Action detection
- [ ] Use pose estimation
- [x] Repetition counting
- [ ] Action assessment

## Installation

```
git clone https://github.com/iucario/workoutdetector.git
cd WorkoutDetector

conda env create -f conda_env.yml
pip install openmim
mim install mmcv
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
export $PROJ_ROOT=$PWD
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

## React demo

1. Download onnx model. [OneDrive](https://1drv.ms/u/s!AiohV3HRf-34i_VY0jVJGvLeayIdjQ)
2. `cd app && uvicorn server:app --port 8000`
3. open http://localhost:8000/

<img src="images/demo.gif" alt="React demo" width="800"/>

## Run Gradio demo

1. Download onnx model. Same as the React demo. [OneDrive](https://1drv.ms/u/s!AiohV3HRf-34i_VY0jVJGvLeayIdjQ?e=XqAvLa)
2. Copy to checkpoints
3. `python WorkoutDetector/demo.py`
4. open http://localhost:7860/

## Repetition counting

Two model types, image and video, can be used.

Method is naive. The transition of states is counted as one repetition. It's online counting. Only previous frames are used.

### Evaluation

1. Inference videos and save results to a directory. Results of each video will be saved in a JSON file.
   `workoutdetector/utils/inference_count.py`
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

### Local

Be sure to modify config `WorkoutDetector/settings/global_settings.py` to your project root.
E.g. `PROJ_ROOT = '/home/your_name/WorkoutDetector/'`
Or set environment variable `PROJ_ROOT` to your project root.
E.g. `export PROJ_ROOT=$PWD`

### Build dataset

`Workouts` are subsets taking from Countix and RepCount.
For now, I am using 11 classes. [Dataset.md](datasets/Dataset.md)

Use soft links to avoid copying files.

See `WorkoutDetector/scripts/build_datasets.py` for details.

```
data/Workouts/
├── rawframes
│   ├── Countix
│   │   ├── train
│   │   └── val
│   ├── RepCount
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── test.txt
│   ├── test_repcount.txt
│   ├── train.txt
│   ├── train_countix.txt
│   ├── train_repcount.txt
│   ├── val.txt
│   ├── val_countix.txt
│   └── val_repcount.txt
├── test.txt
├── train.txt
└── val.txt
```

### Train video with mmaction2

`python workoutdetector/train.py`
config: `configs/tsm_action_recogition_sthv2.py`

## Train an exercise state recognition model

### Train video with tsm

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

### Train rep with mmaction2

```
python workoutdetector/train_rep.py \
 --action=pull_up \
 --data-prefix=data/RepCount/rawframes/ \
 --ann-dir=data/relabeled/pull_up \
 --ckpt=work_dirs/tsm_MultiActionRepCount_sthv2_20220625-224626/best_top1_acc_epoch_5.pth
```

Classification of action states, e.g. for action `push_up`, up and down are two states.
Also supports multiple actions and multiple states.

Training code is in `workoutdetector/train_rep.py`. Uses `mmaction2` to train the time shift module.
Configs are in `workoutdetector/configs`.

1. Prepare label files.
   The row in the label file is of format: `path/to/rawframe_dir start_frame num_frames label`.
   Frames of indices `start_frame` to `start_frame + num_frames` will be used.
   Don't need to move and rename frames in this way. Just need to modify the label file.

## Train an image recognition model

`python workoutdetector/train_img.py --cfg workoutdetector/configs/pull_up.yaml`

Uses PyTorch Lightning to train a model.

### workoutdetector/utils/inference_count.py

- Inference every frames in a video using image model. Will write count to the `--output` file.
  And save predicted scores to a JSON file in `--output` directory.
  ```
  python workoutdetector/utils/inference_count.py \
  -ckpt checkpoints/pull-up-image-swin-1x3x224x224.onnx \
  --model-type image \
  -i path/to/input/video.mp4 \
  --output out/video.mp4 \
  --action pull_up
  ```

## Scripts

`workoutdetector/scripts/`

- `mpvscreenshot_process.py`
  Until I create or find a nice video segment tool, I'll use this script to annotate videos.
  How to use:

  1.  The mpv screenshot filename template config is `screenshot-template=~/Desktop/%f_%P`
  2.  Will get files like `stu2_48.mp4_00_00_09.943.png`
  3.  If saved in train, val, test folders, use `label_from_split(root_dir)`
  4.  If screenshots are saved in a single folder, I need to write a new script.
  5.  And `screenshots_to_csv` can save filenames with timestamp to csv for future usage.

- `build_label_list.py`
  - `relabeled_csv_to_rawframe_list`
    Use this with `mpvscreenshot_process.py` together.
    Generates label files for mmaction rawframe datasets.

## Acknowledgements

This project uses pretrained models from:

- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [TSM](https://hanlab.mit.edu/projects/tsm/)
