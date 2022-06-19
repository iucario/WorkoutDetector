# Workout Detector

This project uses the [MMAction2](https://github.com/open-mmlab/mmaction2)

- [x] Clean and process datasets
- [x] Action recognition
- [ ] Train on more datasets
- [x] Action detection
- [ ] Use pose estimation
- [x] Repetition counting
- [ ] Action accessment

## Installation

```
git clone --recursive https://github.com/iucario/WorkoutDetector.git
cd WorkoutDetector

conda env create -f conda_env.yml
pip install openmim
mim install mmcv
pip install -r requirements.txt

cd mmaction2
pip install -e .
pip install -r requirements/optional.txt
```

## React demo

1. Download onnx model. [OneDrive](https://1drv.ms/u/s!AiohV3HRf-34i_VY0jVJGvLeayIdjQ)
2. `cd app && uvicorn server:app --port 8000`
3. open http://localhost:8000/

<img src="images/demo.gif" alt="React demo" width="800"/>

Kown issue: After stopping streaming, WebSocket will disconnect. You need to refresh to restart streaming.

Going to fix the frontend React code.

## Run Gradio demo

1. Download onnx model. Same as the React demo. [OneDrive](https://1drv.ms/u/s!AiohV3HRf-34i_VY0jVJGvLeayIdjQ?e=XqAvLa)
2. Copy to checkpoints
3. `python WorkoutDetector/demo.py`
4. open http://localhost:7860/

## Inference

### Repetition counting

Two model types, image and video, can be used.

Method is naive. The transition of states is counted as one repetition. It's online counting.

1. Prepare `onnx` model trained using `run.py`
2. Run script
   ```
   python utils/inference_count.py \
        --onnx ../checkpoints/tsm_video_binary_jump_jack.onnx \
        --video path/to/input/video.mp4 \
        -o path/to/output/video.mp4
   ```

## Train

### Colab

Check `WorkoutDetector/tutorial.py` in [Google Colab](https://colab.research.google.com/github/iucario/WorkoutDetector/blob/main/WorkoutDetector/tutorial.ipynb)

### Local

Be sure to modify config `WorkoutDetector/settings/global_settings.py` to your project root.
E.g. `PROJ_ROOT = '/home/your_name/WorkoutDetector/'`

### Build dataset

`Workouts` are subsets taking from Countix and RepCount.
For now, I am using 11 classes. [Dataset.md](datasets/Dataset.md)

Use soft links to avoid copying files.

See `WorkoutDetector/scripts/build_datasets.py` for details.

```
data/Workouts/
├── rawframes
│   ├── Countix
│   │   ├── train -> countix/rawframes/train
│   │   └── val -> countix/rawframes/val
│   ├── RepCount
│   │   ├── test -> RepCount/rawframes/test
│   │   ├── train -> RepCount/rawframes/train
│   │   └── val -> RepCount/rawframes/val
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
