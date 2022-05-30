# Workout Detector

This project uses the [MMAction2](https://github.com/open-mmlab/mmaction2)

- [x] Clean and process datasets
- [x] Action recognition
- [ ] Train on more datasets
- [x] Action detection
- [ ] Use pose estimation
- [ ] Repetition counting
- [ ] Action accessment
- [ ] Replace Openmmlab's code


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

1. Download onnx model. [OneDrive](https://1drv.ms/u/s!AiohV3HRf-34i_VY0jVJGvLeayIdjQ?e=XqAvLa)
2. `cd app && uvicorn server:app --port 8000`
3. open http://localhost:8000/

![demo](images/demo.gif)

Kown issue: After stopping streaming, WebSocket will disconnect. You need to refresh to restart streaming.

Going to fix the frontend React code.

## Run Gradio demo

1. Download weights. [OneDrive](https://1drv.ms/u/s!AiohV3HRf-34ipwMjFz1tADQH5U-2w)
2. Copy to checkpoints
3. `python WorkoutDetector/demo.py`


## Train

Check `WorkoutDetector/tutorial.py` in [Google Colab](https://colab.research.google.com/github/iucario/WorkoutDetector/blob/main/WorkoutDetector/tutorial.ipynb)