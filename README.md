# Workout Detector

This project uses the [MMAction2](https://github.com/open-mmlab/mmaction2].)

- [x] Clean and process datasets
- [x] Action recognition
- [ ] Train on more datasets
- [ ] Action detection
- [ ] Repetition counting
- [ ] Action accessment


## Installation


```
conda env create -f conda_env.yml
pip install openmim
mim install mmcv
pip install -r requirements.txt

cd mmaction2
pip install -e .
pip install -r requirements/optional.txt
```

## Run Gradio demo

`python WorkoutDetector/demo.py`


## Train

Open `WorkoutDetector/tutorial.py` in [Google Colab](https://colab.research.google.com/github/iucario/WorkoutDetector/blob/main/WorkoutDetector/tutorial.ipynb)