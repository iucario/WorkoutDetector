from setuptools import setup

setup(
    name='workoutdet',
    version='0.0.1',
    description='Workout Detector',
    packages=[],
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'pytorch-lightning',
        'tqdm',
        'fvcore',
        'einops',
        'hmmlearn'
    ],
    homepage='https://github.com/iucario/WorkoutDetector',
)
