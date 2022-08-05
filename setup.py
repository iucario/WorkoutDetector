from setuptools import setup

setup(
    name='workoutdetector',
    version='0.0.1',
    description='WorkoutDetector',
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
