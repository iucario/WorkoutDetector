import os

PROJ_ROOT = '/work'
DATA_ROOT = os.path.expanduser('~/data')

if os.environ.get('PROJ_ROOT') is not None:
    PROJ_ROOT = os.environ['PROJ_ROOT']

_repcount_anno_relative_path = 'datasets/RepCount/annotation.csv'
REPCOUNT_ANNO_PATH = os.path.join(PROJ_ROOT, _repcount_anno_relative_path)