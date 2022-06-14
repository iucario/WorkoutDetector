import os
import yaml

__config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yml')))
PROJ_ROOT = __config['proj_root']