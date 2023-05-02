import os

import numpy as np

HF_TOKEN = os.getenv('HF_TOKEN')
UPLOAD_REPO_ID = os.getenv('UPLOAD_REPO_ID')
UPLOAD_RESULT_IMAGE = os.getenv('UPLOAD_RESULT_IMAGE') == '1'

# UI options
SHOW_DUPLICATE_BUTTON = os.getenv('SHOW_DUPLICATE_BUTTON', '0') == '1'
SHOW_DEVICE_WARNING = os.getenv('SHOW_DEVICE_WARNING', '1') == '1'
SHOW_ADVANCED_OPTIONS = os.getenv('SHOW_ADVANCED_OPTIONS', '1') == '1'
SHOW_UPSCALE_TO_256_BUTTON = os.getenv('SHOW_UPSCALE_TO_256_BUTTON',
                                       '0') == '1'
SHOW_NUM_IMAGES = os.getenv('SHOW_NUM_IMAGES_OPTION', '1') == '1'
SHOW_CUSTOM_TIMESTEPS_1 = os.getenv('SHOW_CUSTOM_TIMESTEPS_1', '1') == '1'
SHOW_CUSTOM_TIMESTEPS_2 = os.getenv('SHOW_CUSTOM_TIMESTEPS_2', '1') == '1'
SHOW_NUM_STEPS_1 = os.getenv('SHOW_NUM_STEPS_1', '0') == '1'
SHOW_NUM_STEPS_2 = os.getenv('SHOW_NUM_STEPS_2', '0') == '1'
SHOW_NUM_STEPS_3 = os.getenv('SHOW_NUM_STEPS_3', '1') == '1'
GALLERY_COLUMN_NUM = int(os.getenv('GALLERY_COLUMN_NUM', '4'))

# Parameters
MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', '10'))
MAX_SEED = np.iinfo(np.int32).max
MAX_NUM_IMAGES = int(os.getenv('MAX_NUM_IMAGES', '4'))
DEFAULT_NUM_IMAGES = min(MAX_NUM_IMAGES,
                         int(os.getenv('DEFAULT_NUM_IMAGES', '4')))
MAX_NUM_STEPS = int(os.getenv('MAX_NUM_STEPS', '200'))
DEFAULT_CUSTOM_TIMESTEPS_1 = os.getenv('DEFAULT_CUSTOM_TIMESTEPS_1',
                                       'smart100')
DEFAULT_CUSTOM_TIMESTEPS_2 = os.getenv('DEFAULT_CUSTOM_TIMESTEPS_2', 'smart50')
DEFAULT_NUM_STEPS_3 = int(os.getenv('DEFAULT_NUM_STEPS_3', '40'))

# Model options
DISABLE_AUTOMATIC_CPU_OFFLOAD = os.getenv(
    'DISABLE_AUTOMATIC_CPU_OFFLOAD') == '1'
DISABLE_SD_X4_UPSCALER = os.getenv('DISABLE_SD_X4_UPSCALER') == '1'

# Other options
RUN_GARBAGE_COLLECTION = os.getenv('RUN_GARBAGE_COLLECTION', '1') == '1'
DEBUG = os.getenv('DEBUG') == '1'

# Default options for the public demo
if os.getenv('IS_PUBLIC_DEMO') == '1':
    # UI
    SHOW_DUPLICATE_BUTTON = True
    SHOW_NUM_STEPS_3 = False
    SHOW_CUSTOM_TIMESTEPS_1 = False
    SHOW_CUSTOM_TIMESTEPS_2 = False
    SHOW_NUM_IMAGES = False
    # parameters
    DEFAULT_CUSTOM_TIMESTEPS_1 = 'smart50'
    UPLOAD_RESULT_IMAGE = True
    # model
    DISABLE_AUTOMATIC_CPU_OFFLOAD = True
    RUN_GARBAGE_COLLECTION = False
