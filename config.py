import os

# definition of path to images & annotations and parameters
ROOT_PATH = os.path.abspath('.')
DATA_PATH = os.path.join(ROOT_PATH, 'data')

COCO_PATH = os.path.join(DATA_PATH, 'coco')

CUSTOM_PATH = os.path.join(DATA_PATH, 'custom')

WEIGHTS_DIR = os.path.join(ROOT_PATH, 'weights')

# model parameters
IMAGE_SHAPE = 416

# training parameters
LEARNING_RATE = 0.0001

BATCH_SIZE = 16

MAX_ITER = 15000

IOU_THRESHOLD = 0.5

CLASS_THRESHOLD = 0.7
