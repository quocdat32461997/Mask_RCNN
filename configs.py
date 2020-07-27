import os

# definition of path to images & annotations and parameters
ROOT_PATH = os.path.abspath('.')
DATA_PATH = os.path.join(ROOT_PATH, 'data')

COCO_PATH = os.path.join(DATA_PATH, 'coco')

CUSTOM_PATH = os.path.join(DATA_PATH, 'custom')

RESNET50_WEIGHTS = os.path.join(ROOT_PATH, 'weights', 'resnet50_weights.h5')

# model parameters
IMAGE_SHAPE = 416

# training parameters
LEARNING_RATE = 0.0001

BATCH_SIZE = 16

MAX_ITER = 15000

RPN_NMS_THRESHOLD = 0.5

CLASS_THRESHOLD = 0.7

MAX_OBJECTS = 20

ANCHORS = [16,64,32,32,64,16,32,128,64,64,128,32,64,256,128,128,256,64,128,512,256,256,512,64,256,1024,512,512,1024,256]

NUM_CLASS = 80

ANCHOR_SCALES = [32, 64, 128, 256, 512]

ANCHOR_RATIOS = [0.5, 1, 2]
