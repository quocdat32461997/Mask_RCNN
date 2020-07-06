"""
train.py - module to train Mask RCNN
"""

""" import dependencies """
import tensorflow as tf
import numpy as np
import os
from absl import flags, app, logging
from tensorflow.keras.optimizers import Adam

from models.mask_rcnn import MaskRCNN
from utils.data import COCOLoader
import config as cfg

# define script FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'Mask_RCNN_1', 'Model name')
flags.DEFINE_float('learning_rate', 0.0001, 'learning_rate')
flags.DEFINE_integer('epoch', 200, 'Iterations over dataset')
flags.DEFINE_integer('batch_size', 16, 'Number of samples for every batch')
flags.DEFINE_string('log_path', 'logs', 'Path to Logging foler to save log ingo')
flags.DEFINE_string('weights_path', 'weights', 'Path to folder to save trained weights')
flags.DEFINE_string('data_path', 'coco', 'Path to data folder')
flags.DEFINE_bool('pretrained', False, 'Use pretrained model')
flags.DEFINE_string('model_path', None, 'Path to pretrained Mask_RCNN model')

def main(args):
    # beginning of training
    print("Star training", FLAGS.model_name)

    num_class = cfg.NUM_CLASS #by default, 80 classes of COCO datset

    # build data loader
    print("Building data loader for:", FLAGS.data_path)
    # by default, training on COCO dataset
    if FLAGS.data_path == 'coco':
        train_coco = COCOLoader(image_path = os.path.join(cfg.COCO_PATH, 'train2017'), \
            annotation_path = os.path.join(cfg.COCO_PATH, 'annotations/instances_train2017.json'), \
            data_augmentation = True)
        val_coco = COCOLoader(image_path = os.path.join(cfg.COCO_PATH, 'val2017'), \
            annotation_path = os.path.join(cfg.COCO_PATH, 'annotations/instances_val2017.json'), \
            data_augmentation = True)
        num_class = len(train_coco.get_class())
    else:
        # custom dataset
        print("custom dataset")
        num_class = 80

    model = MaskRCNN(anchors = cfg.ANCHORS, num_class = num_class, batch_size = 8, image_shape = cfg.IMAGE_SHAPE, max_objects = cfg.MAX_OBJECTS)
    model.compile(optimizer = Adam(learning_rate = cfg.LEARNING_RATE), loss = 'mse')
    model.predict(np.zeros((8, cfg.IMAGE_SHAPE, cfg.IMAGE_SHAPE, 3)))
    print(model.summary())
if __name__ == '__main__':
    app.run(main)
