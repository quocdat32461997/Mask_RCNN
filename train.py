"""
train.py - module to train Mask RCNN
"""

""" import dependencies """
import tensorflow as tf
import numpy as np
from absl import flags, app, logging
import argparse
from models.mask_rcnn import MaskRCNN
from utils.data import COCOLoader

# define script FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'Mask_RCNN_1', 'Model name')
flags.DEFINE_float('learning_rate', 0.0001, 'learning_rate')
flags.DEFINE_integer('epoch', 200, 'Iterations over dataset')
flags.DEFINE_integer('batch_size', 16, 'Number of samples for every batch')
flags.DEFINE_string('log_path', 'logs', 'Path to Logging foler to save log ingo')
flags.DEFINE_string('weights_path', 'weights', 'Path to folder to save trained weights')
flags.DEFINE_string('data_path', None, 'Path to data folder')
flags.DEFINE_bool('pretrained', False, 'Use pretrained model')
flags.DEFINE_string('model_path', None, 'Path to pretrained Mask_RCNN model')

def main():
    model = MaskRCNN(anchors = [1,2,3,4], num_class = 80, image_shape = 416, max_objects = 20, resnet_unfreeze = [-1])
if __name__ == '__main__':
    print("Star training", FLAGS.model_name)
    logging.info("Start training", FLAGS.model_name)

    app.run(main)
