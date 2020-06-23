"""
data.py - module to implement functions to fetch, preprocess, and feed data into estimators
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.image import resize_with_pad, flip_left_right

import json
import cv2

def input_fn(data_path, batch_size = 16, image_shape = 416):
    with open(data_path, 'r') as file:
        data = json.load(file)

    def generator(inputs):
        for input in inputs:
            yield input
class dataset:
    def __init__(self, data_path, batch_size = 16, image_shape = 416, data_augmentation = True):
        with open(data_path, 'r') as file:
            self.data = json.load(file)
        self.batch_size = batch_size
        self.image_shape = image_shpae
        self.data_augmentation = data_augmentation

    def generator(self, inputs):
        for input in inputs:
            # load iamge
            image = cv2.imread(input['file_name'])
            image = resize_with_pad(image, self.image_shape, self.image_shape, method = 'bilinear')

            # augment images

    def train_input_fn(self):
        return None
