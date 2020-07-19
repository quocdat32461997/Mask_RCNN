"""
import config as cfg
import os
from utils.data import COCOLoader

coco_train = COCOLoader(image_path = os.path.join(cfg.COCO_PATH, 'val2017'), annotation_path = os.path.join(cfg.COCO_PATH, 'annotations/instances_val2017.json'), data_augmentation = True)

for x in coco_train():
    print(type(x))
    input()
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file
from models.resnet import ResNet

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
inputs = tf.keras.layers.Input((416, 416, 3))
resnet = ResNet()
outputs = resnet(inputs)
model = tf.keras.Model(inputs = inputs, outputs = outputs)
print(model)

preloaded_layers = model.layers.copy()
preloaded_weights = []
for pre in preloaded_layers:
    preloaded_weights.append(pre.get_weights())
weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
model.load_weights(weights_path, by_name = True)
# compare previews weights vs loaded weights
for layer, pre in zip(model.layers, preloaded_weights):
    weights = layer.get_weights()

    if weights:
        if np.array_equal(weights, pre):
            print('not loaded', layer.name)
        else:
            print('loaded', layer.name)
model(tf.constant(np.zeros((8, 416, 416, 3))))
print(model.summary())
