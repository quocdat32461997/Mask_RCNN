"""
resnet.py - module to implement pretrained ResNet model as backbone
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model

class ResNet(tf.keras.Model):
    def __init__(self, input_tensor, input_shape, name = 'ResNet'):
        self.name = name
        self.resnet = tf.keras.applications.ResNet101V2(include_top = False, weights = 'imagenet'
            input_tensor = input_tensor, input_shape = input_shape, pooling = None)
    def call(self, inputs):
        return self.resnet(inputs)
