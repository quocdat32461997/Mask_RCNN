"""
resnet.py - module to implement pretrained ResNet model as backbone
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model

class ResNet(tf.keras.Model):
    """
    ResNet - class for Transfer Learning implementation of ResNet50 model
    """
    def __init__(self, input_tensor, input_shape, include_top = False, weights = 'imagent',
        pooling = None, name = 'ResNet'):
        """
        Inputs:
            input_tensor - output of Input layer
            input_shape - output shape of Input layer
            name - layer name
        """
        self.name = name
        self.resnet = tf.keras.applications.ResNet50(include_top = include_top, weights = weights,
            input_tensor = input_tensor, input_shape = input_shape, pooling = pooling)

    def call(self, inputs):
        """
        Inputs:
            inputs - tensor of images in shape of [batch_size, height, width, 3]
        Outputs:
            self.resnet(inputs) - tensor of conv feature maps extracted from 4th CONV block in ResNet50
        """
        return self.resnet(inputs)
