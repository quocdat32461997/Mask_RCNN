"""
resnet.py - module to implement pretrained ResNet model as backbone
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model

class ResNet:
    """
    ResNet - class to implement ResNet model
    """
    def __init__(self, input_tensor, input_shape, name = 'ResNet'):
        """
        Inputs:
            input_tensor - output of Input layer
            input_shape - output shape of Input layer
            name - layer name
        """
        self.name = name
        self.resnet = tf.keras.applications.ResNet101V2(include_top = False, weights = 'imagenet',
            input_tensor = input_tensor, input_shape = input_shape, pooling = None)

    def call(self):
        """
        Outputs: ResNet model
        """
        return self.resnet
