"""
roi.py - module to implement ROIAling layer for Mask RCNN
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model

class ROIAlign(tf.keras.Model):
    """
    ROIAlign - layer to align feature maps with proposed regions
    """
    def __init__(self, name = 'ROIAlign'):
        self.name = name

    def call(self, features, regions):
        """
        Inputs:
            features - CONV feature maps from ResNet
            regions - proposed bounding boxes from RPN
        Ouputs:
            features - feature maps aligned with proposed regions
        """
        return features
