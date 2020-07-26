"""
roi.py - module to implement ROIAling layer for Mask RCNN
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model

class ROIAlign(tf.keras.layers.Layer):
    """
    ROIAlign - layer to align feature maps with proposed regions
    """
    def __init__(self, image_shape, batch_size, crop_size = 7, num_anchors = 9, name = 'ROIAlign'):
        super(ROIAlign, self).__init__()
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.crop_size = crop_size

    def call(self, features, regions, scores):
        """
        Inputs:
            features - CONV feature maps from ResNet
                    in shaep of [batch_size, height, width, filters]
            regions - proposed bounding boxes from RPN
                    in shape of [batch_size, h x w x num_anchors, 4]
            scores- object detection for proposed bounding boxes from RPN
                    in shape of [batch_size, h x w x num_anchors, 2]
        Ouputs:
            features - feature maps aligned with proposed regions
                in shape of [batch_size x h x w x num_anchors, crop_height, crop_width, filters]
            regions - proposed regions
                in shape of [batch_size, h x w x num_anchors, 4]
            scores - object confidence
                in shape of [batch_size, h x w x num_anchors, 2]
        """

        # calculate downsampling scale
        h, w = features.shape[1:3] # get shape of feature input
        num_boxes = regions.shape[1]
        scale_h, scale_w = h / self.image_shape, w / self.image_shape

        #scores = tf.reshape(scores, (-1, 2))
        #regions = tf.reshape(regions, (-1, 4))

        """
        ROIAlign - align region indices by diving [x/strides, y/strides]
            that strides is the size gap between orgiinal image and
            input conv feature map. Then, BILINEAR interpolation is performed
            to align coordinates.
        """
        box_indices = tf.reshape(tf.tile(\
            tf.expand_dims(tf.range(self.batch_size), axis = -1),\
            multiples = [1, num_boxes]), [-1])
        features = tf.image.crop_and_resize(features, tf.reshape(regions, (-1, 4)),\
            box_indices = box_indices, crop_size = (self.crop_size, self.crop_size),\
            name = 'cropping_objects')
        return features, regions, scores
