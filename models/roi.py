"""
roi.py - module to implement ROIAling layer for Mask RCNN
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.image import crop_and_resize

class ROIAlign(tf.keras.Model):
    """
    ROIAlign - layer to align feature maps with proposed regions
    """
    def __init__(self, image_shape, batch_size, crop_size = 7, num_anchors = 9, name = 'ROIAlign'):
        self.name = name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.num_anchors  num_anchors
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

        # rescale offsets of ROIs
        regions[:, :, [0, 2]] *= scale_h
        regions[:, :, [1, 3]] *= scale_w
        regions = tf.reshape(regions, (-1, 4))

        # reshaep classes to [batch_size * num_boxes, 2]
        scores = tf.reshape(scores, (-1, 2))

        # crop and resize
        box_indices = tf.reshape(tf.tile(
            tf.expand_dims(tf.range(features.shape[0]), axis = -1),
                multiples = [1, num_boxes]), (-1))
        features = crop_and_resize(features, regions,
            box_indices = box_indices, crop_size = (self.crop_size, self.crop_size),
            name = 'cropping_objects')
        
        return features, regions, scores
