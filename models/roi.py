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
                    in shape of [batch_size, 4 * num_anchors]
            scores- object detection for proposed bounding boxes from RPN
                    in shape of [batch_size, 2 * num_anchors]
        Ouputs:
            features - feature maps aligned with proposed regions
                in shape of [batch_size, crop_height, crop_width, filters]
            regions - proposed regions
                in shape of [batch_size * num_anchors, 4]
            scores - object confidence
                in shape of [batch_size, num_anchors, 2]
        """

        # calculate downsampling scale
        h, w = features.shape[1:3] # get shape of feature input
        #regions = tf.reshape(regions, (regions.shape[0], -1, 4)) # reshape to [batch_size, num_anchors, 4]
        #scores = tf.reshape(scores, (scores.shape[0], -1, 2))
        regions = tf.reshape(regions, (-1, 4))
        scale_h, scale_w = h / self.image_shape, w / self.image_shape

        # rescale offsets of ROIs
        regions[:, [0, 2]] *= scale_h
        regions[:, [1, 3]] *= scale_w


        # crop and resize
        box_indices = tf.reshape(tf.tile(
            tf.expand_dims(tf.range(features.shape[0]), axis = -1),
                multiples = [1, self.batch_size * self.num_anchors]), (-1))
        features = crop_and_resize(features, regions, box_indices = box_indices, crop_size = (self.crop_size, self.crop_size), name = 'cropping_objects')

        return features, regions, scores

    def roi_align(args):
        regions = args['regions']
        scores = args['scores']
        features = args['features']

        box_indices = tf.zeros_like()
        # crop and resize
        #crops = crop_and_resize(features, regions, )
