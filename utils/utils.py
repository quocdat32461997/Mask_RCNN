"""
utils.py - module to implement helper functions for trainig and inference
"""

""" import dependenceis """
import numpy as np
import tensorflow as tf

def rand(a=0, b=1):
    """
    rand - method to randomly generate float number between a and b
    Inputs:
        a - lower bound
        b - upper bound
    Outputs:
        __ - random float number between a and b
    """
    return np.random.rand()*(b-a) + a


def generate_pyramid_anchors(scales, ratios, feature_shapes, backbone_strides, anchor_strides):
    """
    Generate anchors at different scales and different ratios
    Inputs:
        - scales : list of anchor scale
        - ratios : list of anchor ratio
        - feature_shapes : output shape of the last ResNet layer
        - feature_strides : strides within given features
        - anchor_stride : stride of the given anchor
    Outputs:
        - anchors : list of anchors
            In shape of [N, (x1, y1, x2, y2)]
    """
    anchors = []
    for scale, feature_shape, feature_stride in zip(scales, feature_shapes, feature_strides):
        anchors.append(generate_anchors(scale, ratios, feature_shape, feature_stride, anchor_stride))
    return np.concatenate(anchors, axis = 0)
def generate_anchors(scale, ratios, feature_shape, feature_stride, anchor_stride):
    """
    generate_anchors - function to generate anchors that are relative to the backbone's feature maps
    Inputs:
        - scale : Integer
            Scale of the anchor
        - ratio : list of Float numbers
            List of Ratios of one side to another side in an anchor
        - feature_shape : list
            Shape of the given backbone feature
        - feature_stride : Integer
            The given stride within the feature map
        - anchor_stride : Integer
    Outputs:
        - anchors : list of anchors according to the given inputs
    """
    # Get all combinations of scales and ratios
    #scales, ratios = np.meshgrid(np.array(scale), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    scales, ratios = tf.meshgrid(np.array(scales), np.array(ratios))

    # Enumerate heights and widths from scales and ratios
    heights = scales / ratios
    widhts = scales * ratios

    # Enumerate shifts in feature space
    grid_x = tf.tile(tf.reshape(tf.arange(start = 0, limit = feature_shape[0], \
        delta = anchor_stride), [1, -1, 1]), [feature_shape[1], 1, 1]) * feature_stride
    grid_y = tf.tile(tf.reshape(tf.arange(start = 0, limit = feature_shape[1], \
        delta = anchor_stride), [-1, 1, 1]), [1, feature_shape[0], 1]) * feature_stride
    heights = tf.tile(tf.reshape(heights, [1, -1, 1]), [heights.shape, 1, 1])
    widths = tf.tile(tf.reshape(widths, [-1, 1, 1]), [1, widths.shape, 1])
    #  Enumerate combinations of shifts, widths, and shifts
    box_xy = tf.concat([grid_x, grid_y], axis = 2).reshape([-1, 2])
    box_wh = tf.concat([heights, widths], axis = 2).reshape([-1, 2])

    return tf.concat([box_xy - 0.5 * box_wh, box_xy + 0.5 * box_wh], axis = 1)
