"""
rpn.py - module to implement Region Proposal Network
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class RegionProposalNetwork(tf.keras.Model):
    """
    RegionProposalNetwork - a small model to propose bounding boxes for ROIAlign layer
    """
    def __init__(self, num_anchors, name = 'RPN'):
        """
        Inputs:
            num_anchors - maximum possible proposals for each location
            name - RegionProposalNetwork model
        """
        self.num_anchors = num_anchors
        self.rpn = Conv2D(filters = 512, kernel_size = 3, activation = 'relu')
        self.obj = Dense(2 * num_anchors, activation = 'softmax')
        self.reg = Dense(4 * num_anchors)

        self.name = name

    def call(self, inputs):
        """
        Inputs:
            inputs - numpy array of feature maps
                    in shape of [batch_size, height, width, filters]
        Outputs
            self.obj(outputs)
                    - object detection for corresponding bounding boxes
                    in shape of [batch_size, 2 * num_anchors]
            self.reg(outputs)
                    - proposed bounding boxes
                    in shape of [batch_size, 4 * num_anchors]
        """
        outputs = self.rpn(inputs)
        return self.obj(outputs), self.reg(outputs)

    def loss(self, classes, bboxes, targets):
        """
        Inputs:
            classes - proposed classes in shape of [batch_size, max_objects]
            bboxes - proposed bounding boxes in shape of [batch_size, max_objects * 4]
            targets - numpy array of targets in shape of [batch_size, max_objects, 5]
        Outputs:
            loss - integer of sum of class and bbox loss
        """
        lsos = 0
        raw_regions = targets[:4]
        raw_classes = targets[4:]

        return loss
