"""
rpn.py - module to implement Region Proposal Network
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Softmax

class RegionProposalNetwork(tf.keras.layers.Layer):
    """
    RegionProposalNetwork - a small model to propose bounding boxes for ROIAlign layer
    """
    def __init__(self, anchors, name = 'RPN'):
        """
        Inputs:
            num_anchors - maximum possible proposals for each location
            name - RegionProposalNetwork model
        """
        super(RegionProposalNetwork, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(self.anchors) // 2
        self.conv = Conv2D(filters = 512, kernel_size = 3, activation = 'relu', padding = 'same')
        self.obj = Conv2D(filters = 2 * self.num_anchors, kernel_size = 1, name = 'rpn_obj_detection')
        self.reg = Conv2D(filters = 4 * self.num_anchors, kernel_size = 1, name = 'rpn_bbox_detection')

    def call(self, inputs):
        """
        Inputs:
            inputs - numpy array of feature maps
                    in shape of [batch_size, height, width, filters]
        Outputs
            bboxes
                    - proposed bounding boxes
                    in shape of [batch_size, h x w x num_anchors, 4]
            obj_class_logits
                    - object classifier logits (before softmax)
                    in shape of [batch_size, h x w x num_anchors, 2]
            obj_classes
                    - object classifier probabilities
                    in shape of [batch_size, h x w x num_anchors, 2]
        """
        batch_size, h, w, _ = inputs.shape
        outputs = self.conv(inputs)
        obj_class = self.obj(outputs)
        bboxes = self.reg(outputs)

        # reshape to [batch_size, h x w x anchors_per_location, 2]
        # object classification for 9 anchors at each location
        obj_class_logits = tf.reshape(obj_class, shape = [-1, h * w * self.num_anchors, 2])
        obj_classes = Softmax()(obj_class)

        # reshape to [batch_size, h x w x anchors_per_location, 2]
        # get top-left and bototm-right coordinates of 9 regions at each location
        bboxes = tf.reshape(bboxes, shape = [-1, h * w * self.num_anchors, 4])

        print("obj_class: {}".format(obj_classes))
        print("bboxes: {}".format(bboxes))
        return bboxes, obj_class_logits, obj_classes
