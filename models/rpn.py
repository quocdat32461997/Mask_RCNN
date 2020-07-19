"""
rpn.py - module to implement Region Proposal Network
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Softmax

class RegionProposalNetwork(tf.keras.Model):
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
        self.conv = Conv2D(filters = 512, kernel_size = 3, activation = 'relu')
        self.obj = Conv2D(filters = 2 * self.num_anchors, kernel_size = 1, name = 'rpn_obj_detection')
        self.reg = Conv2D(filters = 4 * self.num_anchors, kernel_size = 1, name = 'rpn_bbox_detection')

    def call(self, inputs):
        """
        Inputs:
            inputs - numpy array of feature maps
                    in shape of [batch_size, height, width, filters]
        Outputs
            self.obj(outputs)
                    - object detection for corresponding bounding boxes
                    in shape of [batch_size, h x w x num_anchors, 2]
            self.reg(outputs)
                    - proposed bounding boxes
                    in shape of [batch_size, h x w x num_anchors, 4]
        """
        batch_size, h, w, _ = inputs.shape
        outputs = self.conv(inputs)
        obj_class = self.obj(outputs)
        bboxes = self.reg(outputs)

        obj_class = tf.reshape(obj_class, shape = [-1, h * w * self.num_anchors, 2]) #reshape to [batch_size, h x w x anchors_per_location, 2]
        obj_class = Softmax()(obj_class)
        bboxes = tf.reshape(bboxes, shape = [-1, h * w * self.num_anchors, 4])

        print("obj_class: {}".format(obj_class))
        print("bboxes: {}".format(bboxes))
        return obj_class, bboxes

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
