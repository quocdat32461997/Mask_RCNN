"""
rpn.py - module to implement Region Proposal Network
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class RegionProposalNetwork:
    def __init__(self, num_anchors, compute_loss = True):
        self.rpn = Conv2D(filters = 512, kernel_size = 3, activation = 'relu')
        self.cls = Conv2D(filters = 2 * num_anchors, kernel_size = 1)
        self.reg = Conv2D(filters = 4 * num_anchors, kernel_size = 1)

    def train(self, inputs, targets):
        outputs = self.rpn(inputs)
        cls_outputs, reg_outputs = self.cls(outputs), self.reg(outputs)

        return 

    def predict(self, inputs):
        outputs = self.rpn(inputs)
        cls_outputs = self.cls(outputs)
        reg_outputs = self.reg(outputs)
