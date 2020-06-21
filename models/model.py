"""
modle.py - module to implement Mask RCNN
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Concatenate

from models import resnet
from models import rpn

class Mask_RCNN(tf.keras.Model):
    def __init__(self, anchors, resnet_unfreeze, name = 'Mask_RCNN'):
        self.name = name
        self.resnet = resnet.ResNet(resnet_unfreeze, )
        self.rpn = rpn.RegionProposalNetwork(anchors)
        self.roi = None

    def call(self, inputs):
        outputs = self.resnet(inputs)
        rpns = self.rpn(outputs)
        outputs = self.roi(Concatenate(axis = 0)([outputs, prns]))
        return outputs
