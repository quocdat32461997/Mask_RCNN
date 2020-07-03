"""
modle.py - module to implement Mask RCNN
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv2D, UpSampling2D

from models.resnet import ResNet
from models.rpn import RegionProposalNetwork
from models.roi import ROIAlign

class Mask_RCNN(tf.keras.Model):
    def __init__(self, anchors, num_class, image_shape = 416, max_objects = 20, resnet_unfreeze, name = 'Mask_RCNN'):
        """
        Inputs:
            anchors - list of anchors
            resnet_unfreeze = list of layers to unfreeze
            name - model name
        """
        self.name = name
        self.resnet = ResNet(input_tensor = Input(shape = (image_shape, image_shape, 3)), input_shape = image_shape)
        self.rpn = RegionProposalNetwork(anchors)
        self.roi = ROIAlign(image_shape = image_shape)
        self.FC_layers = []
        self.mask_layers = []
        for filtes in [256, num_class]:
            self.FC_layers.append(Dense(units = 4096))
            self.mask_layers.append(Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same'))
        self.class_layer = Dense(units = num_classs)
        self.bbox_layer = Dense(units = max_objects * num_class)

    def call(self, inputs):
        """
        Inputs:
            inputs - numpy array of images in shape of [batch_size, height, width, 3]
        Outputs:
            classes - numpy array of object classes in shape of [batch_size, max_objects, num_class]
            bboxes - numpy array of bounding boxes in shape of [batch_size, max_objects, 4 * num_class]
            masks - numpy array of masks in shape of [batch_size, max_obejcts, height, width]
        """
        outputs = self.resnet(inputs)
        prns = self.rpn(outputs)
        outputs = self.roi(outputs, prns)

        

        for layer in self.FC_layers:
            outputs = layer(outputs)

        # generate regression outputs of classes and bboxes
        classes = self.class_layer(outputs)
        bboxes = self.bbox_layer(outputs)

        # generate mask
        masks = UpSampling2D(size = (2,2))(outputs) # upsample feature maps
        for layer in self.mask_layers:
            masks = layer(masks)

        return classes, bbboxes, masks

    def loss(self, classes, bboxes, masks, true_bboxes, true_masks):
        """
        Inputs:
            classes - predicted classes in shape of [batch_size, max_objects, num_class]
            bboxes - predicted bounding boxes in shape of [batch_size, max_objects * num_class]
            masks - predicted masks in shape of [batch_size, max_objects, height, width]
            true_bboxes - target bounding boxes in shape of [batch_size, height, width, 5]
            true_masks - target masks in shape of [batch_size, max_objects, height, width]
        Outputs:
            loss - accumulate loss from ROIAlign and RPN layers
        """
        loss = 0
        return loss
