"""
modle.py - module to implement Mask RCNN
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv2D, BatchNormalization, ReLU, AveragePooling2D, Softmax, TimeDistributed, Conv2DTranspose

from models.resnet import ResNet
from models.rpn import RegionProposalNetwork
from models.roi import ROIAlign

class
class Mask_RCNN(tf.keras.Model):
    """
    Mask_RCNN - implementation of Mask RCNN model consisting of ResNet50, Region Proposal Network, and Classifier & Mask Generator
    """
    def __init__(self, anchors, num_class, image_shape = 416, max_objects = 20, resnet_unfreeze, name = 'Mask_RCNN'):
        """
        Inputs:
            anchors - list of anchors
            num_class - number of classes
            image_shape - input image shape
            max_objects - maximum number of objects
            resnet_unfreeze - indices of layers to unfreeze in backbone
            name - model name
        """

        # model parameters
        self.name = name
        self.anchors = anchors
        self.num_class = num_class
        self.image_shape = image_shape
        self.max_objects = max_objects

        # definition of layers

        # backbone
        self.resnet = ResNet(input_tensor = Input(shape = (image_shape, image_shape, 3)), input_shape = image_shape)

        # regino proposal network
        self.rpn = RegionProposalNetwork(anchors)

        # roi network
        self.roi = ROIAlign(image_shape = image_shape)
        self.conv1 = Conv2D(1024, kernel_size = 3, padding = 'same')
        self.conv2 = Conv2D(2048, kernel_size = 3, padding = 'same')

        # rpn classifider
        self.fc = TimeDistributed(Dense(2048, activation = 'relu'))
        self.class_layer = TimeDistributed(Dense(num_classs))
        self.bbox_layer = TimeDistributed(Dense(num_class * 4))

        # mask generator
        self.mask_conv1 = TimeDistributed(Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, activation = 'relu'))
        self.mask_conv2 = TimeDistributed(Conv2D(filters = num_class, kernel_size = 1, padding = 'same'))

    def call(self, inputs):
        """
        Inputs:
            inputs - numpy array of images in shape of [batch_size, height, width, 3]
        Outputs:
            scores - object confidence for corresponding bounding boxes
            class_probs - numpy array of object classes in shape of [batch_size, max_objects, num_class]
            bboxes - numpy array of bounding boxes in shape of [batch_size, max_objects, 4 * num_class]
            masks - numpy array of masks in shape of [batch_size, max_obejcts, height, width]
        """
        outputs = self.resnet(inputs)
        regions, scores = self.rpn(outputs)
        outputs = self.roi(outputs, regions, scores)
        outputs = ReLU()(BatchNormalization()(self.conv1(outputs)))
        outptus = ReLU()(BatchNormalization()(self.conv2(outputs)))

        # class and bbox prediction
        class_logits, class_probs, bboxes = self.rpn_classifier(outputs)

        # mask prediction
        masks = self.mask_generator(outputs)

        return scores, class_probs, bboxes, masks

    def mask_generator(self, inputs):
        masks = self.mask_conv1(inputs)
        masks = self.mask_conv2(masks)

        return masks
    def rpn_classifier(self, inputs):
        outputs = TimeDistributed(AveragePooling2D())(intputs)
        outputs = self.fc(outputs)

        # compute class probs
        class_logits = self.class_layer(outputs)
        class_probs = TimeDistributed(Softmax())(class_logits)

        # compute bounding boxes
        bboxes = self.bbox_layer(outputs)

        return class_logits, class_probs, bboxes

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
