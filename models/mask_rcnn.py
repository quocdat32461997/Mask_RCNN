"""
mask_rcnn.py - module to implement Mask RCNN
"""

""" import dependencies """

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Conv2D, BatchNormalization, ReLU, AveragePooling2D, Softmax, TimeDistributed, Conv2DTranspose

from models.resnet import ResNet
from models.rpn import RegionProposalNetwork as RPN
from models.roi import ROIAlign

class MaskRCNN(tf.keras.Model):
    """
    Mask_RCNN - implementation of Mask RCNN model consisting of ResNet50, Region Proposal Network, and Classifier & Mask Generator
    """
    def __init__(self, anchors, num_class, batch_size, backbone_weights = 'weights/resnet50_weights.h5', image_shape = 416, max_objects = 20, training = True, name = 'Mask_RCNN'):
        """
        Inputs:
            anchors - list of anchors
            num_class - number of classes
            batch_size - number of images in a batch
            image_shape - input image shape
            max_objects - maximum number of objects
            name - model name
            training - boolean value to denoe the model in training mode
        """
        super(MaskRCNN, self).__init__()
        # model parameters
        self.anchors = anchors
        self.num_class = num_class
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.max_objects = max_objects
        self.training = training

        # definition of layers
        # backbone
        self.resnet = ResNet(architecture = 'resnet50', stage5 = False, train_bn = True)

        # regino proposal network
        self.rpn = RPN(anchors)

        # roi network
        self.roi = ROIAlign(image_shape = image_shape, batch_size = self.batch_size, crop_size = 7, num_anchors = len(self.anchors) // 2)
        self.conv1 = Conv2D(1024, kernel_size = 3, padding = 'same')
        self.conv2 = Conv2D(2048, kernel_size = 3, padding = 'same')

        # rpn classifider
        self.fc = TimeDistributed(Dense(2048, activation = 'relu'))
        self.class_layer = TimeDistributed(Dense(num_class))
        self.bbox_layer = TimeDistributed(Dense(num_class * 4))

        # mask generator
        self.mask_conv1 = Conv2DTranspose(filters = 256, kernel_size = 2, strides = 2, activation = 'relu')
        self.mask_conv2 = Conv2D(filters = num_class, kernel_size = 1, padding = 'same')

    def call(self, inputs):
        """
        Inputs:
            inputs - numpy array of images in shape of [batch_size, height, width, 3]
        Outputs:
            scores - object confidence for corresponding bounding boxes
            class_probs - numpy array of object classes in shape of [batch_size, max_objects, num_class]
            bboxes - numpy array of bounding boxes in shape of [batch_size, max_objects, 4 * num_class]
            masks - numpy array of masks in shape of [batch_size, num_rois, height, width, num_classes]
        """
        # resnet
        outputs = self.resnet(inputs)

        # rpn
        regions, scores = self.rpn(outputs)
        rpn_boxes, rpn_class_logits, rpn_classes = self.rpn(outputs)

        # roi align
        outputs, roi_bboxes, roi_classes = self.roi(outputs, rpn_boxes, rpn_classes)
        outputs = ReLU()(BatchNormalization()(self.conv1(outputs)))
        outptus = ReLU()(BatchNormalization()(self.conv2(outputs)))

        # class and bbox prediction
        class_logits, classes, bboxes = self.rpn_classifier(outputs)

        # mask prediction
        masks = self.mask_generator(outputs)

        if self.training:
            #return {'rpn_bboxes' : rpn_boxes, 'rpn_class_logits' : rpn_class_loits, 'class_logits' : class_logits, 'bboxes' : bboxes, 'masks' : masks}
            return LossLayer()()
        else:
            return {'bboxes' : bboxes, 'masks' : masks, 'classes' : classes}

    def mask_generator(self, inputs):
        """
        mask_generator - function to generate a binary object segmentation
        """
        masks = self.mask_conv1(inputs)
        masks = self.mask_conv2(masks)

        return masks

    def rpn_classifier(self, inputs):
        outputs = AveragePooling2D()(inputs)
        outputs = self.fc(outputs)

        # compute class probs
        class_logits = self.class_layer(outputs)
        class_probs = TimeDistributed(Softmax())(class_logits)

        # compute bounding boxes
        bboxes = self.bbox_layer(outputs)

        return class_logits, class_probs, bboxes

class LossLayer(tf.keras.layers.Layer):
    """
    Loss - a class to implement loss of Mask RCNN
    """
    def __init__(self, name = 'loss_layer', **kwargs):
    #def __init__(self, classes, bboxes, masks, true_bboxes, true_masks):
        super(LossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Inputs:
            - rpn_bboxes : Tensor of shape [batch_size, num_rois, 4]
            - rpn_class_logits : Tensor of shape [batch_size, num_rois, 2]
            - class_logits : Tensor of shape [batch_size, max_objects, num_classes]
            - bboxes : Tensor of shape [bach_size, max_objects, 4]
            - masks : Tensor of shape [batch_size, num_rois, height, width, num_class]
        Outputs:
            loss - accumulate loss from ROIAlign and RPN layers
        """

        return inputs
