"""
backbones.py - module to implement pretrained ResNet model as backbone
"""

""" import dependencies """
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import  Conv2D, BatchNormalization, ReLU, Add, ZeroPadding2D, MaxPool2D

class ResNet(tf.keras.layers.Layer):
    def __init__ (self, architecture = 'resnet50', stage5 = False, train_bn = True, name = 'ResNet'):
        """
        ResNet  - class for Transfer Learning implementation of ResNet50 model
                - ResNet50 is used as backbone, hence the last Fully Connected layer and pooling of the last conv feature map is removed
        """
        """
        Building the ResNet graph is based on the following Github repos:
            * https://github.com/matterport/Mask_RCNN
            * https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
        Inputs:
            input_tensor: tensor of shape [None, height, width, depth]
            input_shape: shape of tensor input
            architecture: Can be resnet 50 or resnet 1010
            stage5: False, If True, stage of 5 of the network is created
            train_bn: Boolean. Train or freeze Batch Norm layers
            name: Can be ResNet or FPN
        """
        self.architecture = architecture
        self.stage5 = stage5
        self.train_bn = train_bn

    def _identity_block(self, input_tensor, kernel_size, filters, stage, block,
                       use_bias=True, train_bn=True):
        """The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                      use_bias=use_bias)(input_tensor)
        x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
        x = ReLU()(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
        x = ReLU()(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                      use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

        x = Add()([x, input_tensor])
        x = ReLU(name='res' + str(stage) + block + '_out')(x)
        return x

    def _conv_block(self, input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_bias: Boolean. To use or not use a bias in conv layers.
            train_bn: Boolean. Train or freeze Batch Norm layers
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        """
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
        x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
        x = ReLU()(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                      name=conv_name_base + '2b', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
        x = ReLU()(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                      '2c', use_bias=use_bias)(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut, training=train_bn)

        x = Add()([x, shortcut])
        x = ReLU(name='res' + str(stage) + block + '_out')(x)
        return x

    def __call__(self, inputs):
        """
        Building the ResNet graph is based on the following Github repos:
            * https://github.com/matterport/Mask_RCNN
            * https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
        architecture: Can be resnet 50 or resnet 1010
        stage5: False, If True, stage of 5 of the network is created
        train_bn: Boolean. Train or freeze Batch Norm layers
        """
        assert self.architecture in ['resnet50', 'resnet101']

        # Stage 1
        outputs = ZeroPadding2D(padding = (3, 3), name = 'conv1_pad')(inputs)
        outputs = Conv2D(filters = 64, kernel_size = 7, strides = 2, name = 'conv1_conv', use_bias = True)(outputs)
        outputs = BatchNormalization(name = 'bn_conv1')(outputs, training = self.train_bn)
        outputs = ReLU()(outputs)
        C1 = outputs = MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'same')(outputs)
        # Stage 2
        outputs = self._conv_block(outputs, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=self.train_bn)
        outputs = self._identity_block(outputs, 3, [64, 64, 256], stage=2, block='b', train_bn=self.train_bn)
        C2 = outputs = self._identity_block(outputs, 3, [64, 64, 256], stage=2, block='c', train_bn=self.train_bn)
        # Stage 3
        outputs = self._conv_block(outputs, 3, [128, 128, 512], stage=3, block='a', train_bn=self.train_bn)
        outputs = self._identity_block(outputs, 3, [128, 128, 512], stage=3, block='b', train_bn=self.train_bn)
        outputs = self._identity_block(outputs, 3, [128, 128, 512], stage=3, block='c', train_bn=self.train_bn)
        C3 = outputs = self._identity_block(outputs, 3, [128, 128, 512], stage=3, block='d', train_bn=self.train_bn)
        # Stage 4
        outputs = self._conv_block(outputs, 3, [256, 256, 1024], stage=4, block='a', train_bn=self.train_bn)
        block_count = {"resnet50": 5, "resnet101": 22}[self.architecture]
        for i in range(block_count):
            outputs = self._identity_block(outputs, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=self.train_bn)
        C4 = outputs
        # Stage 5
        if self.stage5:
            outputs = self._conv_block(outputs, 3, [512, 512, 2048], stage=5, block='a', train_bn=self.train_bn)
            outputs = self._identity_block(outputs, 3, [512, 512, 2048], stage=5, block='b', train_bn=self.train_bn)
            C5 = outputs = self._identity_block(outputs, 3, [512, 512, 2048], stage=5, block='c', train_bn=self.train_bn)
        else:
            C5 = None
        #return tf.keras.Model(inputs = inputs, outputs = [C1, C2, C3, C4, C5] if self.stage5 else C4)
        return [C1, C2, C3, C4, C5] if self.stage5 else C4
