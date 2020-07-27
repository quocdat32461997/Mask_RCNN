"""
proposal.py - module to implement Proposal Layer using Non-Max-Suppression
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

class ProposalLayer(Layer):
    """
    ProposalLayer - a class to subsample/filter proposals
    from RPN layer that the task is done by using anchor scores
    and non-max-suppression
    """
    def __init__(self, nms_threshold, **kwargs):
        self.nms_threshold = nms_threshold
        super(ProposalLayer, self).__init_()

    def call(self, inputs):
        """
        Inputs:
            - rpn_probs : Tensor
                In shape of [batch, num_anchors, 2] for background and foreground probabilities
            - rpn_bbox : Tensor
                In shape of [batch, num_anchors, 4] for proposed bboxes 
            - anchors : Tensor
                In shape of
        """
