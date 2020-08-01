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
    def __init__(self, proposal_count, nms_threshold, configs = None, **kwargs):
        self.nms_threshold = nms_threshold
        self.proposal_count = proposal_count
        self.configs = configs
        super(ProposalLayer, self).__init_()

    def call(self, inputs):
        """
        Inputs:
            - classes : Tensor
                In shape of [batch, num_anchors, 2] for background and foreground probabilities
            - bboxes : Tensor
                In shape of [batch, num_anchors, 4] for proposed bboxes
            - anchors : Tensor
                In shape of [batch, num_anchors, [x1, y1, x2, y2]]
        """
        # classes
        classes = inputs['classes']
        # Bounding box refinement deltas to anchors
        bboxes = inputs['bboxes']
        bboxes = bboxes * np.reshape(self.configs.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs['anchors']

        # Trim to top anchors by score
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, anchors.shape[1])
        indices = tf.math.top_k(classes, pre_nms_limit, sorted = True, name = "top_anchors").indices
        classes = tf.slice(classes,  begin = [0, ])
