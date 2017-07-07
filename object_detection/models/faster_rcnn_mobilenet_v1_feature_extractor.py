"""MobileNet V1 Faster R-CNN implementation.
"""

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import mobilenet_v1
slim = tf.contrib.slim

class FasterRCNNMobileNetV1FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN MobileNet V1 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               reuse_weights=None,
               weight_decay=0.0,
               depth_multiplier=1.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._depth_multiplier = depth_multiplier
    super(FasterRCNNResnetV1FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN MobileNet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    channel_means = [123.68, 116.779, 103.939]
    return resized_inputs - [[channel_means]]

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(
                          weight_decay=self._weight_decay)):
        with slim.arg_scope([slim.batch_norm], is_training=False):
          with tf.variable_scope('MobilenetV1', 
                                 reuse=self._reuse_weights) as scope:
            rpn_feature_map, _ = mobilenet_v1.mobilenet_v1_base(
                  preprocessed_inputs,
                  final_endpoint='Conv2d_13_pointwise',
                  min_depth=self._first_stage_features_stride,
                  depth_multiplier=self._depth_multiplier,
                  scope=scope)
    return rpn_feature_map

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    return proposal_feature_maps


