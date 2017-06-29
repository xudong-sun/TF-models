#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# testing demo
#
# run examples:
# "draw"
# python test.py --task_type=draw --ckpt_path=ckpt/train/wider/frozen_inference_graph.pb --test_images_root=test_images \
#   --test_images=test1.jpg,test3.jpg --labels_path=data/face_label_map.pbtxt
# "fddb"
# python test.py --task_type=fddb --ckpt_path=ckpt/train/wider/frozen_inference_graph.pb

import numpy as np
import os
import sys
import time
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from PIL import Image

from utils import label_map_util

flags = tf.app.flags

# task
flags.DEFINE_string('task_type', 'none', 'task type, "none", "draw", "fddb"')

# checkpoint
flags.DEFINE_string('ckpt_path', '', 'path to frozen detection graph')

# detection
flags.DEFINE_integer('num_classes', 1, 'number of classes in detection')
flags.DEFINE_float('conf_thresh', 0.5, 'confidence threshold')

# if task_type is "none" or "draw"
flags.DEFINE_string('test_images_root', '', 'path to test images')
flags.DEFINE_string('test_images', '', 'test images, separated by comma')

# if task type is "draw"
flags.DEFINE_string('labels_path', '', 'path to label file')

# if task type is "fddb"
flags.DEFINE_integer('fddb_fold', 7, 'FDDB fold to test')
flags.DEFINE_string('fddb_root', 'data/datasets/FDDB', 'FDDB root directory')
flags.DEFINE_string('fddb_output', 'ckpt/save/fddb_result.txt', 'path to output FDDB result file')

FLAGS = flags.FLAGS

ALL_TASK_TYPE = {'none', 'draw', 'fddb'}

def load_labelmap():
    # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. 
    # Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(FLAGS.labels_path)
    # categories is a list of dict: {id: int, name: str}, len(categories) == num_classes
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=FLAGS.num_classes, use_display_name=True)
    # category_index is a dict: {id: {id: int, name: str}}, id in [1, num_classes]
    category_index = label_map_util.create_category_index(categories)
    return category_index

class Detection:
    def __init__(self, ckpt_path):
        self._detection_graph = self._load_model(ckpt_path)
        with self._detection_graph.as_default():
            # input/output tensors
            self._image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self._boxes_tensor = self._detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self._scores_tensor = self._detection_graph.get_tensor_by_name('detection_scores:0')
            self._classes_tensor = self._detection_graph.get_tensor_by_name('detection_classes:0')
            self._num_detections_tensor = self._detection_graph.get_tensor_by_name('num_detections:0')
    def __enter__(self):
        self._sess = tf.Session(graph=self._detection_graph)
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._sess.close()

    def _load_model(self, ckpt_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def im_detect(self, image, image_path='', verbose=True):
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self._sess.run(
            [self._boxes_tensor, self._scores_tensor, self._classes_tensor, self._num_detections_tensor],
            feed_dict={self._image_tensor: image})
        time_elapsed = time.time() - start_time
        if verbose: print '{} time: {:.3f}s'.format(image_path, time_elapsed)
        return boxes, scores, classes, num_detections

def read_image_and_preprocess(image_path):
    '''read from image_path
    returns: image_np [H,W,C], image_np_expanded: [1,H,W,C]
    '''
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    im_width, im_height = image.size
    try:
        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    except: # single-channel image
        image_np = np.array(image.getdata()).reshape((im_height, im_width)).astype(np.uint8)
        image_np = np.tile(image_np[:, :, np.newaxis], (1, 1, 3))
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np, image_np_expanded

def visualize(image, boxes, scores, classes, category_index):
    from utils import visualization_utils as vis_util
    from matplotlib import pyplot as plt
    vis_util.visualize_boxes_and_labels_on_image_array(image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, 
        use_normalized_coordinates=True, max_boxes_to_draw=None, min_score_thresh=FLAGS.conf_thresh)
    plt.imshow(image)
    plt.show()

def main(_):
    assert FLAGS.task_type in ALL_TASK_TYPE, 'Unrecognized task_type: {}'.format(FLAGS.task_type)
    
    if FLAGS.task_type == 'draw': category_index = load_labelmap()
    
    with Detection(FLAGS.ckpt_path) as detection:
        #warmup
        image = np.ones((1, 500, 500, 3), dtype=np.float32) * 128
        detection.im_detect(image, verbose=False)

        if FLAGS.task_type in ('none', 'draw'):
            for image_path in FLAGS.test_images.split(','):
                image_path = os.path.join(FLAGS.test_images_root, image_path)
                image_np, image_np_expanded = read_image_and_preprocess(image_path)
                boxes, scores, classes, num_detections = detection.im_detect(image_np_expanded, image_path=image_path)
                if FLAGS.task_type == 'draw': 
                    visualize(image_np, boxes, scores, classes, category_index)
        
        elif FLAGS.task_type == 'fddb':
            im_list_file = os.path.join(FLAGS.fddb_root, 'FDDB-folds', 'FDDB-fold-{:02d}.txt'.format(FLAGS.fddb_fold))
            from utils import fddb_utils
            with fddb_utils.FDDB(FLAGS.fddb_output, score_thresh=FLAGS.conf_thresh) as fddb:
                with open(im_list_file) as f:
                    for line in f:
                        filename_short = line.strip()
                        image_path = os.path.join(FLAGS.fddb_root, 'originalPics', filename_short + '.jpg')
                        image_np, image_np_expanded = read_image_and_preprocess(image_path)
                        boxes, scores, classes, num_detections = detection.im_detect(image_np_expanded, image_path=image_path)
                        im_height, im_width, _ = image_np.shape
                        fddb.write(filename_short, np.squeeze(boxes), np.squeeze(scores), im_width, im_height)

if __name__ == '__main__':
    tf.app.run()
