#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# testing demo

import numpy as np
import os
import sys
import time
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/data/ObjectDetection/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

if __name__ == '__main__':
    # load model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. 
    # Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories is a list of dict: {id: int, name: str}, len(categories) == NUM_CLASSES
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    # category_index is a dict: {id: {id: int, name: str}}, id in [1, NUM_CLASSES]
    category_index = label_map_util.create_category_index(categories)

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # input/output tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
            classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections_tensor = detection_graph.get_tensor_by_name('num_detections:0')

            # warmup
            image = np.ones((1, 500, 500, 3), dtype=np.float32) * 128
            sess.run([boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor], feed_dict={image_tensor: image})

            # actual detection
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection
                start_time = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                    feed_dict={image_tensor: image_np_expanded})
                time_elapsed = time.time() - start_time
                print 'image: {}, time: {:.3f}s'.format(image_path, time_elapsed)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True)
                plt.imshow(image_np)
                plt.show()
     
