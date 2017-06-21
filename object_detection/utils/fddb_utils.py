#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# utils for writing FDDB txt
# author: Xudong Sun

import numpy as np

class FDDB:
    def __init__(self, write_target, score_thresh=0.5):
        self._write_target = write_target
        self._score_thresh = score_thresh
    def __enter__(self):
        self._file = open(self._write_target, 'w')
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()

    def write(self, name_tag, boxes, scores, image_width, image_height):
        inds = np.where(scores > self._score_thresh)[0]
        boxes, scores = boxes[inds], scores[inds]
        print>>self._file, name_tag
        print>>self._file, len(inds)
        for (y1, x1, y2, x2), score in zip(boxes, scores):
            x1, y1, x2, y2 = int(round(x1 * image_width)), int(round(y1 * image_height)), int(round(x2 * image_width)), int(round(y2 * image_height))
            print>>self._file, x1, y1, x2-x1, y2-y1, score


