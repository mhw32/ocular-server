from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import dlib
import numpy as np

import imutils
from imutils import face_utils

from ocular import detector, predictor


def get_facial_keypoints_from_frame(frame):
    """Generate keypoints from frame.

    @param frame: openCV frame object
    @return: list of maps containing keypoints and bounding boxes
             for each face in the frame.

    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    face2keypts = []
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face2keypts.append({'keypoints': shape, 'bbox': (x, y, w, h)})

    return face2keypts
