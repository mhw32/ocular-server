from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import dlib

predictor_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    'data/shape_predictor_68_face_landmarks.dat')
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
