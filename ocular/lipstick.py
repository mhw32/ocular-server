from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import PIL
import copy
import math
import numpy as np
from PIL import Image


class Lipstick(object):
    def _compute_rotation(self, face):
        left_face = face['keypoints'][0:4]
        right_face = face['keypoints'][14:18]
        middle_face = face['keypoints'][27:35]

        rotation = []
        for i in xrange(len(left_face)):
            rotation.append(np.linalg.norm(middle_face - right_face[i], axis=1) / 
                            np.linalg.norm(middle_face - left_face[i], axis=1))
        rotation = np.concatenate(rotation)
        return np.mean(rotation)

    def _compute_angle(self, face):
        left_eye = face['keypoints'][36:42]
        right_eye = face['keypoints'][42:48]
        diff_eye = right_eye - left_eye
        angle_eye = 2 * math.pi - np.arctan2(diff_eye[:, 1], diff_eye[:, 0])
        angle_eye = np.mean(angle_eye)
        return angle_eye

    def place_lipstick(self, face):
        lips_path = [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60, 48, 60, 
                     67, 66, 65, 64, 54, 55, 56, 57, 58, 59, 48]
        lips_path = face['keypoints'][lips_path]
        return lips_path.reshape((-1, 1, 2))
