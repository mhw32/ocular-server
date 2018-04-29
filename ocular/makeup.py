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
    def place_lipstick(self, face):
        lips_path = [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60, 48, 60, 
                     67, 66, 65, 64, 54, 55, 56, 57, 58, 59, 48]
        lips_path = face['keypoints'][lips_path]
        return lips_path.reshape((-1, 1, 2))
