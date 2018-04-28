from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from PIL import Image


class Glasses(object):
    def __init__(self):
        self.left_earpiece = None
        self.right_earpiece = None
        self.left_eyepiece = None
        self.right_eyepiece = None
        self.center_piece = None

    def load_pieces_from_directory(self, dir):
        # Given some directory with properly named PNG files, 
        # load them to the glass.
        # 
        # We expect the following directory structure:
        # dir/
        #   dir/left_earpiece.png
        #   dir/right_earpiece.png
        #   dir/left_eyepiece.png
        #   dir/right_eyepiece.png
        #   dir/center_piece.png
        left_earpiece = Image.open(os.path.join(dir, 'left_earpiece.png'))
        right_earpiece = Image.open(os.path.join(dir, 'right_earpiece.png'))
        left_eyepiece = Image.open(os.path.join(dir, 'left_eyepiece.png'))
        right_eyepiece = Image.open(os.path.join(dir, 'right_eyepiece.png'))
        center_piece = Image.open(os.path.join(dir, 'center_piece.png'))

        self.left_earpiece = np.asarray(left_earpiece)
        self.right_earpiece = np.asarray(right_earpiece)
        self.left_eyepiece = np.asarray(left_eyepiece)
        self.left_eyepiece = np.asarray(left_eyepiece)
        self.center_piece = np.asarray(center_piece)
