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
        # NOTE: user should overload this function.
        #
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
        self.left_earpiece = Image.open(os.path.join(dir, 'left_earpiece.png'))
        self.right_earpiece = Image.open(os.path.join(dir, 'right_earpiece.png'))
        self.left_eyepiece = Image.open(os.path.join(dir, 'left_eyepiece.png'))
        self.right_eyepiece = Image.open(os.path.join(dir, 'right_eyepiece.png'))
        self.center_piece = Image.open(os.path.join(dir, 'center_piece.png'))

    def place_glasses(self, face):
        # NOTE: only call after <load_pieces_from_directory>
        left_eyepiece = self._place_left_eyepiece(face)
        right_eyepiece = self._place_right_eyepiece(face)
        left_earpiece = self._place_left_earpiece(face, left_eyepiece)
        right_earpiece = self._place_right_earpiece(face, right_eyepiece)
        center_piece = self._place_center_piece(face, left_eyepiece, right_earpiece)
        return {
            'left_eyepiece': left_eyepiece,
            'right_eyepiece': right_eyepiece,
            'left_earpiece': left_earpiece,
            'right_earpiece': right_earpiece,
            'center_piece': center_piece,
        }

    def _place_left_eyepiece(self, face):
        assert self.left_earpiece is not None

    def _place_right_eyepiece(self, face):
        assert self.right_earpiece is not None

    def _place_left_earpiece(self, face, left_eyepiece):
        assert self.left_earpiece is not None

    def _place_right_earpiece(self, face, right_eyepiece):
        assert self.right_earpiece is not None

    def _place_center_piece(self, face, left_eyepiece, right_eyepiece):
        assert self.center_piece is not None

