from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import math
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
        # left_earpiece = self._place_left_earpiece(face, left_eyepiece)
        # right_earpiece = self._place_right_earpiece(face, right_eyepiece)
        center_piece = self._place_center_piece(face, left_eyepiece, right_eyepiece)
        return {
            'left_eyepiece': left_eyepiece,
            'right_eyepiece': right_eyepiece,
            # 'left_earpiece': left_earpiece,
            # 'right_earpiece': right_earpiece,
            'center_piece': center_piece,
        }

    def _place_left_eyepiece(self, face, width_factor=1.75):
        assert self.left_eyepiece is not None
        eyepiece = copy.deepcopy(self.left_eyepiece)
        eye = face['keypoints'][36:42]
        # compute center of eye
        eye_center = np.round(np.mean(eye, axis=0)).astype(np.int)
        eye_width = np.max(eye[:, 0]) - np.min(eye[:, 0])

        # compute size with assumption of the width of a glasses is 
        # 2x the width of eye. This should be make tunable
        old_eyepiece_width, old_eyepiece_height = eyepiece.size
        new_eyepiece_width = int(np.round(eye_width * width_factor))
        new_eyepiece_height = int(np.round(new_eyepiece_width / float(old_eyepiece_width) *  old_eyepiece_height))

        # things are easier if sizes are all old
        if new_eyepiece_width % 2 == 0:
            new_eyepiece_width += 1
        if new_eyepiece_height % 2 == 0:
            new_eyepiece_height += 1

        # reshape eyepiece
        eyepiece = eyepiece.resize((new_eyepiece_width, new_eyepiece_height))
        eyepiece = np.asarray(eyepiece)

        return {
            'data': eyepiece,
            'center': eye_center,
            'loc': (int(eye_center[1] - (new_eyepiece_height - 1) / 2), 
                    int(eye_center[0] - (new_eyepiece_width - 1) / 2),
                    new_eyepiece_height, new_eyepiece_width)
        }

    def _place_right_eyepiece(self, face, width_factor=1.75):
        assert self.right_earpiece is not None
        eyepiece = copy.deepcopy(self.right_eyepiece)
        eye = face['keypoints'][42:48]
        # compute center of eye
        eye_center = np.round(np.mean(eye, axis=0)).astype(np.int)
        eye_width = np.max(eye[:, 0]) - np.min(eye[:, 0])

        # compute size with assumption of the width of a glasses is 
        # 2x the width of eye. This should be make tunable
        old_eyepiece_width, old_eyepiece_height = eyepiece.size
        new_eyepiece_width = int(np.round(eye_width * width_factor))
        new_eyepiece_height = int(np.round(new_eyepiece_width / float(old_eyepiece_width) *  old_eyepiece_height))

        # things are easier if sizes are all old
        if new_eyepiece_width % 2 == 0:
            new_eyepiece_width += 1
        if new_eyepiece_height % 2 == 0:
            new_eyepiece_height += 1

        # reshape eyepiece
        eyepiece = eyepiece.resize((new_eyepiece_width, new_eyepiece_height))
        eyepiece = np.asarray(eyepiece)

        return {
            'data': eyepiece,
            'center': eye_center[::-1],
            'loc': (int(eye_center[1] - (new_eyepiece_height - 1) / 2),
                    int(eye_center[0] - (new_eyepiece_width - 1) / 2), 
                    new_eyepiece_height, new_eyepiece_width)
        }

    # def _place_left_earpiece(self, face, left_eyepiece):
    #     assert self.left_earpiece is not None

    # def _place_right_earpiece(self, face, right_eyepiece):
    #     assert self.right_earpiece is not None

    def _place_center_piece(self, face, left_eyepiece, right_eyepiece):
        assert self.center_piece is not None
        center_piece = self.center_piece
        ly, lx, lh, lw = left_eyepiece['loc']
        ry, rx, rh, rw = right_eyepiece['loc']

        right_pivot = [int(ly + lh / 2), int(lx + lw)]
        left_pivot = [int(ry + rh / 2), int(rx)]
        angle = math.atan2(right_pivot[0] - left_pivot[0], float(right_pivot[1] - left_pivot[1]))

        old_cpiece_width, old_cpiece_height = center_piece.size
        new_cpiece_width = int((rx - (lx + lw)) / abs(math.cos(angle)))
        new_cpiece_height = int(np.round(new_cpiece_width / float(old_cpiece_width) *  old_cpiece_height))

        cpiece = center_piece.resize((new_cpiece_width, new_cpiece_height))
        cpiece = np.asarray(cpiece)
        return {
            'data': cpiece,
            'loc': (right_pivot[0], right_pivot[1], 
                    new_cpiece_height, new_cpiece_width),
        }
