from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import PIL
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
        angle = self._compute_angle(face)
        # NOTE: only call after <load_pieces_from_directory>
        left_eyepiece = self._place_left_eyepiece(face, angle)
        right_eyepiece = self._place_right_eyepiece(face, angle)
        # left_earpiece = self._place_left_earpiece(face, left_eyepiece)
        # right_earpiece = self._place_right_earpiece(face, right_eyepiece)
        center_piece = self._place_center_piece(face, left_eyepiece, right_eyepiece, angle)
        return {
            'left_eyepiece': left_eyepiece,
            'right_eyepiece': right_eyepiece,
            # 'left_earpiece': left_earpiece,
            # 'right_earpiece': right_earpiece,
            'center_piece': center_piece,
        }

    def _compute_angle(self, face):
        left_eye = face['keypoints'][36:42]
        right_eye = face['keypoints'][42:48]
        diff_eye = right_eye - left_eye
        angle_eye = 2 * math.pi - np.arctan2(diff_eye[:, 1], diff_eye[:, 0])
        angle_eye = np.mean(angle_eye)
        return angle_eye

    def _place_left_eyepiece(self, face, angle, width_factor=1.75):
        assert self.left_eyepiece is not None
        eyepiece = copy.deepcopy(self.left_eyepiece)
        eye = face['keypoints'][36:42]  # (x, y)
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
        eyepiece = eyepiece.rotate(math.degrees(angle), expand=True)
        new_eyepiece_width, new_eyepiece_height = eyepiece.size
        eyepiece = np.asarray(eyepiece)

        return {
            'data': eyepiece,
            'center': eye_center,
            # (x, y, w, h) convention
            'loc': (int(eye_center[0] - (new_eyepiece_width - 1) / 2),
                    int(eye_center[1] - (new_eyepiece_height - 1) / 2), 
                    new_eyepiece_width, new_eyepiece_height)
        }

    def _place_right_eyepiece(self, face, angle, width_factor=1.75):
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
        eyepiece = eyepiece.rotate(math.degrees(angle), expand=True)
        new_eyepiece_width, new_eyepiece_height = eyepiece.size
        eyepiece = np.asarray(eyepiece)

        return {
            'data': eyepiece,
            'center': eye_center[::-1],
            'loc': (int(eye_center[0] - (new_eyepiece_width - 1) / 2), 
                    int(eye_center[1] - (new_eyepiece_height - 1) / 2),
                    new_eyepiece_width, new_eyepiece_height)
        }

    # def _place_left_earpiece(self, face, left_eyepiece):
    #     assert self.left_earpiece is not None

    # def _place_right_earpiece(self, face, right_eyepiece):
    #     assert self.right_earpiece is not None

    def _place_center_piece(self, face, left_eyepiece, right_eyepiece, angle):
        assert self.center_piece is not None
        center_piece = copy.deepcopy(self.center_piece)
        lx, ly, lw, lh = left_eyepiece['loc']
        rx, ry, rw, rh = right_eyepiece['loc']

        old_cpiece_width, old_cpiece_height = center_piece.size
        new_cpiece_width = int(rx - (lx + lw))
        new_cpiece_width = int((face['keypoints'][42][0] - face['keypoints'][39][0]) * 0.65)
        new_cpiece_height = int(np.round(new_cpiece_width / float(old_cpiece_width) *  old_cpiece_height))
        center_piece = center_piece.resize((new_cpiece_width, new_cpiece_height))
        center_piece = center_piece.rotate(math.degrees(angle), expand=True)
        new_cpiece_width, new_cpiece_height = center_piece.size
        center_piece = np.asarray(center_piece)
        
        # grab the nose (where we will center the bridge)
        ctrpt_x = int(face['keypoints'][27][0] - new_cpiece_width / 2)  # x, y
        ctrpt_y = int(face['keypoints'][27][1])

        return {
            'data': center_piece,
            'loc': (ctrpt_x, ctrpt_y,
                    new_cpiece_width, new_cpiece_height),
        }
