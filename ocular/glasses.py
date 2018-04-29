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
        # NOTE: user can overload this function.
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

    def place_glasses(self, face, width_factor=1.75):
        angle = self._compute_angle(face)
        rotation = self._compute_rotation(face)
        # NOTE: only call after <load_pieces_from_directory>
        left_eyepiece = self._place_left_eyepiece(face, angle, width_factor=width_factor)
        right_eyepiece = self._place_right_eyepiece(face, angle, width_factor=width_factor)
        left_earpiece = self._place_left_earpiece(face, angle)
        right_earpiece = self._place_right_earpiece(face, angle)
        center_piece = self._place_center_piece(face, angle)
        return {
            'left_eyepiece': left_eyepiece,
            'right_eyepiece': right_eyepiece,
            'left_earpiece': left_earpiece,
            'right_earpiece': right_earpiece,
            'center_piece': center_piece,
        }

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

    def _place_left_earpiece(self, face, angle):
        assert self.left_earpiece is not None
        earpiece = copy.deepcopy(self.left_earpiece)
        eye = face['keypoints'][36]
        ear = face['keypoints'][0]
        diff = eye - ear
        angle = np.arctan2(diff[1], diff[0])
        earpiece = earpiece.rotate(-(90 + math.degrees(angle)), expand=True)
        old_width, old_height = earpiece.size
        new_width = int(np.round((eye[0] - ear[0])))
        new_height = int(np.round(new_width / float(old_width) *  old_height))
        earpiece = earpiece.resize((new_width, new_height))
        earpiece = np.asarray(earpiece)
        return {
            'data': earpiece,
            'loc': (ear[0], ear[1], new_width, new_height),
        }

    def _place_right_earpiece(self, face, angle):
        assert self.right_earpiece is not None
        earpiece = copy.deepcopy(self.right_earpiece)
        eye = face['keypoints'][45]
        ear = face['keypoints'][16]
        diff = ear - eye
        angle = np.arctan2(diff[1], diff[0])
        earpiece = earpiece.rotate(-270 - math.degrees(angle), expand=True)
        old_width, old_height = earpiece.size
        new_width = int(np.round((ear[0] - eye[0])))
        new_height = int(np.round(new_width / float(old_width) *  old_height))
        earpiece = earpiece.resize((new_width, new_height))
        earpiece = np.asarray(earpiece)
        return {
            'data': earpiece,
            'loc': (eye[0], eye[1], new_width, new_height),
        }

    def _place_center_piece(self, face, angle):
        assert self.center_piece is not None
        center_piece = copy.deepcopy(self.center_piece)
        old_cpiece_width, old_cpiece_height = center_piece.size
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
