from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from ocular import ocular
from ocular.lipstick import Lipstick


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=255)
    parser.add_argument('--g', type=int, default=0)
    parser.add_argument('--b', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.5, help='opacity of overlay [default: 0.5]')
    args = parser.parse_args()

    # load a set of glasses
    lipstick = Lipstick()

    # start video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        faces = ocular.get_facial_keypoints_from_frame(frame)
        for (i, face) in enumerate(faces):
            coords = lipstick.place_lipstick(face)
            overlay = frame.copy()
            cv2.fillPoly(overlay, np.int_([coords]), (args.b, args.g, args.r))
            cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0, frame)

            for (x, y) in face['keypoints']:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            (x, y, w, h) = face['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
