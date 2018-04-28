from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from ocular import ocular


# start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    faces = ocular.get_facial_keypoints_from_frame(frame)
    for (i, face) in enumerate(faces):
        (x, y, w, h) = face['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y) in face['keypoints']:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
