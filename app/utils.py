import cv2
import base64
import numpy as np


def read_base64_image(base64_str):
    """Converts base64 string to numpy array.
    
    @param base64_str: base64 encoded string
    @return: numpy array RGB (H x W x C)
    """
    nparr = np.fromstring(base64.b64decode(base64_str), np.uint8)
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def write_base64_image(image):
    """Inverse of read_base64_image."""
    return base64.b64encode(image.tostring())
