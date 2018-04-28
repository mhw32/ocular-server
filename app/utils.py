import base64
import numpy as np


def read_base64_image(base64_str):
    """Converts base64 string to numpy array.
    
    @param base64_str: base64 encoded string
    @return: numpy array RGB (H x W x C)
    """
    return np.fromstring(base64.b64decode(base64_str), np.uint8)


def write_base64_image(image):
    """Inverse of read_base64_image."""
    return base64.b64encode(np.tostring(image))
