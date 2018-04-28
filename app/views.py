from app import app
from flask import jsonify
from flask import request, Response
from utils import read_base64_image
from utils import write_base64_image

import os
import json
import numpy as np
import sys; sys.path.append('..')
from ocular import ocular
from ocular.glasses import Glasses


@app.route('/')
@app.route('/index')
def index():
    return "Ocular Server"


@app.route('/render', methods=['POST'])
def render():
    data = json.loads(request.data)
    base64_str = data.get('image')
    frame = read_base64_image(base64_str)  # np array

    dummy = Glasses()
    dummy.load_pieces_from_directory(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../img/dummy'))
    faces = ocular.get_facial_keypoints_from_frame(frame)
    
    for (i, face) in enumerate(faces):
        pieces = dummy.place_glasses(face)
        for piece in pieces.itervalues():
            x, y, w, h= piece['loc']
            alpha_s = piece['data'][:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in xrange(0, 3):
                # render images as (y, x, h, w) as openCV does 
                # this means i need to swapaxes
                frame_slice = np.round((alpha_s * piece['data'][:, :, c] + 
                                        alpha_l * frame[y:y+h, x:x+w, c]))
                frame[y:y+h, x:x+w, c] = frame_slice.astype(np.uint8)

    rendering = write_base64_image(frame)
    response = json.dumps({'image': rendering})
    return Response(
        response=response, 
        status=200, 
        mimetype='appliation/json')
