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
    glasses_type = data.get('type')
    width_factor = float(data.get('scale'))
    frame = read_base64_image(base64_str)  # np array

    glasses = Glasses()
    glasses.load_pieces_from_directory(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../img/%s' % glasses_type))
    faces = ocular.get_facial_keypoints_from_frame(frame)
    
    for (i, face) in enumerate(faces):
        pieces = glasses.place_glasses(face, width_factor=width_factor)
        for piece in pieces.itervalues():
            x, y, w, h= piece['loc']
            alpha_s = piece['data'][:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in xrange(0, 3):
                # render images as (y, x, h, w) as openCV does 
                # this means i need to swapaxes
                try:
                    frame_slice = np.round((alpha_s * piece['data'][:, :, c] + 
                                            alpha_l * frame[y:y+h, x:x+w, c]))
                    frame[y:y+h, x:x+w, c] = frame_slice.astype(np.uint8)
                except:
                    pass

    rendering = write_base64_image(frame)
    response = json.dumps({'image': rendering})
    return Response(
        response=response, 
        status=200, 
        mimetype='appliation/json')


@app.route('/lipstick', methods=['POST'])
def lipstick():
    data = json.loads(request.data)
    base64_str = data.get('image')
    red_factor = data.get('r')
    green_factor = data.get('g')
    blue_factor = data.get('b')
    alpha = float(data.get('alpha'))
    frame = read_base64_image(base64_str)  # np array

    # load a set of glasses
    lipstick = Lipstick()
    faces = ocular.get_facial_keypoints_from_frame(frame)

    for (i, face) in enumerate(faces):
        coords = lipstick.place_lipstick(face)
        overlay = frame.copy()
        cv2.fillPoly(overlay, np.int_([coords]), (blue_factor, green_factor, red_factor))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    rendering = write_base64_image(frame)
    response = json.dumps({'image': rendering})
    return Response(
        response=response, 
        status=200, 
        mimetype='appliation/json')
