from app import app
from flask import request, Response

@app.route('/')
@app.route('/index')
def index():
    return "Ocular Server"
