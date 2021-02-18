from flask import Flask, redirect, url_for
import os
from flask_cors import CORS, cross_origin
from mangler import *
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    getpicture()
    return redirect(url_for('static', filename='output.jpg'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)