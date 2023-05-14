from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model import predict_class

app = Flask(__name__) # create a flask app
@app.route('/')
def index():
    return "App is Working"
@app.route('/predict', methods=['POST'])
def predict_request():
    # Get file and save it
    image = request.get_json()['image']
    plant = request.get_json()['plant']
    # Send prediction request
    resp = predict_class(image,plant)
    return jsonify({
        "class_name":resp[0],
        # "probability" : resp[1],
        # "scores" : resp[2]
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)