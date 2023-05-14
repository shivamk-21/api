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
    print(request)
    print(request.get_json()['image'])
    print(request.get())
    image = request.get(request.get_json()['image'])
    # Send prediction request
    resp = predict_class(image)
    return jsonify({
        "class":resp[0],
        "probability" : resp[1],
        "scores" : resp[2]
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)