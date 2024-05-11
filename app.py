from __future__ import division, print_function
import requests
import sys
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

pexels_api_key = os.getenv('PEXELS_API_KEY')
app = Flask(__name__)

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        # Process prediction result
        pred_class = decode_predictions(preds, top=1)[0][0]
        result = {"label": pred_class[1], "probability": float(pred_class[2])}
        return jsonify(result)
    return None


#search function
@app.route('/form', methods = ['POST', 'GET']) 
def form():
    if request.method == 'POST':
        user_input = request.form['name']
        url = f'http://api.pexels.com/videos/search?query={user_input}'
        headers = {'Authorization': pexels_api_key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            video_data = response.json()
            return render_template('videos.html', videos=video_data.get('videos', []))
        else:
            error_message = f"Error: {response.status_code}"
            return render_template('error.html', error_message=error_message)
    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)

