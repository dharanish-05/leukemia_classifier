from __future__ import division, print_function
#import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template,redirect,url_for  
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)


UPLOAD_FOLDER = './static/uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CUSTOM_MODEL_PATH = './leukemia_model/leukemia_model2.h5'  # Update with your actual model path

def allowed_file(filename):
    filename=filename.split('.')
    return filename[1].lower() in ALLOWED_EXTENSIONS

def predict_image_class(img_path):
    custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
    target_size=(224,224)
    img = cv2.imread(img_path)  # Update with your image size
    img= cv2.resize(img,target_size)
    img = np.expand_dims(img, axis=0)
    # img_array = preprocess_input(img_array)

    predictions = custom_model.predict(img)
    
    # Implement your own logic for decoding predictions based on your model's output
    # Example: Get the class with the highest probability
    decoded_predictions = np.argmax(predictions, axis=1)
    # print(decoded_predictions)
    result_dic={0:"Benign",1:"Early",2:"Pre",3:"Pro"}

    return result_dic[decoded_predictions[0]]  # Replace this with your own logic

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/upload', methods = ['GET', 'POST'])
def classifier():
    if 'file' not in request.files:
        return render_template('classifier.html', 'No file part')

    file = request.files['file']
    # print(file.filename)
    if file.filename == '':
        return render_template('classifier.html','No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # print(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # print(filepath)
        prediction = predict_image_class(filepath)

        return redirect(url_for('result', filename=filename, prediction=prediction))

    return render_template('classifier.html','Invalid Format')

@app.route('/classifier', methods = ['GET', 'POST'])
def classify():
    return render_template("classifier.html")

@app.route('/result')
def result():
    # Get the parameters from the URL
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')

    return render_template('result.html', filename=filename, prediction=prediction)


@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")

    
if __name__ == "__main__":
    app.run(debug=True)
