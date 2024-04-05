from __future__ import division, print_function
#import sys
import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template,redirect,url_for  
from werkzeug.utils import secure_filename
from groq import Groq


app = Flask(__name__)


UPLOAD_FOLDER = './static/uploaded_images' #input file storage path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} #allowed file formats

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
CUSTOM_MODEL_PATH = './leukemia_model/leukemia_model2.h5'  # trained and saved model path

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
    print(decoded_predictions)
    result_dic={0:"Benign",1:"Early",2:"Pre",3:"Pro"}

    return result_dic[decoded_predictions[0]]  

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/upload', methods = ['GET', 'POST']) #logic for image file handling
def classifier():
    if 'file' not in request.files:
        return render_template('classifier.html', 'No file part')

    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        return render_template('classifier.html','No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(filepath)
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

@app.route('/chat_with_llm', methods = ['GET', 'POST'])
def chat():
    return render_template("llm_process.html")

@app.route('/process_text', methods=['POST'])
def process_text():
    user_input = request.form['user_input']
    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
       messages=[
        {
            "role": "system",
            "content": "you are trained ai medical assistant.You are only used to solve querries of the user regarding the leukemia cancer and its causes, preventions or any related information.you are not allowed to respond to any other topic"
        },
        {
            "role": "user",
            "content": user_input
        }
    ],
    temperature=0.5,
    max_tokens=256,
    top_p=1,
    stream=False,
    stop=None,
    )

    answer = completion.choices[0].message.content
    return render_template('llm_process.html', user_input=user_input, answer=answer)



@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")

    
if __name__ == "__main__":
    app.run(debug=True)