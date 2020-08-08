from flask import Flask, request
import pandas as pd
import numpy as np
from flasgger import Swagger
import os
from werkzeug.utils import secure_filename

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

app=Flask(__name__)
Swagger(app)

# Load your trained model
model = load_model('final.h5')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Fire"
    else:
        preds = "Non fire"

    return preds


@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict_note():
    """Fire-detector
    This is using docstrings for specifications.
    ---
    parameters:
      - name: image
        in: query
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        return preds
    return None

if __name__=='__main__':
    app.run()