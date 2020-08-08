from flask import Flask, request, render_template
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image

app=Flask(__name__)

model = load_model('final.h5')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Fire-detected🔴"
    else:
        preds = "Fire-Not-detected"

    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict_fire():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        return preds
    return None

if __name__=='__main__':
    app.run(debug=True)