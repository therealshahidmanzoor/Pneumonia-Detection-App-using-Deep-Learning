from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__, static_url_path='/static')
import os

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = load_model('model/best_model.h5')

def predict_pneumonia(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    return prediction[0][0]  # Assuming a binary classification, adjust as needed

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict_pneumonia(file_path)
            return render_template('result.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
