import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model/plant_disease_prediction_model.h5")
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Configure upload folder
upload_dir = os.path.join(working_dir, 'uploads')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
app.config['UPLOAD_FOLDER'] = upload_dir


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(file, target_size=(224, 224)):
    # Load the image
    img = Image.open(file)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, file, class_indices):
    preprocessed_img = load_and_preprocess_image(file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Render upload page with uploaded image
            return render_template('upload.html', image_file=filename)
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    prediction = predict_image_class(model, file_path, class_indices)
    return render_template('result.html', prediction=prediction, image_file=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
    localhost:5000
