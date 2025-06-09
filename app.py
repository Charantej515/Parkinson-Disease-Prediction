from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("C:/Users/Charan Tej S/Desktop/Charan/Final Year Project/UI parkinson_DL/VGG16_best_model.h5")

# Image size (same as model input size)
IMAGE_SIZE = (128, 128)

# Function to predict image
def predict_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "Parkinson" if prediction > 0.5 else "Healthy"
    return result

# Homepage route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file is uploaded
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Ensure upload directory exists
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        # Secure the filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Make a prediction
        prediction = predict_image(file_path)

        return render_template("index.html", prediction=prediction, image_path=file_path)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
