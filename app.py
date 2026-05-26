import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = load_model("best_resnet50_model.h5")


CLASS_NAMES = [
    "Anthracnose",
    "Black Soothy Mold",
    "healthy"
]

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            img = prepare_image(img_path)
            preds = model.predict(img)[0]

            class_id = np.argmax(preds)
            prediction = CLASS_NAMES[class_id]
            confidence = round(float(preds[class_id]) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path
    )

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
