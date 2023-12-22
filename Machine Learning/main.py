import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
import pandas as pd

model = keras.models.load_model("model.h5")

def get_job_recommendation(user_data):
    # Process user data (adjust as needed based on your model's input requirements)
    # For example, convert user input to the format expected by your model

    # Perform inference using the loaded model
    try:
        predictions = model.predict(user_data)
        # You might need additional post-processing based on your model's output
        return predictions.tolist()  # Convert predictions to a JSON-serializable format
    except Exception as e:
        return {'error': str(e)}

app = Flask(__name__)

@app.route("/", methods=["POST"])
@app.route("/", methods=["POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided"})

    # Process the file as needed

    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)
