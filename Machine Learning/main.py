import os
import json
from keras.preprocessing.text import tokenizer_from_json
from tensorflow import keras
from flask import Flask, request, jsonify
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 500
padding = 'post'
truncating = 'post'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

job_label = pd.read_csv("./Machine Learning/Dataset/data_capstone.csv", delimiter=";")
job_label = job_label['job_title']
job_label = job_label[:425]

model = keras.models.load_model("./Machine Learning/model.h5")


def load_tokenizer(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer)
    return tokenizer


degree_token = load_tokenizer("./Machine Learning/degree_token.json")
job_token = load_tokenizer("./Machine Learning/job_token.json")
key_token = load_tokenizer("./Machine Learning/key_token.json")


def process_input(degree, job_exp, key_skill):
    input_degree_seq = degree_token.texts_to_sequences(degree)
    input_degree_pad = pad_sequences(input_degree_seq, maxlen=maxlen, padding=padding, truncating=truncating)
    input_job_seq = job_token.texts_to_sequences(job_exp)
    input_job_pad = pad_sequences(input_job_seq, maxlen=maxlen, padding=padding, truncating=truncating)
    input_key_seq = key_token.texts_to_sequences(key_skill)
    input_key_pad = pad_sequences(input_key_seq, maxlen=maxlen, padding=padding, truncating=truncating)
    return input_degree_pad, input_job_pad, input_key_pad


def predict(input_degree, input_job, input_key):
    predictions = model.predict(
        {
            'degree': input_degree,
            'job': input_job,
            'key': input_key,
        }
    )
    predicted_label = np.argmax(predictions, axis=1)
    label = [job_label[i] for i in predicted_label]
    return label


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "No file provided"})

        try:
            form_data = json.load(file)

            degree = str(form_data['degree'])
            job = str(form_data['job'])
            key = str(form_data['key'])

            if None in [degree, job, key]:
                return jsonify({"error": "Missing required form data"})

            degree_pad, job_pad, key_pad = process_input([degree], [job], [key])
            label = predict(degree_pad, job_pad, key_pad)

            return jsonify({"label": label})
        except Exception as e:
            return jsonify({"error": str(e)})
    return jsonify({'get': 'clear'})


if __name__ == "__main__":
    app.run(debug=True)
