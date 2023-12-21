import numpy as np
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json
import pandas as pd

maxlen = 500
padding = 'post'
truncating = 'post'

model = keras.models.load_model('./Machine Learning/model.h5')

job_label = pd.read_csv("./Machine Learning/Dataset/data_capstone.csv", delimiter=";")
job_label = job_label['job_title']
job_label = job_label[:425]


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
    predicted_label = np.argsort(predictions, axis=1)[:, ::-1][0, :3]
    label = [job_label[index] for index in predicted_label]
    return label


degree_input, job_input, key_input = process_input(['Sci&Tech'], ['1'], ['Java'])
predicted_labels = predict(degree_input, job_input, key_input)
print("Predicted Labels:", predicted_labels)

