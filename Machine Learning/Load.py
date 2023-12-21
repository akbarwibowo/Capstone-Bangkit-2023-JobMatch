import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 500
padding = 'post'
truncating = 'post'

model = keras.models.load_model("model.h5")


def process_input(degree, job_exp, key_skill):
    degree_token = Tokenizer()
    input_degree_seq = degree_token.texts_to_sequences(degree)
    input_degree_pad = pad_sequences(input_degree_seq, maxlen=maxlen, padding=padding, truncating=truncating)

    job_token = Tokenizer()
    input_job_seq = job_token.texts_to_sequences(job_exp)
    input_job_pad = pad_sequences(input_job_seq, maxlen=maxlen, padding=padding, truncating=truncating)

    key_token = Tokenizer()
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
    pred = predictions[0]
    label = pred[0:3]
    return label


