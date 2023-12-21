from os import getcwd

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import json

from One_Hot import all_one_hot
from Read_Data import read_file
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

# initiate variables
cwd = getcwd()
path = f"./Dataset/data_capstone.csv"  # path for the dataset
train_size = 0.9
oov = 'oov'
padding = 'post'
truncating = 'post'
maxlen = 500
vocab_size = 400

# get each column
job_title = read_file(path, 'job_title', True)
degree = read_file(path, 'degree', True)
key_skill = read_file(path, 'key_skills', True)
job_experience = read_file(path, 'job_experience', True)

key_token = Tokenizer(oov_token=oov, num_words=vocab_size)
key_token.fit_on_texts(key_skill)
key_seq = key_token.texts_to_sequences(key_skill)
key_pad = keras.preprocessing.sequence.pad_sequences(key_seq,
                                                     maxlen=maxlen,
                                                     padding=padding,
                                                     truncating=truncating)

job_token = Tokenizer(oov_token=oov, num_words=vocab_size)
job_token.fit_on_texts(job_experience)
job_seq = job_token.texts_to_sequences(job_experience)
job_pad = keras.preprocessing.sequence.pad_sequences(job_seq,
                                                     maxlen=maxlen,
                                                     padding=padding,
                                                     truncating=truncating)

degree_token = Tokenizer(oov_token=oov, num_words=vocab_size)
degree_token.fit_on_texts(degree)
degree_seq = degree_token.texts_to_sequences(degree)
degree_pad = keras.preprocessing.sequence.pad_sequences(degree_seq,
                                                        maxlen=maxlen,
                                                        padding=padding,
                                                        truncating=truncating)

job_title_hot = all_one_hot(job_title)


def create_model():
    degree_input = keras.Input(
        shape=(None,),
        name='degree'
    )

    job_input = keras.Input(
        shape=(None,),
        name='job'
    )

    key_input = keras.Input(
        shape=(None,),
        name='key'
    )

    degree_feature = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=maxlen
    )(degree_input)

    job_feature = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=maxlen
    )(job_input)

    key_feature = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=maxlen
    )(key_input)

    degree_feature = keras.layers.Bidirectional(keras.layers.LSTM(
        128
    ))(degree_feature)

    job_feature = keras.layers.Bidirectional(keras.layers.LSTM(
        128
    ))(job_feature)

    key_feature = keras.layers.Bidirectional(keras.layers.LSTM(
        128
    ))(key_feature)

    degree_feature = keras.layers.Flatten()(degree_feature)
    job_feature = keras.layers.Flatten()(job_feature)
    key_feature = keras.layers.Flatten()(key_feature)

    x = keras.layers.concatenate([
        degree_feature,
        job_feature,
        key_feature,
    ])

    job_title_pred = keras.layers.Dense(425, 'softmax', name='job_output')(x)

    model = keras.Model(
        inputs=[degree_input, job_input, key_input],
        outputs=job_title_pred
    )

    return model


def save_tokenizer(tokenizer, file_path):
    tokenizer_json = tokenizer.to_json()
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


save_tokenizer(degree_token, 'degree_token.json')
save_tokenizer(job_token, 'job_token.json')
save_tokenizer(key_token, 'key_token.json')

# model = create_model()
# model.compile(
#     optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
#     loss=keras.losses.categorical_crossentropy,
#     metrics=['accuracy']
# )
#
# model.fit(
#     {
#         'degree': degree_pad,
#         'job': job_pad,
#         'key': key_pad,
#     },
#     {
#         'job_output': job_title_hot
#     },
#     32,
#     100
# )
#
# path = os.path.join(getcwd(), str(1))
#
# keras.models.save_model(
#     model,
#     path,
#     overwrite=True,
#     save_format=None,
#     signatures=None,
#     options=None
# )
#
# model.save('model.h5')
