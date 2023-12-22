import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import json
import pandas as pd
from Read_Data import read_file

maxlen = 500
padding = 'post'
truncating = 'post'

# model = keras.models.load_model('1')

# converter = tf.lite.TFLiteConverter.from_saved_model('1')
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False  # Disable experimental lowering
# tflite_model = converter.convert()

# converter = tf.lite.TFLiteConverter.from_saved_model('./1')
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
# ]
# converter.experimental_enable_resource_variables = True
# tflite_model = converter.convert()
#
#
# with open("model.tflite", "wb") as f:
#     f.write(tflite_model)

job_title = read_file('./Dataset/data_capstone.csv', 'job_title', True)
print(job_title)

with open("label.txt", "w") as f:
    f.write(str(job_title))

