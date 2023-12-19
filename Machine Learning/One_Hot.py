import tensorflow as tf
import numpy as np


def all_one_hot(data):
    num_data = len(data)

    data_mapping = {label: i for i, label in enumerate(data)}

    data_encoded = np.array([data_mapping[label] for label in data], dtype=np.int32)

    data_one_hot = tf.one_hot(data_encoded, num_data)

    return data_one_hot


def unique_one_hot(data):
    unique = []
    for i in data:
        if i not in unique:
            unique.append(i)

    unique_len = len(unique)
    mapping = {label: i for i, label in enumerate(unique)}

    encoded = np.array([mapping[label] for label in unique], dtype=np.int32)

    one_hot = tf.one_hot(encoded, unique_len)

    return one_hot
