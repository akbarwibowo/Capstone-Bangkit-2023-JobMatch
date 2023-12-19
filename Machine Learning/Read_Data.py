import csv
import sklearn
from  sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

def read_file(path, column_name, convert_to_np):
    with open(path) as file:
        read = csv.DictReader(file, delimiter=";")
        data = []
        for row in read:
            if len(row[column_name]) > 0:
                data.append(row[column_name].strip())

        if convert_to_np is True:
            data = np.array(data)
            return data
        else:
            return data




