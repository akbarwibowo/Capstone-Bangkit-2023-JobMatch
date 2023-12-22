# JOB MATCH - MACHINE LEARNING

## Description
this is a machine learning project that uses a neural network to predict the probability of a candidate to be hired for a job.
The architecture of the neural network can be found in the file `UseFunctional.py`. Data preprocessing, training, and 
saving the model is done in the same file.
The model is saved in three options: `model.h5`, `SavedModel` with folder name of `1`, and `model.tflite`.
Tokenizers that used for preprocessing are saved in `degree_token.json`, `job_token.json`, and `key_token.json` files.
Loading model and predicting done in the file `main.py`. For preprocessing data ro predict, needs to load three tokenizers
to preprocess data.
`Convert.py` is used to converts SavedModel to `.tflite` format. 

## Dependencies

following deppendencies are needed:
* Tensorflow
* Pandas
* Numpy
* JSON
* OS

## How to clone
The model is already saved in `model.h5` format or others format available. All needs to do is
clone the repository and run `main.py` file. Inside `main.py` has already available helper functions
to preprocess input data and for predicting. If you want to train the model again, you can run 
`UseFunctional.py` file. The dataset is available in Dataset directory with file name
`capstone_dataset.csv`.
