# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:52:21 2018

@author: Marcus
"""
from flask import Flask, request, render_template, jsonify
from parser_own import read
from keras import metrics, callbacks, backend
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from skimage import filters
from PIL import Image
import os
from image_proc import img_to_b64, score_word, b64_remove_header, b64_preprocess
# fix random seed for reproducibility
np.random.seed(7)
# load pima indians dataset


def model(k1, k2, k3, k_out, in_shape, hid_act_fct, out_act_fct):
    """model: neural network with 3 hidden layers"""
    m = Sequential([
        Dense(units=k1, input_shape=in_shape),
        Activation(hid_act_fct),
        Dense(units=k2, input_shape=in_shape),
        Activation(hid_act_fct),
        Dense(units=k3, input_shape=in_shape),
        Activation(hid_act_fct),
        Dense(units=k_out),
        Activation(out_act_fct),
        ])
    return m


def predict_loaded_model(Xtest):
    backend.clear_session()
    model = load_model('my_model.h5')
    model.compile(optimizer=opt, loss=loss, metrics=metric)
    ypred = model.predict_classes(Xtest)
    return ypred


"#DATA"
X, y = read(dataset='training')
N = 1000
X = X[:N, :, :]
ytrain = pd.get_dummies(y[:N])
Xtrain = X.reshape((N, 28*28))

"#MODEL PARAMETERS"
k1 = 64
k2 = 32
k3 = 16
k_out = 10
in_shape = (784, )
hid_act_fct = 'relu'
out_act_fct = 'softmax'
opt = 'rmsprop'
loss = 'categorical_crossentropy'
epochs = 1000
batch_size = 1000
verbose_fit = 2
metric = ["accuracy"]


"#CALL FUNCTIONS"
model = model(k1, k2, k3, k_out, in_shape, hid_act_fct, out_act_fct)
model.compile(optimizer=opt, loss=loss, metrics=metric)
model.fit(Xtrain, ytrain, epochs, batch_size, verbose=2)
model.evaluate(Xtrain, ytrain, verbose=0)
model.save('my_model.h5')
del model  


"#APP"
app = Flask(__name__)
name = "test"

@app.route('/submit_image', methods=['POST'])
def submit_image():
    print('IMAGE RECEIVED')
    error = None
    if request.method == 'POST':
        img_in = request.form['imgBase64']

        # Ensure image has no header
        img_in = b64_remove_header(img_in)

        # Convert image to process-able format
        img = b64_preprocess(img_in)
        img.save('image.png')
        small = img.resize((28, 28))
        pix = np.array(small).reshape((1, 784))
        pix = 255 - pix
        pix = filters.sobel(pix)
        solution = str(predict_loaded_model(pix))
        return jsonify({'response': solution})


@app.route("/")
def main():
    return render_template("index.html")
