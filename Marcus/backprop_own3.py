# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:08:33 2018

@author: Marcus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:00:30 2018

@author: Marcus
"""

####XOR sum w1 und w2 > - 2 * w_bias
import numpy as np
import pandas as pd

np.random.seed(42)

###FUNCTIONS
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_diff(x):
    return x * (1 - x)

def activation(w, X):
    return sigmoid(np.sum(w * X, axis = 1)) #same: np.dot(w, X.T)

def hidden_output(inputweight_1, inputweight_2, X):
    activate_output = X[:,0], activation(inputweight_1, X), activation(inputweight_2, X)
    return pd.DataFrame.from_records(activate_output).T.values
    
def output(inputweight_1, inputweight_2, outputweight, X):
    return sigmoid(np.sum(outputweight * hidden_output(inputweight_1, inputweight_2, X), axis = 1))
    
def output_rounded(inputweight_1, inputweight_2, outputweight, X):
    return np.round(output(inputweight_1, inputweight_2, outputweight, X))

def loss(y, inputweight_1, inputweight_2, outputweight, X):
    return y - output(inputweight_1, inputweight_2, outputweight, X)

def w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
    w_change_per_epoch = []
    for i, j in enumerate(outputweight):
        w_change_per_epoch.append(np.sum(- loss(y, inputweight_1, inputweight_2, outputweight, X) * \
                        sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) *  \
                        hidden_output(inputweight_1, inputweight_2, X)[:,i] * learnrate)) 
    return outputweight - w_change_per_epoch
    
def hidden_loss(outputweight):
    return sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) * outputweight.reshape((3,1))

def w_change_in_1(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
    zusammen = - sigmoid_diff(hidden_output(inputweight_1, inputweight_2, X))[:,1] * hidden_loss(outputweight)[1]
    change = np.sum((zusammen.reshape((4,1)) * X * learnrate), axis = 0)
    return inputweight_1 - change

def w_change_in_2(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
    zusammen = - sigmoid_diff(hidden_output(inputweight_1, inputweight_2, X))[:,2] * hidden_loss(outputweight)[2]
    change = np.sum((zusammen.reshape((4,1)) * X * learnrate), axis = 0)
    return inputweight_2 - change

###OUTPUTS
#output(inputweight_1, inputweight_2, outputweight, X)
#output_rounded(inputweight_1, inputweight_2, outputweight, X)
#
#w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate)

###Inputs/ Initial Parameters
data = np.array([[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
X = data[:,0:3]
y = data[:,-1]
learnrate = 1
m = 6
n = 3
inputweight_1 = np.array([[-1.4, 1.2, -1]])
inputweight_2 = np.array([[-1.4, -1, 1.2]])
outputweight = np.random.uniform(-(np.sqrt(6/(m + n))), np.sqrt(6/(m + n)), 3)
z = 0.05

iter = 0
while (abs(loss(y, inputweight_1, inputweight_2, outputweight, X)) > z).any(): # and iter < 5
    out_new = w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate)
    innew_1 = w_change_in_1(y, inputweight_1, inputweight_2, outputweight, X, learnrate)
    innew_2 = w_change_in_2(y, inputweight_1, inputweight_2, outputweight, X, learnrate)
    outputweight = out_new
#    inputweight_1 = innew_1
#    inputweight_2 = innew_2
    if iter % 100 == 0:
        print(outputweight)
        print(sum(abs(loss(y, inputweight_1, inputweight_2, outputweight, X))))
    iter += 1