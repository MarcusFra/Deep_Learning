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
    return sigmoid(np.dot(w, X.T)) #same: #np.sum(w * X, axis = 1)


def hidden_output(inputweight_1, inputweight_2, X): #inputweight_1[0][0] * 
    activate_output = X[:, 0], activation(inputweight_1, X), activation(inputweight_2, X)
    return pd.DataFrame.from_records(activate_output).T.values


def output(inputweight_1, inputweight_2, outputweight, X):
    return sigmoid(np.dot(outputweight, hidden_output(inputweight_1, inputweight_2, X).T))  ##np.sum(outputweight * hidden_output(inputweight_1, inputweight_2, X), axis = 1)
    

def output_rounded(inputweight_1, inputweight_2, outputweight, X):
    return np.round(output(inputweight_1, inputweight_2, outputweight, X))


def loss(y, inputweight_1, inputweight_2, outputweight, X):
    return y - output(inputweight_1, inputweight_2, outputweight, X)


def w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
    loss_lr = loss(y, inputweight_1, inputweight_2, outputweight, X) * learnrate
    sig_loss = sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) * loss_lr
    w_change_per_epoch = - np.dot(sig_loss , hidden_output(inputweight_1, inputweight_2, X))  
    return outputweight - w_change_per_epoch

#def w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
#    w_change_per_epoch = []
#    for i, j in enumerate(outputweight):
#        w_change_per_epoch.append(np.sum(- loss(y, inputweight_1, inputweight_2, outputweight, X) * \
#                        sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) *  \
#                        hidden_output(inputweight_1, inputweight_2, X)[:,i] * learnrate)) 
#    return outputweight - w_change_per_epoch

#def w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
#    w_change_per_epoch = []
#    for i, j in enumerate(outputweight):
#        w_change_per_epoch.append(np.sum(- loss(y, inputweight_1, inputweight_2, outputweight, X) * \
#                        sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) *  \
#                        hidden_output(inputweight_1, inputweight_2, X)[i,:].reshape((3,1)) * learnrate)) 
#    return outputweight - w_change_per_epoch

#def w_change_out(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
#    w_change_per_epoch = np.sum(- loss(y, inputweight_1, inputweight_2, outputweight, X) * \
#                                sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) *  \
#                         hidden_output(inputweight_1, inputweight_2, X).reshape((3,4)) * learnrate, axis=1)
#    return outputweight - w_change_per_epoch


def hidden_loss(outputweight):
    return outputweight *  sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)).reshape((4,1)) #sigmoid_diff(output(inputweight_1, inputweight_2, outputweight, X)) * outputweight.reshape((3,1))


def w_change_in_1(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
    zusammen = - sigmoid_diff(hidden_output(inputweight_1, inputweight_2, X))[:,1] * hidden_loss(outputweight)[:,1]
    #change = np.sum((zusammen.reshape((4,1)) * X * learnrate), axis = 0)
    change = np.dot(zusammen , X)
    return inputweight_1 - change * learnrate


def w_change_in_2(y, inputweight_1, inputweight_2, outputweight, X, learnrate):
    zusammen = - sigmoid_diff(hidden_output(inputweight_1, inputweight_2, X))[:,2] * hidden_loss(outputweight)[:,2]
    #change = np.sum((zusammen.reshape((4,1)) * X * learnrate), axis = 0)
    change = np.dot(zusammen , X)
    return inputweight_2 - change * learnrate

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
inputweight_1 = np.random.uniform(-(np.sqrt(6/(m + n))), np.sqrt(6/(m + n)), 3)#np.array([[-1.4, 1.2, -1]])
inputweight_2 = np.random.uniform(-(np.sqrt(6/(m + n))), np.sqrt(6/(m + n)), 3)#np.array([[-1.4, -1, 1.2]])
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
    

