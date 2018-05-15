# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:14:24 2018

@author: Marcus
"""
import numpy as np

data = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
X = data[:,0:3]
y = data[:,-1]

##AND: -w_bias < sum w1 und w2 <= - 2 * w_bias
###geg: X_0, X_1, X_2, w_0, w_1, w_2, y
###ges: beta_0, beta_1, beta_2, y_hat, error
w = np.array([[-0.5, 0.25, 0.3]])
activation_value = np.sum(w * X, axis = 1)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

output = np.round(sigmoid(activation_value))

print(output)
print(X)

#####OR 
w_or = np.array([[-0.5, 0.6, 0.6]])
activation_value_or = np.sum(w_or * X, axis = 1)
output_or = np.round(sigmoid(activation_value_or))

print(output_or)
print(X)

####XOR sum w1 und w2 > - 2 * w_bias
import numpy as np
import pandas as pd

###Inputs
data = np.array([[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
X = data[:,0:3]
y = data[:,-1]
w0_1 = np.array([[-1.4, 1.2, -1]]) #bias, x1_node1, x2_node1 ##AND
w0_2 = np.array([[-1.4, -1, 1.2]])
w1_out = np.array([[-0.1, 0.2, 0.2]]) #bias, node1_out, node2_out ##OR

###FUNCTIONS
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def activation(w, X):
    return sigmoid(np.sum(w * X, axis = 1)) #same: np.dot(w, X.T)

def output(w0_1, w0_2, w1_out, X):
    activate_output = X[:,0], activation(w0_1, X), activation(w0_2, X)
    activate_output = pd.DataFrame.from_records(activate_output).T
    return sigmoid(np.sum(w1_out * activate_output, axis = 1))
    
def output_rounded(w0_1, w0_2, w1_out, X):
    return np.round(output(w0_1, w0_2, w1_out, X))

###OUTPUTS
output(w0_1, w0_2, w1_out, X)
output_rounded(w0_1, w0_2, w1_out, X)
