import numpy as np
from random import shuffle


# Helper functions
def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sig_deriv(s):
    return s * (1 - s)


def feed_noround(weights, data, bias):
    data_new = np.append(data, bias)
    summe = np.dot(weights, data_new)
    sigm = sigmoid(summe)
    return sigm


def xor(data, weights_and1, weights_and2, weights_or, bias_and1, bias_and2, bias_or):
    summe_and1 = feed_noround(weights_and1, data, bias_and1)
    summe_and2 = feed_noround(weights_and2, data, bias_and2)
    summe_or = feed_noround(weights_or, [summe_and1, summe_and2], bias_or)
    return summe_and1, summe_and2, summe_or


### Back Propagation:

def back_prop(data, weights_and1, weights_and2, weights_or, learn_rate):
    # Claculate Network
    summe_and1, summe_and2, summe_or = xor(data[:2], weights_and1, weights_and2, weights_or, 1, 1, 1)
    layer1 = np.array([summe_and1, summe_and2, 1])
    losss = summe_or - data[2]

    # Update last layer weights
    weights_or -= sig_deriv(summe_or) * losss * layer1 * learn_rate

    # Calculate hidden loss
    hl = [sig_deriv(summe_or) * losss * weights_or[0],
            sig_deriv(summe_or) * losss * weights_or[1]]

    # Update first layer weights
    data_new = np.array(data[:2] + [1])
    weights_and1 -= sig_deriv(summe_and1) * hl[0] * data_new * learn_rate
    weights_and2 -= sig_deriv(summe_and2) * hl[1] * data_new * learn_rate

    return weights_and1, weights_and2, weights_or, losss


# Initialize random weights
weights_and1 = np.random.normal(size=3)
weights_and2 = np.random.normal(size=3)
weights_or = np.random.normal(size=3)


for _ in range(2000):
    data = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    shuffle(data)
    loss_sum = 0
    for i in data:
        weights_and1, weights_and2, weights_or, losss = back_prop(i, weights_and1, weights_and2, weights_or, 0.5)
        loss_sum += abs(losss)
    if _%200 == 0:
        print("Iteration: ", _, "Loss: ", loss_sum)

print("-------------------------------")
print("Weights: ")
print("And1: {}".format(weights_and1))
print("And2: {}".format(weights_and2))
print("Or: {}".format(weights_or))
print("-------------------------------")

outputs = []
datas = []
for i in data:
    summe_and1, summe_and2, summe_or = xor(i[:2], weights_and1, weights_and2, weights_or, 1, 1, 1)
    outputs.append(summe_or)
    datas.append(i[:2])

for i in zip(datas, outputs):
    print("Data: ", i[0], "Output: ", i[1])

print("-------------------------------")
