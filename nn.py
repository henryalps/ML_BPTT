import numpy as np
import scipy
import math

# a 3*4*3 neural network, 
# applying sigmoid function as primitive function,
# for understanding backpropagation(BPTT)

learning_rate = 0.01
input_layer_size = 3
hidden_layer_size = 4
output_layer_size = 3

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def diff_sigmoid(x):
    return x*(1 - x)

def get_layer_vector(layer):
    res = []
    for node in layer:
        res.append(node.cur_output)
    return(res)

def get_layer_error(layer):
    res = []
    for node in layer:
        res.append(node.cur_error)
    return(res)

class Node:
    cur_output = 0
    cur_error = 0

hidden_layer = np.empty(4, dtype = Node)
output_layer = np.empty(3, dtype = Node)
error_layer = np.empty(3, dtype = Node)
weights1 = np.random.randint(0, high=100, size=[4, 3])
weights2 = np.random.randint(0, high=100, size=[3, 4])

# propagation
for x_i in x:
    for node_index, node in enumerate(hidden_layer):
        node.cur_output = sigmoid(np.multiply(weights1[node_index]), x_i))
        node.cur_error = diff_sigmoid(node.cur_output)

    hidden_layer_output = get_layer_vector(hidden_layer)
    for node_index, node in enumerate(output_layer):
        node.cur_output = sigmoid(np.multiply(hidden_layer_output, weights2[node_index])))
        node.cur_error = diff_sigmoid(node.cur_output)
    