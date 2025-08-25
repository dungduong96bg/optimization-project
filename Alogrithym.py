import pandas as pd
import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def logistic_loss(y_true, y_pred):
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
def sigmoid_loss(y_true, y_pred):
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
def tanh_loss(y_true, y_pred):
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
def relu_loss(y_true, y_pred):
    return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
def gradient_descent(x, y, learning_rate, iterations):
    
    return np.linalg.inv(x.T @ x + learning_rate * y.T @ y)

def backtracking(x, y, learning_rate, iterations):
    return np.linalg.inv(x.T @ x + learning_rate * y.T @ y)

