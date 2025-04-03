import pandas as pd
import numpy as np

def get_train_data():
    return pd.read_csv('train.csv')

def get_test_data():
    return pd.read_csv('test.csv')

def train(X: np.ndarray, y: np.ndarray, alpha: float, epochs: int):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = []
    for _ in range(epochs):
        predictions = np.dot(X, w) + b  
        error = predictions - y
        
        cost = np.mean(error ** 2)
        cost_history.append(cost)

        grad_w = np.dot(X.T, error) / m
        grad_b = np.sum(error) / m

        w -= alpha * grad_w
        b -= alpha * grad_b
    return w, b, cost_history