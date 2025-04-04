import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_train_data():
    return pd.read_csv('train.csv')

def get_test_data():
    return pd.read_csv('test.csv')

def plot_cost(cost_history):
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', linestyle='-')
    plt.xlabel("Epochs")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function Over Time")
    plt.grid(True)
    plt.show()

#region Linear Regression

def train_linear_regression(X: np.ndarray, y: np.ndarray, alpha: float, epochs: int):
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

#endregion

#region Softmax Regression

def train_softmax_regression(X, y, alpha, lambd, epochs, num_classes):
    m, n = X.shape
    W = np.zeros((n, num_classes))
    B = np.zeros(num_classes)
    cost_history = []
    for _ in range(epochs):
        Z = np.dot(X, W) + B # (m, num_classes) Logits

        # exp_Z could be large number so we are using keepdims for overflow exception
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # FT: Since Z_max is the largest value in each row, subtracting it shifts all values down, making the largest value in Z - Z_max 0
        H = exp_Z / np.sum(exp_Z, axis=1, keepdims=True) # (m, num_classes) matrix where each row has probabilities that sum to 1

        Y_one_hot = np.eye(num_classes)[y] # (m, num_classes)

        cost = -np.sum(Y_one_hot * np.log(H + 1e-8)) / m
        cost += (lambd / (2 * m)) * np.sum(W ** 2)
        cost_history.append(cost)

        grad_W = np.dot(X.T, (H - Y_one_hot)) / m + (lambd / m) * W
        grad_B = np.sum(H - Y_one_hot, axis=0) / m

        W -= alpha * grad_W
        B -= alpha * grad_B

    return W, B, cost_history

def predict(X, W, B):
    Z = np.dot(X, W) + B # (1, num_classes)

    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # (1, num_classes)
    H = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)  # (1, num_classes)
    
    predicted_class = np.argmax(H, axis=1) # (1, 1)
    
    return predicted_class

#endregion
