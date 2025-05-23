import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_train_data():
    return pd.read_csv('train.csv')

def get_test_data():
    return pd.read_csv('test.csv')

def get_test_labels():
    return pd.read_csv('test_labels.csv')

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
        H = get_H_for_softmax(X, W, B)

        Y_one_hot = np.eye(num_classes)[y] # (m, num_classes)

        cost = -np.sum(Y_one_hot * np.log(H + 1e-8)) / m
        cost += (lambd / (2 * m)) * np.sum(W ** 2)
        cost_history.append(cost)

        grad_W = np.dot(X.T, (H - Y_one_hot)) / m + (lambd / m) * W
        grad_B = np.sum(H - Y_one_hot, axis=0) / m

        W -= alpha * grad_W
        B -= alpha * grad_B

    return W, B, cost_history

def predict_for_softmax(X, W, B):
    H = get_H_for_softmax(X, W, B)

    predicted_class = np.argmax(H, axis=1) # (1, 1)
    
    return predicted_class

def percent_correct_for_softmax(X_test, y_test, W, B):
    H = get_H_for_softmax(X_test, W, B)

    y_pred = np.argmax(H, axis=1)
    correct = np.sum(y_pred == y_test)
    total = len(y_test)
    accuracy_percent = (correct / total) * 100

    print(f"Accuracy: {accuracy_percent:.2f}%")

    return accuracy_percent

def get_H_for_softmax(X, W, B):
    Z = np.dot(X, W) + B # (m, num_classes) Logits
    # exp_Z could be large number so we are using keepdims for overflow exception
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # FT: Since Z_max is the largest value in each row, subtracting it shifts all values down, making the largest value in Z - Z_max 0
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True) # (m, num_classes) matrix where each row has probabilities that sum to 1

#endregion

#region Neural Network

class NeuralNetwork:
    def __init__(self, layer_sizes: np.ndarray):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(1 / self.layer_sizes[i]))
            self.biases.append(np.random.randn(1, self.layer_sizes[i + 1]))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        
        self.activations = [X] # FT: At the end there will be num_layers number of activations
        self.z_values = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            a = sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)

        return self.activations[-1] # FT: Returning last, output, activation
    
    def SGD(self, training_data, epochs, mini_batch_size, alpha, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))
    
    def update_mini_batch(self, mini_batch, alpha):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``"""
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            # FT: Backprop is a fast way for calculating gradients of weights and biases
            # Calculates how much each weight should change for x, y pair
            delta_nabla_weights, delta_nabla_biases = self.backprop(x, y)
            nabla_weights = [nw + dnw 
                            for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
            nabla_biases = [nb + dnb 
                           for nb, dnb in zip(nabla_biases, delta_nabla_biases)]
            
        m = len(mini_batch)

        self.weights = [w - (alpha / m) * nw # nw is derivative of cost func with respect to w
                        for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b - (alpha / m) * nb # nb is derivative of cost func with respect to b
                       for b, nb in zip(self.biases, nabla_biases)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_weights, nabla_biases)`` representing the
        gradient for the cost function C_x."""
        # Ensure shapes: x -> (1, n_features), y -> (1, n_output) or scalar
        x = np.atleast_2d(x)

        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        activation = x
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activation, w) + b
            zs.append(z)
            # print(f'{activation.shape} dot {w.shape} => {z.shape}')
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        # dC/dw = dz/dw * da/dz * dC/da = a(l-1) * d'(z(l)) * 2(a(l) - y)
        # dC/db = dz/db * da/dz * dC/da =          d'(z(l)) * 2(a(l) - y)
        y_one_hot = np.eye(self.layer_sizes[-1])[y] # if the value of y is [2] make it [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        error = activations[-1] - y_one_hot
        # δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) * σ′(z⁽ᴸ⁾)
        delta = error * sigmoid_derivative(zs[-1]) # [1, 0, ..., -0.5] - [2, 1, ..., 4]
        # print(delta.shape) # (1, 10)
        # print(activations[-2].shape) # (1, 15)
        nabla_weights[-1] = np.dot(activations[-2].T, delta)
        nabla_biases[-1] = delta
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sd = sigmoid_derivative(z)
            # print(delta.shape) # (1, 10)
            # print(self.weights[-l+1].T.shape) # (10, 15)
            delta = np.dot(delta, self.weights[-l+1].T) * sd
            # print(activations[-l-1].T.shape) # (10, 1)
            # print(delta.shape) # (1, 10)
            nabla_weights[-l] = np.dot(activations[-l-1].T, delta)
            nabla_biases[-l] = delta

        # ∇w, ∇b - dC/dw, dC/db
        return nabla_weights, nabla_biases
    
    def evaluate(self, test_data):
        # FT: Unpack y as a scalar because it's (1,) shape
        test_results = [
            (np.argmax(self.forward(x)), y_scalar)
            for (x, (y_scalar,)) in test_data
        ]
        return sum(int(pred == y) for (pred, y) in test_results)

    def predict(self, X):
        outputs = self.forward(X)
        # If multiple samples
        if outputs.ndim == 2:
            return [int(np.argmax(o)) for o in outputs]
        # Single sample
        return int(np.argmax(outputs))

def sigmoid(z):
    z = np.clip(z, -500, +500) # Avoiding overflow
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1.0 - a)

def draw_neural_net(layer_sizes, max_neurons=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    h_spacing = 1.0 / float(len(layer_sizes) - 1)
    neuron_coords = []

    for i, layer_size in enumerate(layer_sizes):
        layer_coords = []
        if layer_size <= max_neurons:
            neurons_to_draw = list(range(layer_size))
        else:
            half = max_neurons // 2
            neurons_to_draw = list(range(half)) + ['...'] + list(range(layer_size - half, layer_size))

        total_neurons = len(neurons_to_draw)
        v_spacing = 0.05
        top = (1 - v_spacing * (total_neurons - 1)) / 2

        for j, idx in enumerate(neurons_to_draw):
            x = i * h_spacing
            y = top + j * v_spacing
            if idx == '...':
                ax.text(x, y, '...', ha='center', va='center')
                layer_coords.append(None)
            else:
                circle = plt.Circle((x, y), 0.02, color='black', fill=True)
                ax.add_patch(circle)
                layer_coords.append((x, y))

        neuron_coords.append(layer_coords)

    # Draw connections (skip connections to/from '...')
    for i in range(len(neuron_coords) - 1):
        for a in neuron_coords[i]:
            for b in neuron_coords[i + 1]:
                if a is not None and b is not None:
                    line = plt.Line2D((a[0], b[0]), (a[1], b[1]), c='gray', lw=0.5)
                    ax.add_line(line)

    plt.title('Neural Network Architecture (simplified)', fontsize=14)
    plt.show()

#endregion