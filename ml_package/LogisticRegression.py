import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)
            class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
            return class_pred