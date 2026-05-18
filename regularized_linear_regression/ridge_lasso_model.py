import numpy as np
class RidgeRegression:

    def __init__(self, learning_rate=0.001, epochs=1000, lambda_param=0.1):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (
                (2 / n_samples)
                * np.dot(X.T, (y_predicted - y))
                + (2 * self.lambda_param * self.weights)
            )

            db = (2 / n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias


class LassoRegression:

    def __init__(self, learning_rate=0.001, epochs=1000, lambda_param=0.1):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param

        self.weights = None
        self.bias = None
    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_predicted = np.dot(X, self.weights) + self.bias

            l1_gradient = np.sign(self.weights)

            dw = (
                (2 / n_samples)
                * np.dot(X.T, (y_predicted - y))
                + self.lambda_param * l1_gradient
            )

            db = (2 / n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
    def predict(self, X):

        return np.dot(X, self.weights) + self.bias