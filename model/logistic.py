import numpy as np


class LogisticRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.reg_lambda = 0.1  # L2 regularization parameter
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_cost(self, X, y):
        # Number of training examples
        m = X.shape[0]

        # Compute predictions
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)

        # Compute cost
        cost = (-1 / m) * np.sum(
            y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        )

        # Add regularization term
        reg_cost = (self.reg_lambda / (2 * m)) * np.sum(np.square(self.weights))

        return cost + reg_cost

    def fit(self, X, y):
        # Number of training examples and features
        m, n = X.shape

        # Initialize parameters
        self.initialize_parameters(n)

        # Gradient descent
        for i in range(self.num_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute gradients
            dw = (1 / m) * np.dot(X.T, (predictions - y)) + (
                self.reg_lambda / m
            ) * self.weights
            db = (1 / m) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Print cost every 100 iterations
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                print(f"Cost after iteration {i}: {cost}")

    def predict_proba(self, X):
        # Compute probability of class 1
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        # Predict class
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
