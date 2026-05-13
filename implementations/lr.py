"""Logistic Regression"""

import math
import numpy as np
from typing import List, Optional


class LogisticRegression:

    # learning_rate: float
    # num_iterations: int

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.accuracy = math.inf
        self.accuracy_threshold = 0.01

    def sigmoid(self, z):
        x = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-x))


    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.bias = 0

        i = 0

        while i < self.num_iterations and self.accuracy > self.accuracy_threshold:
            z = np.dot(X, self.theta) + self.bias
            h = self.sigmoid(z)

            dw = np.dot(X.T, (h - y)) / m
            db = np.sum(h - y) / m

            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.theta) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """Returns 1 for true and 0 for false"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

