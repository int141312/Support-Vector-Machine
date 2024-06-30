### SVM_kernel

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.model_selection
from sklearn import datasets
from numpy import random


class Kernelmodel:
    def __init__(self, kernel='rbf', sigma=0.2, learning_rate=0.0001, lambda_param=0.01, epochs=1000):
        self.kernel = kernel
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.alpha = None

    def gaussian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * (self.sigma ** 2)))

    def compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.gaussian_kernel(X[i], X[j])
        return K

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        n_samples, n_features = X.shape
        K = self.compute_kernel_matrix(X)
        self.alpha = np.zeros(n_samples)

        for epoch in range(self.epochs):
            for i in range(n_samples):
                gradient = 1 - Y[i] * np.sum(self.alpha * Y * K[:, i])
                self.alpha[i] += self.learning_rate * (gradient - self.lambda_param * self.alpha[i])
                self.alpha[i] = max(0, self.alpha[i])  # alpha 값은 0 이상이어야 함

    def predict(self, X_test):
        K = np.array([self.gaussian_kernel(x_train, X_test) for x_train in self.X])
        return np.sign(np.sum(self.alpha * self.Y * K))

    def plot_decision_boundary(self):
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = np.array([self.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bwr)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, s=100, edgecolors='k', marker='o', cmap=plt.cm.bwr)
        plt.show()

    def score(self, X_test, Y_test):
        predictions = np.array([self.predict(x) for x in X_test])
        return np.mean(predictions == Y_test)


# Example usage with dummy data
dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset[0], dataset[1], test_size = 0.3, random_state = 100)

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# 모델 학습 및 결과 시각화
model = Kernelmodel()
model.fit(X_train, y_train)

print('Model accuracy:', model.score(X_test, y_test))
model.plot_decision_boundary()


