import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class SoftMarginSVM:
    def __init__(self, C=1.0, learning_rate=0.001, epochs=1000):
        self.C = C  # Regularization parameter
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.epochs = epochs  # Number of iterations
        self.w = None  # Weights
        self.b = 0  # Bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Training process
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.w / n_samples)
                else:
                    self.w -= self.learning_rate * (2 * self.w / n_samples - np.dot(x_i, y[idx]) * self.C)
                    self.b -= self.learning_rate * (-y[idx] * self.C)

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def plot_decision_boundary(self, X, y):
        
        # 데이터 시각화
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)

        # 그래프의 한계 설정
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = (np.dot(xy, self.w) + self.b).reshape(XX.shape)

        # Decision boundary 시각화
        ax.contour(XX, YY, Z, levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], colors='k')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Soft SVM Decision Boundary')
        plt.show()


# iris 데이터 준비
iris = datasets.load_iris()
X = iris.data[50:, 2:]
y = iris.target[50:] 

# 데이터에서 y의 값을 1 or -1로 변환
y = np.where(y == 1, 1, -1)

# SVM 모델 생성 및 훈련
model = SoftMarginSVM()
model.fit(X, y)

# 예측 및 정확도 평가
print(y, "\n" ,model.predict(X)) 
print("Model accuracy:", model.score(X, y))

# 시각화
model.plot_decision_boundary(X, y)

