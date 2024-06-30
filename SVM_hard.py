import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


class HardMaringSVM:

    # 하이퍼파라미터의 경우 10^-4 의 경우가 가장 효과적이었다. 
    def __init__(self, learning_rate=0.0001, lambda_param=0.02, iteration=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.iteration = iteration
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1) # y의 값을 1 or -1로 변환

        # 가중치와 바이어스 초기화
        # local minimum을 피하기 위해 초기값을 다르게 설정
        self.w = np.array([float(25.3), float(-5.2)]) 
        self.b = 20

        #self.w = np.zeros(n_features)
        #self.b = 0
        # 보통은 w=0, b=0으로 하는데 이 경우는 0.99 정확도가 나온다. 

        while self.iteration > 0:
            for idx in range(len(X)):
                x_i = X[idx]
                
                if y[idx] * (np.dot(x_i, self.w) - self.b) > 1:
                    self.w -= self.lr * (self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
            
            self.iteration -= 1


    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b) 
    
    def weight_result(self, X):
        print("w : ", self.w, "b : ", self.b) # 학습된 w, b, 예측값을 출력
    


def visualization():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", color=np.where(y == -1, 'red', 'blue'))

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
    x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


# 아이리스 데이터 준비
iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]  

# 데이터에서 y의 값을 1 or -1로 변환
y = np.where(y == 1, 1, -1)

# 모델 학습
model = HardMaringSVM()
model.fit(X, y)
y_pred = model.predict(X)

# 예측 및 겱과 평가
model.weight_result(X)
print(y, "\n" ,y_pred)
print("HardMaringSVM classification accuracy", accuracy(y, y_pred))

# 시각화
visualization()