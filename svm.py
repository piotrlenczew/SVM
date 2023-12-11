import numpy as np
import matplotlib.pyplot as plt


class SVMParams:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'linear',
        degree: int = 3,
        sigma: float = 0.5,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma


class SVM:
    def __init__(self, params: SVMParams):
        self.params = params
        self.X = None
        self.y = None
        self.alpha = None
        self.bias = None

    def _linear_kernel(self, u, v):
        return np.dot(u.T, v)

    def _polynomial_kernel(self, u, v):
        return (1 + np.dot(u.T, v)) ** self.params.degree

    def _rbf_kernel(self, u, v):
        return np.exp(-np.linalg.norm(u - v) ** 2 / (2 * self.params.sigma ** 2))

    def _kernel(self, u, v):
        if self.params.kernel == 'linear':
            return self._linear_kernel(u, v)
        elif self.params.kernel == 'polynomial':
            return self._polynomial_kernel(u, v)
        elif self.params.kernel == 'rbf':
            return self._rbf_kernel(u, v)
        else:
            raise ValueError("Unsupported kernel type")

    def _loss(self):
        return np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * np.outer(self.y, self.y) * self._kernel(self.X, self.X))

    def _calculate_bias(self):
        index = np.where((self.alpha) > 0 & (self.alpha < self.params.C))[0]
        b_i = self.y[index] - (self.alpha * self.y).dot(self._kernel(self.X, self.X[index]))
        return np.mean(b_i)

    def _learning_rate(self, xi):
        return 1/self._kernel(xi, xi)

    def _gradient(self, yi, xi):
        return 1 - yi * np.sum([self.alpha[j] * self.y[j] * self._kernel(self.X[j], xi) for j in range(self.X.shape[0])])

    def fit(self, X, y, epochs=500):
        self.X = X
        self.y = y

        self.alpha = np.random.random(X.shape[0])
        self.bias = 0

        losses = []
        for _ in range(epochs):
            for i in range(X.shape[0]):
                gradient = self._gradient(y[i], X[i])
                self.alpha[i] = self.alpha[i] + self._learning_rate(X[i]) * gradient

            self.alpha[self.alpha > self.params.C] = self.params.C
            self.alpha[self.alpha < 0] = 0

            losses.append(self._loss())

        self.bias = self._calculate_bias()

        plt.plot(losses)
        plt.title("loss per epochs")
        plt.show()

    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self._kernel(self.X, X)) + self.bias

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
