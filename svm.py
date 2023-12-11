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
        if self.params.kernel == 'linear':
            self.kernel = self._linear_kernel
        elif self.params.kernel == 'polynomial':
            self.kernel = self._polynomial_kernel
        elif self.params.kernel == 'rbf':
            self.kernel = self._rbf_kernel
        else:
            raise ValueError("Unsupported kernel type")
        self.X = None
        self.y = None
        self.alpha = None
        self.bias = None

    def _linear_kernel(self, u, v):
        return u.dot(v.T)

    def _polynomial_kernel(self, u, v):
        return (1 + u.dot(v.T)) ** self.params.degree

    def _rbf_kernel(self, u, v):
        return np.exp(-(1 / self.params.sigma ** 2) * np.linalg.norm(u[:, np.newaxis] - v[np.newaxis, :], axis=2) ** 2)

    def _calculate_bias(self):
        index = np.where((self.alpha) > 0 & (self.alpha < self.params.C))[0]
        b_i = self.y[index] - (self.alpha * self.y).dot(self.kernel(self.X, self.X[index]))
        return np.mean(b_i)

    def _gradient(self, yi, xi):
        return 1 - yi * np.sum([self.alpha[j] * self.y[j] * self.kernel(self.X[j], xi) for j in range(self.X.shape[0])])

    def fit(self, X, y, lr=1e-3, epochs=500):
        self.X = X
        self.y = y

        self.alpha = np.random.random(X.shape[0])
        self.bias = 0
        ones = np.ones(X.shape[0])

        y_iy_jk_ij = np.outer(y, y) * self.kernel(X, X)

        losses = []
        for _ in range(epochs):
            gradient = ones - y_iy_jk_ij.dot(self.alpha)
            self.alpha = self.alpha + lr * gradient

            self.alpha[self.alpha > self.params.C] = self.params.C
            self.alpha[self.alpha < 0] = 0

            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_iy_jk_ij)
            losses.append(loss)

        index = np.where((self.alpha) > 0 & (self.alpha < self.params.C))[0]
        b_i = self.y[index] - np.sum(self.alpha * self.y * self.kernel(self.X, self.X[index]), axis=0)
        self.bias = np.mean(b_i)

        plt.plot(losses)
        plt.title("loss per epochs")
        plt.show()

    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.bias

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
