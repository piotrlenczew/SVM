import numpy as np


class SVMParams:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        degree: int = 3,
        sigma: float = 1.0,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma


class SVM:
    def __init__(self, params: SVMParams):
        self.params = params
        if self.params.kernel == 'polynomial':
            self.kernel = self._polynomial_kernel
        elif self.params.kernel == 'rbf':
            self.kernel = self._rbf_kernel
        else:
            raise ValueError("Unsupported kernel type")
        self.X = None
        self.y = None
        self.alpha = None
        self.bias = None

    def _polynomial_kernel(self, u, v):
        return (1 + u.dot(v.T)) ** self.params.degree

    def _rbf_kernel(self, u, v):
        return np.exp(-(1 / 2 * self.params.sigma ** 2) * np.linalg.norm(u[:, np.newaxis] - v[np.newaxis, :], axis=-1) ** 2)

    def _calculate_bias(self):
        index = np.where((self.alpha > 0) & (self.alpha < self.params.C))[0]
        if len(index) == 0:
            return 0
        b = self.y[index] - (self.alpha * self.y).dot(self.kernel(self.X, self.X[index]))
        return np.mean(b)

    def fit(self, X, y, lr=1e-5,  epochs=500):
        self.X = X
        self.y = y

        self.alpha = np.random.random(X.shape[0])
        self.bias = 0
        ones = np.ones(X.shape[0])

        yi_yj_k = np.outer(y, y) * self.kernel(X, X)

        losses = []
        for _ in range(epochs):
            gradient = ones - yi_yj_k.dot(self.alpha)
            self.alpha = self.alpha + lr * gradient

            self.alpha[self.alpha > self.params.C] = self.params.C
            self.alpha[self.alpha < 0] = 0

            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * yi_yj_k)
            losses.append(loss)

        self.bias = self._calculate_bias()
        return losses


    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.bias

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(y == prediction)
