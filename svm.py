import numpy as np


class SVMParams:
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
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

    def _polynomial_kernel(self, X1, X2):
        return (1 + X1.dot(X2.T)) ** self.params.degree

    def _rbf_kernel(self, X1, X2):
        return np.exp(-(1 / self.params.sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)

    # def _hinge_loss(self, alpha, bias):
    #     return np.maximum(1 - np.dot(alpha * self.y, self.kernel(self.X, self.X)) - self.bias)

    # def _regularization_term(self, alpha):
    #     return self.params.lambda_param * np.dot(alpha * alpha * self.y * self.y, self.kernel(self.X, self.X))
    #
    # def _objective_function(self, alpha, bias):
    #     return self._hinge_loss(alpha, bias) + self._regularization_term(alpha)

    def _calculate_bias(self):
        index = np.where((self.alpha) > 0 & (self.alpha < self.params.C))[0]
        b_i = self.y[index] - (self.alpha * self.y).dot(self.kernel(self.X, self.X[index]))
        return np.mean(b_i)

    # def _decision_function(self, X):
    #     return self.weights.dot(self.kernel(self.X, X)) + self.bias
    #
    # def _margin(self, X, y):
    #     return y * self._decision_function(X)

    def fit(self, X, y, lr=1e-5, epochs=500):
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

    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.bias

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(y == prediction)
