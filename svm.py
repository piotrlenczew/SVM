import numpy as np
from autograd import grad


class SVMParams:
    def __init__(
        self,
        lambda_param: float = 0.01,
        kernel: str = 'linear',
        degree: int = 3,
        sigma: float = 0.5,
    ):
        self.lambda_param = lambda_param
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
        return np.dot(u.T, v)

    def _polynomial_kernel(self, u, v):
        return (1 + np.dot(u.T, v)) ** self.params.degree

    def _rbf_kernel(self, u, v):
        return np.exp(-np.linalg.norm(u - v) ** 2 / (2 * self.params.sigma ** 2))

    def _hinge_loss(self, alpha, bias):
        return np.maximum(1 - np.dot(alpha * self.y, self.kernel(self.X, self.X)) - self.bias)

    def _regularization_term(self, alpha):
        return self.params.lambda_param * np.dot(alpha * alpha * self.y * self.y, self.kernel(self.X, self.X))

    def _objective_function(self, alpha, bias):
        return self._hinge_loss(alpha, bias) + self._regularization_term(alpha)

    def fit(self, X, y, lr=1e-3, epochs=500):
        self.X = np.array(X)
        self.y = np.array(y)
        self.alpha = np.random.random(X.shape)
        self.bias = 0

        objective_gradient = grad(self._objective_function)

        for _ in range(epochs):
            gradient_alphas, gradient_bias = objective_gradient(self.alpha, self.bias)

            self.alpha -= lr * gradient_alphas
            self.bias -= lr * gradient_bias

    def _decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.bias

    def predict(self, X):
        X = np.array(X)
        return np.sign(self._decision_function(X))

    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
