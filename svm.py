import numpy as np
from autograd import grad


class SVMParams:
    def __init__(
        self,
        lambda_param: float = 0.01,
        num_iter: int = 1000,
        learning_rate: float = 0.01,
        kernel: str = 'linear',
        degree: int = 3,
        sigma: float = 0.5,
    ):
        self.lambda_param = lambda_param
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma


class SVM:
    def __init__(self, params: SVMParams):
        self.params = params
        self.X_train = None
        self.Y_train = None
        self.alphas = None
        self.bias = None

    def _linear_kernel(self, u, v):
        return np.dot(u.T, v)

    def _polynomial_kernel(self, u, v):
        return (1 + np.dot(u.T, v)) ** self.params.degree

    def _rbf_kernel(self, u, v):
        return np.exp(-np.linalg.norm(u - v) ** 2 / (2 * self.params.sigma ** 2))

    def _apply_kernel(self, u, v):
        if self.params.kernel == 'linear':
            return self._linear_kernel(u, v)
        elif self.params.kernel == 'polynomial':
            return self._polynomial_kernel(u, v)
        elif self.params.kernel == 'rbf':
            return self._rbf_kernel(u, v)
        else:
            raise ValueError("Unsupported kernel type")

    def _hinge_loss(self, alphas, bias):
        loss = []
        for i in range(self.X_train.shape[0]):
            f = np.sum(alphas[i] * self.Y_train[i] * self._apply_kernel(self.X_train[i], self.X_train) - bias)
            loss.append(max(1-np.dot(f, self.Y_train[i]), 0))
        return max(loss)

    def _regularization_term(self, alphas):
        sum = 0.0
        for i in range(self.X_train.shape[0]):
            for j in range(self.X_train.shape[0]):
                sum += np.dot(alphas[i]*alphas[j]*self.Y_train[i]*self.Y_train[j], self._apply_kernel(self.X_train[i], self.X_train[j]))
        return self.params.lambda_param * sum

    def _objective_function(self, alphas, bias):
        return self._hinge_loss(alphas, bias) + self._regularization_term(alphas)

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        num_samples, num_features = X.shape
        self.alphas = np.zeros(num_samples)
        self.bias = 0

        objective_gradient = grad(self._objective_function, argnums=(0, 1))

        for _ in range(self.params.num_iter):
            gradient_alphas, gradient_bias = objective_gradient(self.alphas, self.bias)

            self.alphas -= self.params.learning_rate * gradient_alphas
            self.bias -= self.params.learning_rate * gradient_bias

    def _decision_function(self):
        return

    def predict(self, X):
        for i in range()
