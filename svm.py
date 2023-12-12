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
        self._support_vectors = None
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.K = None

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

    # def _calculate_bias(self):
    #     index = np.where((self.alpha) > 0 & (self.alpha < self.params.C))[0]
    #     b_i = self.y[index] - (self.alpha * self.y).dot(self.kernel(self.X, self.X[index]))
    #     return np.mean(b_i)

    # def _decision_function(self, X):
    #     return self.weights.dot(self.kernel(self.X, X)) + self.bias
    #
    # def _margin(self, X, y):
    #     return y * self._decision_function(X)

    def __decision_function(self, X):
        return self.w.dot(self.kernel(self.X, X)) + self.b

    def __margin(self, X, y):
        return y * self.__decision_function(X)

    def fit(self, X, y, lr=1e-5, epochs=500):
        self.w = np.random.randn(X.shape[0])
        self.b = 0

        self.X = X
        self.y = y
        # Kernel Matrix
        self.K = self.kernel(X, X)

        loss_array = []
        for _ in range(epochs):
            margin = self.__margin(X, y)

            misclassified_pts_idx = np.where(margin < 1)[0]
            d_w = self.K.dot(self.w) - self.params.C * y[misclassified_pts_idx].dot(self.K[misclassified_pts_idx])
            self.w = self.w - lr * d_w

            d_b = - self.params.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b

            loss = (1 / 2) * self.w.dot(self.K.dot(self.w)) + self.params.C * np.sum(np.maximum(0, 1 - margin))
            loss_array.append(loss)

        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]

    # def _decision_function(self, X):
    #     return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.bias

    def predict(self, X):
        return np.sign(self.__decision_function(X))

    def score(self, X, y):
        prediction = self.predict(X)
        return np.mean(y == prediction)
