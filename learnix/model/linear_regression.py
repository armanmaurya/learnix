class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(epochs):
            y_predicted = [self._predict_single(x) for x in X]
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = y_predicted[i] - y[i]
                for j in range(n_features):
                    dw[j] += (1 / n_samples) * error * X[i][j]
                db += (1 / n_samples) * error

            self.weights = [w - learning_rate * d for w, d in zip(self.weights, dw)]
            self.bias -= learning_rate * db

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
