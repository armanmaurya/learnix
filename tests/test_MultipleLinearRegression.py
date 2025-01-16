import unittest
import numpy as np
from learnix.model.MultipleLinearRegression import MultipleLinearRegression

class TestMultipleLinearRegression(unittest.TestCase):
    def setUp(self):
        self.model = MultipleLinearRegression(learning_rate=0.01, iterations=2000)
        self.X = np.array([[1, 1, 1], [1, 2, 2], [2, 2, 3], [2, 3, 4]])
        self.y = np.array([6, 8, 9, 11])

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.coefficients)
        self.model.plot_errors()

        

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)
        accuracy = self.model.calculate_accuracy(self.y, predictions)
        print("arruracy", accuracy)

    def test_mean_squared_error(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        mse = self.model.mean_squared_error(self.y, predictions)
        self.assertGreaterEqual(mse, 0)

if __name__ == '__main__':
    unittest.main()
