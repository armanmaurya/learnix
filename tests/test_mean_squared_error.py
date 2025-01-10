import unittest
import numpy as np
from learnix.loss import mean_squared_error

class TestMeanSquaredError(unittest.TestCase):
    def test_mean_squared_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        self.assertEqual(mean_squared_error(y_true, y_pred), 0.0)

    def test_mean_squared_error_non_zero(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([4, 5, 6])
        self.assertEqual(mean_squared_error(y_true, y_pred), 9.0)

    def test_mean_squared_error_shape_mismatch(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        with self.assertRaises(ValueError):
            mean_squared_error(y_true, y_pred)

if __name__ == '__main__':
    unittest.main()
