import unittest
from learnix.model import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def test_fit_and_predict(self):
        X = [[1, 1], [2, 2], [3, 3], [4, 4]]
        y = [2, 4, 6, 8]
        
        model = LinearRegression()
        model.fit(X, y, epochs=1000, learning_rate=0.01)
        
        predictions = model.predict([[5, 5], [6, 6]])
        self.assertAlmostEqual(predictions[0], 10, places=1)
        self.assertAlmostEqual(predictions[1], 12, places=1)

if __name__ == '__main__':
    unittest.main()
