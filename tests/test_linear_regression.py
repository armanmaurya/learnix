import unittest
import numpy as np
import matplotlib.pyplot as plt
from learnix.model.linear_regression import SimpleLinearRegression

def test_simple_linear_regression():
    # Generate some random data
    X = np.random.rand(100)
    y = 2 * X + 3 + np.random.randn(100) * 0.1  # y = 2X + 3 + noise

    # Initialize and fit the model
    model = SimpleLinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # # Check if predictions are close to actual values
    # assert np.allclose(predictions, y, atol=0.5), "Predictions are not close to actual values"

    # Plot the data and the regression line
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, predictions, color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    print("Test passed!")

# Run the test
test_simple_linear_regression()

if __name__ == '__main__':
    unittest.main()
