import numpy as np

from src.common.metrics import mean_squared_error

# Weight initialization
w = 0
# Bias initialization
b = 0


def fit(x_train, y_train, learning_rate=0.01, epochs=1000):
    global w, b
    n = len(x_train)
    for epoch in range(epochs):
        # Compute predictions
        y_pred = w * x_train + b

        # Compute loss (MSE)
        loss = mean_squared_error(y_train, y_pred)

        # Compute gradient
        dw = (2 / n) * np.sum((y_pred - y_train) * x_train)
        db = (2 / n) * np.sum((y_pred - y_train))

        # update weight and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")


def predict(x_test):
    return w * x_test + b
