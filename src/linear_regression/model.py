import numpy as np

from src.common.metrics import mean_squared_error

epsilon = 1e-8
max_epoch = 1000000

# Weight initialization
w = 0
# Bias initialization
b = 0


def fit(x_train, y_train, learning_rate=0.01):
    epoch = 0
    x_train = np.array(x_train).ravel()
    y_train = np.array(y_train).ravel()
    global w, b
    n = len(x_train)
    curr_loss = float('inf')
    while True:
        if epoch >= max_epoch:
            break

        # Compute predictions
        y_pred = w * x_train + b

        # Compute loss (MSE)
        prev_loss = curr_loss
        curr_loss = mean_squared_error(y_train, y_pred)

        if (abs(prev_loss - curr_loss))/prev_loss < epsilon:
            break

        # Compute gradient
        dw = (2 / n) * np.sum((y_pred - y_train) * x_train)
        db = (2 / n) * np.sum((y_pred - y_train))

        # update weight and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db

        epoch += 1
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")


def predict(x_test):
    print(f"Using w = {w:.4f} and b = {b:.4f} for prediction.")
    x_test = np.array(x_test).ravel()
    return w * x_test + b
