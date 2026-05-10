import numpy as np

# Dataset
x = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])

# Khởi tạo tham số
w = 0.0
b = 0.0

# Learning rate
lr = 0.01

# Số lần học
epochs = 1000

n = len(x)

# Gradient Descent
for epoch in range(epochs):

    # Prediction
    y_pred = w * x + b

    # Cost (MSE)
    cost = (1/n) * np.sum((y_pred - y) ** 2)

    # Gradient
    dw = (2/n) * np.sum((y_pred - y) * x)
    db = (2/n) * np.sum(y_pred - y)

    # Update parameters
    w = w - lr * dw
    b = b - lr * db

    # In kết quả
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost = {cost:.4f}")

print("\nKết quả:")
print("w =", w)
print("b =", b)