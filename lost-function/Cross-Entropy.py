import numpy as np

# Nhãn thật
y_true = np.array([1, 0, 1, 1])

# Xác suất model dự đoán
y_pred = np.array([0.9, 0.2, 0.8, 0.7])

# Tránh log(0)
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

# Binary Cross Entropy
loss = -np.mean(
    y_true * np.log(y_pred) +
    (1 - y_true) * np.log(1 - y_pred)
)

print("Cross Entropy Loss =", loss)