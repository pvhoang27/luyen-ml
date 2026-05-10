import numpy as np

# Nhãn thật (-1 hoặc 1)
y_true = np.array([1, -1, 1, -1])

# Dự đoán của model
y_pred = np.array([0.8, -0.7, 0.4, 0.2])

# Hinge Loss
loss = np.mean(np.maximum(0, 1 - y_true * y_pred))

print("Hinge Loss =", loss)