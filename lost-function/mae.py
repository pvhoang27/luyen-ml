import numpy as np

y_true = np.array([3, 5, 7, 9])
y_pred = np.array([2.5, 5, 8, 10])

mae = np.mean(np.abs(y_true - y_pred))

print("MAE =", mae)