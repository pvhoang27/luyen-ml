import numpy as np

x = np.array([1,2,3,4])
y = np.array([3,5,7,9])

w = 0.0
b = 0.0

lr = 0.01
epochs = 1000
n = len(x)

for epoch in range(epochs):

    y_pred = w * x + b

    dw = (2/n) * np.sum((y_pred - y) * x)
    db = (2/n) * np.sum(y_pred - y)

    w -= lr * dw
    b -= lr * db

print(w, b)