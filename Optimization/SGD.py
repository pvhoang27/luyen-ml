import numpy as np

x = np.array([1,2,3,4])
y = np.array([3,5,7,9])

w = 0.0
b = 0.0

lr = 0.01
epochs = 100

for epoch in range(epochs):

    for i in range(len(x)):

        y_pred = w * x[i] + b

        dw = 2 * (y_pred - y[i]) * x[i]
        db = 2 * (y_pred - y[i])

        w -= lr * dw
        b -= lr * db

print(w, b)