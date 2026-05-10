import numpy as np

x = np.array([1,2,3,4])
y = np.array([3,5,7,9])

w = 0.0
b = 0.0

sw = 0
sb = 0

lr = 0.01
beta = 0.9
eps = 1e-8
epochs = 1000
n = len(x)

for epoch in range(epochs):

    y_pred = w * x + b

    dw = (2/n) * np.sum((y_pred - y) * x)
    db = (2/n) * np.sum(y_pred - y)

    sw = beta * sw + (1 - beta) * (dw ** 2)
    sb = beta * sb + (1 - beta) * (db ** 2)

    w -= lr * dw / (np.sqrt(sw) + eps)
    b -= lr * db / (np.sqrt(sb) + eps)

print(w, b)