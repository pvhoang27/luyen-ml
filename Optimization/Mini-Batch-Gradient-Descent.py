import numpy as np

x = np.array([1,2,3,4])
y = np.array([3,5,7,9])

w = 0.0
b = 0.0

lr = 0.01
epochs = 100
batch_size = 2

for epoch in range(epochs):

    for i in range(0, len(x), batch_size):

        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        y_pred = w * x_batch + b

        dw = (2/len(x_batch)) * np.sum((y_pred - y_batch) * x_batch)
        db = (2/len(x_batch)) * np.sum(y_pred - y_batch)

        w -= lr * dw
        b -= lr * db

print(w, b)