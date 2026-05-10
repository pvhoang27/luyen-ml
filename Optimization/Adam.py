import numpy as np

x = np.array([1,2,3,4])
y = np.array([3,5,7,9])

w = 0.0
b = 0.0

mw, vw = 0, 0
mb, vb = 0, 0

lr = 0.01
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

epochs = 1000
n = len(x)

for t in range(1, epochs + 1):

    y_pred = w * x + b

    dw = (2/n) * np.sum((y_pred - y) * x)
    db = (2/n) * np.sum(y_pred - y)

    mw = beta1 * mw + (1 - beta1) * dw
    vw = beta2 * vw + (1 - beta2) * (dw ** 2)

    mb = beta1 * mb + (1 - beta1) * db
    vb = beta2 * vb + (1 - beta2) * (db ** 2)

    mw_hat = mw / (1 - beta1 ** t)
    vw_hat = vw / (1 - beta2 ** t)

    mb_hat = mb / (1 - beta1 ** t)
    vb_hat = vb / (1 - beta2 ** t)

    w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
    b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

print(w, b)