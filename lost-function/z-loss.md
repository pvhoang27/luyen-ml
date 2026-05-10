# Loss Function trong Machine Learning

## 1. Loss Function là gì?

Loss Function (Hàm mất mát) là hàm dùng để đo mức độ sai lệch giữa:
- giá trị dự đoán của model
- giá trị thực tế

Mục tiêu của Machine Learning là:
> giảm Loss xuống nhỏ nhất có thể.

Nếu:
- Loss nhỏ → model dự đoán tốt
- Loss lớn → model dự đoán sai

---

# 2. Vai trò của Loss Function

Loss Function giúp:
- đánh giá model
- tối ưu model
- cập nhật tham số bằng Gradient Descent

Quy trình:

Data
→ Prediction
→ Loss Function
→ Gradient Descent
→ Update Parameters

---

# 3. Các Loss Function phổ biến

---

## 3.1 Mean Squared Error (MSE)

Dùng cho:
- Regression
- Linear Regression

Công thức:

MSE = (1/n) * Σ(y - y_pred)^2

Đặc điểm:
- bình phương sai số
- phạt lỗi lớn rất mạnh
- nhạy với outlier

Code:

```python
import numpy as np

y_true = np.array([3,5,7,9])
y_pred = np.array([2.5,5,8,10])

mse = np.mean((y_true - y_pred) ** 2)

print(mse)