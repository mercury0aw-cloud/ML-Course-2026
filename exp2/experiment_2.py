import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.neighbors import KNeighborsClassifier
import torch

print("=== 1. 参数估计 (MLE 与 MAP) ===")
data_tensor = torch.tensor([1., 1., 0., 1., 0.])
p_mle_torch = torch.mean(data_tensor)
alpha, beta_param = 2, 2
p_map_torch = (torch.sum(data_tensor) + alpha - 1) / (len(data_tensor) + alpha + beta_param - 2)
print(f"PyTorch 计算得: MLE = {p_mle_torch.item():.4f}, MAP = {p_map_torch.item():.4f}")

data = np.array([1, 1, 0, 1, 0])
N = len(data)
sum_x = np.sum(data)
p = np.linspace(0, 1, 100)
likelihood = p**sum_x * (1 - p)**(N - sum_x)
prior = beta.pdf(p, alpha, beta_param)
posterior = p**(sum_x + 1) * (1 - p)**(N - sum_x + 1)
p_mle = sum_x / N
p_map = (sum_x + 1) / (N + 2)

plt.figure(figsize=(8, 5))
plt.plot(p, likelihood, label="Likelihood")
plt.plot(p, prior, label="Prior")
plt.plot(p, posterior, label="Posterior")
plt.axvline(p_mle, color='r', linestyle='--', label=f"MLE ({p_mle:.2f})")
plt.axvline(p_map, color='g', linestyle='--', label=f"MAP ({p_map:.2f})")
plt.legend()
plt.title("MLE vs MAP")
plt.grid()
plt.show()

print("\n=== 2. KNN 分类 ===")
X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])
xx, yy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))

for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f"KNN Boundary (K = {k})")
    plt.show()

print("\n=== 3. 梯度下降优化 ===")
loss = []
x = torch.tensor([5.0], requires_grad=True)
for i in range(20):
    y_val = x**2 + 2*x + 1
    loss.append(y_val.item())
    y_val.backward()
    with torch.no_grad():
        x -= 0.1 * x.grad
        x.grad.zero_()
        
plt.figure(figsize=(6, 5))
plt.plot(loss, marker='o')
plt.title("Gradient Descent - Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()

x_vals = np.linspace(-5, 5, 100)
y_vals = x_vals**2 + 2*x_vals + 1
x_path = []
x = torch.tensor([5.0], requires_grad=True)
for i in range(10):
    y_val = x**2 + 2*x + 1
    x_path.append(x.item())
    y_val.backward()
    with torch.no_grad():
        x -= 0.3 * x.grad
        x.grad.zero_()
        
plt.figure(figsize=(6, 5))
plt.plot(x_vals, y_vals, label='f(x) = x^2 + 2x + 1')
plt.scatter(x_path, [xx**2 + 2*xx + 1 for xx in x_path], color='r', zorder=5, label='Optimization Path')
plt.title("Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()

print("\n最终优化参数 x 接近: ", x.item())