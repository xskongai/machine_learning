import numpy as np
import matplotlib.pyplot as plt

# 示例数据加载函数
def load_data():
    x = np.array([50, 70, 90])  # 面积（㎡）
    y = np.array([100, 140, 180])  # 房价（万元）
    return x, y

# 损失函数（均方误差）
def compute_cost(x, y, theta_0, theta_1):
    m = len(x)
    predictions = theta_0 + theta_1 * x
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost

# 梯度计算
def compute_gradients(x, y, theta_0, theta_1):
    m = len(x)
    predictions = theta_0 + theta_1 * x
    error = predictions - y
    d_theta_0 = (1 / m) * np.sum(error)
    d_theta_1 = (1 / m) * np.sum(error * x)
    return d_theta_0, d_theta_1

# 梯度下降主函数
def gradient_descent(x, y, alpha=0.0001, iterations=1000):
    theta_0 = 0
    theta_1 = 0
    cost_history = []

    for i in range(iterations):
        d_theta_0, d_theta_1 = compute_gradients(x, y, theta_0, theta_1)
        theta_0 -= alpha * d_theta_0
        theta_1 -= alpha * d_theta_1
        cost = compute_cost(x, y, theta_0, theta_1)
        cost_history.append(cost)

        if i % 100 == 0 or i == iterations - 1:
            print(f"Iteration {i}: Cost = {cost:.4f}, θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}")

    return theta_0, theta_1, cost_history

# 拟合线可视化
def plot_regression_line(x, y, theta_0, theta_1):
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, theta_0 + theta_1 * x, color='red', label='Fitted Line')
    plt.xlabel('Area (㎡)')
    plt.ylabel('Price (万)')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

# 损失函数收敛图
def plot_cost_history(cost_history):
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.grid(True)
    plt.show()

# 主程序调用
x, y = load_data()
theta_0, theta_1, cost_history = gradient_descent(x, y, alpha=0.0001, iterations=1000)
print(f"\nFinal parameters: θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}")
plot_regression_line(x, y, theta_0, theta_1)
plot_cost_history(cost_history)
