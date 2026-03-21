#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 10/30/25
Description: ShowGradientDescentCal
"""


import numpy as np

# 数据：面积（x）和房价（y）
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
m = len(x)  # 样本数量

# 初始化参数
theta_0 = 0
theta_1 = 1
alpha = 0.1  # 学习率
iterations = 5  # 迭代次数

print("初始参数: θ0 = {:.4f}, θ1 = {:.4f}".format(theta_0, theta_1))
print("-" * 40)

# 梯度下降迭代
for i in range(iterations):
    # 预测值
    h = theta_0 + theta_1 * x
    error = h - y

    # 计算导数
    d_theta_0 = (1/m) * np.sum(error)
    d_theta_1 = (1/m) * np.sum(error * x)

    # 更新参数
    theta_0 -= alpha * d_theta_0
    theta_1 -= alpha * d_theta_1

    # 打印每一步
    print(f"第 {i+1} 次迭代:")
    print("  预测值 h(x):", np.round(h, 4))
    print("  误差 h(x)-y:", np.round(error, 4))
    print("  ∂J/∂θ0 = {:.4f}, ∂J/∂θ1 = {:.4f}".format(d_theta_0, d_theta_1))
    print("  更新后: θ0 = {:.4f}, θ1 = {:.4f}".format(theta_0, theta_1))
    print("-" * 40)

def main():
    pass


if __name__ == "__main__":
    main()
