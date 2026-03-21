#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/6/25
Description: linear_polynomial
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def main():
    # 1. 生成非线性数据
    np.random.seed(0)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(100, 1)  # 二次函数加噪声

    # 2. 线性回归模型
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_linear_pred = linear_model.predict(X)

    # 3. 多项式回归模型（2阶）
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    y_poly_pred = poly_model.predict(X)

    # 4. 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='gray', label='Data')
    plt.plot(X, y_linear_pred, color='blue', label='Linear Regression')
    plt.plot(X, y_poly_pred, color='red', label='Polynomial Regression (degree=2)')
    plt.legend()
    plt.title('Linear vs Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
