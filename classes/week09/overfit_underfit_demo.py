#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/6/25
Description: overfit_underfit_demo
"""

"""
这个结构清晰、易于扩展。你可以：

修改 poly_degree 观察不同阶数的拟合效果；

增加样本数量或噪声强度；

添加模型评估指标（如 R²、MSE）；
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def generate_data(n_samples=20, noise_std=1.0, seed=42):
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = 0.5 * X**2 + X + 2 + np.random.normal(0, noise_std, size=X.shape)
    return X, y.ravel()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def evaluate_models_cv(X, y, degrees=[1, 2, 5], cv=5):
    """
    使用交叉验证评估不同阶数的多项式回归模型
    """
    print("交叉验证结果（均方误差，越小越好）:")
    for d in degrees:
        model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mean_mse = -scores.mean()
        print(f"Degree {d}: MSE = {mean_mse:.4f}")



def plot_models(X, y, poly_degree=5):
    # 线性回归模型
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_linear = linear_model.predict(X)

    # 多项式回归模型
    poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
    poly_model.fit(X, y)
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_poly = poly_model.predict(X_plot)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='black', label='原始数据')
    plt.plot(X, y_linear, color='blue', label='线性回归（欠拟合）')
    plt.plot(X_plot, y_poly, color='red', label=f'{poly_degree}阶多项式回归（可能过拟合）')
    plt.title('Bias-Variance Tradeoff: 欠拟合 vs 过拟合')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 生成数据
    X, y = generate_data()

    # 交叉验证评估不同阶数的模型
    evaluate_models_cv(X, y, degrees=[1, 2, 5], cv=5)
    # 可视化模型拟合效果
    plot_models(X, y, poly_degree=10)


if __name__ == "__main__":
    main()
