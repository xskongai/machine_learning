#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/6/25
Description: polynomial
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression_plot(x_values, y_values, degree=2):
        """
        用多项式回归拟合数据并绘图。

        参数：
        x_values: list 或 array，输入特征
        y_values: list 或 array，目标值
        degree: int，多项式阶数（默认2）
        """
        # 转换为 numpy 格式
        X = np.array(x_values).reshape(-1, 1)
        y = np.array(y_values)

        # 多项式特征转换
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        # 拟合模型
        model = LinearRegression()
        model.fit(X_poly, y)

        # 生成预测曲线
        X_plot = np.linspace(min(x_values), max(x_values), 100).reshape(-1, 1)
        X_plot_poly = poly.transform(X_plot)
        y_plot = model.predict(X_plot_poly)

        # 绘图
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='black', label='原始数据点')
        plt.plot(X_plot, y_plot, color='red', label=f'多项式回归（degree={degree}）')
        plt.title('Polynomial Regression 拟合效果')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    # 1. 手动输入数据点
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 输入特征
    y = np.array([1.2, 1.9, 3.2, 3.9, 5.1])  # 目标值（带点非线性）
    polynomial_regression_plot(X,y,degree=1)
    polynomial_regression_plot(X,y,degree=2)



if __name__ == "__main__":
    main()
