#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: decision_tree_regression_house_price
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def dt_regression_demo():
    # 1.构造数据，房屋面积（平方米） 和房价（W 元）
    X = np.array([50, 60, 70, 80, 100, 120, 150]).reshape(-1, 1)
    y = np.array([150, 180, 210, 240, 300, 360, 450])

    # 2. 建立 决策树回归模型
    reg = DecisionTreeRegressor(max_depth=3, random_state=42, ccp_alpha=0.01)
    reg.fit(X, y)

    # 3. 预测新房价
    new_area = np.array([[90]])
    predicted_price = reg.predict(new_area)
    print("area:90, price:", predicted_price)

    # 4. 可视化

    X_test = np.linspace(40, 160, 300).reshape(-1, 1)
    y_pred = reg.predict(X_test)

    plt.scatter(X, y, color="blue", label="real data")
    plt.plot(X_test, y_pred, color="red", label="decision tree regression predict")
    plt.xlabel("house area")
    plt.ylabel("price 10k rmb")
    plt.title("decision tree regression example")
    plt.legend()
    plt.show()


def main():
    print("-------------------decision tree: regression ,house price --------------------------------")
    dt_regression_demo()


if __name__ == "__main__":
    main()
