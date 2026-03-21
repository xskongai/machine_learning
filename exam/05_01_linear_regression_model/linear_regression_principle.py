#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: linear_regression_principle
"""

import numpy as np
import matplotlib.pyplot as plt


def linear_regression_principle():
    # 1. load data,housing price (x:area, y:price)
    x = np.array([50, 60, 70, 80, 90], dtype=float)
    y = np.array([150, 180, 210, 240, 270], dtype=float)

    # 2. initial parameter w and b
    w = 0.0
    b = 0.0
    # 3. define learning rate
    lr = 0.01
    # 4. difine iteration
    epochs = 1000
    # 5. training process
    loss_history = []
    w_history = []
    b_history = []
    for epoch in range(epochs):
        #  model predict
        y_pred = w * x + b
        #         compute mse
        loss = np.mean((y - y_pred) ** 2)
        loss_history.append(loss)
        w_history.append(w)
        b_history.append(b)

        # compute gradient
        dw = -2 * np.mean(x * (y - y_pred))
        db = -2 * np.mean(y - y_pred)

        #         update the parameter
        w -= lr * dw
        b -= lr * db

        if epoch % 200 == 0:
            print(f"{epoch},Loss = {loss:.4f},w = {w:.4f},b = {b:.4f}")

    # 6. training result
    print("it has trained")
    print(f"the final parameter w = {w:.4f},b = {b:.4f}")
    # 7. use the trained model to predict new data
    new_area = 120
    predicted_price = w * new_area + b
    print(f"new area :{new_area},the price is {predicted_price}")

    # 8. visulize the process of training
    plt.figure(figsize=(16, 12))

    # 图1 —— Loss 下降图
    plt.subplot(2, 2, 1)
    plt.plot(loss_history, color='blue')
    plt.title("Loss Curve (Training Process)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    # 图2 —— 参数 w、b 的变化
    plt.subplot(2, 2, 2)
    plt.plot(w_history, label="w")
    plt.plot(b_history, label="b")
    plt.title("Parameter Update Trajectory")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 图3 —— 拟合直线 vs 数据点
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, color='blue', label="Training Data")
    y_fit = w * x + b
    plt.plot(x, y_fit, color='red', label="Fitted Line")
    plt.title("Model Fit Result (Linear Regression)")
    plt.xlabel("Area (m²)")
    plt.ylabel("Price (万)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def main():
    linear_regression_principle()


if __name__ == "__main__":
    main()
