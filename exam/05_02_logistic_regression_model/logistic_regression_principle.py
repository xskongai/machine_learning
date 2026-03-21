#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: logistic_regression_principle
"""
import numpy as np

import matplotlib.pyplot as plt


def logistic_regression_principle():
    #  1. prepare train data
    X = np.array([[1], [2], [3], [7], [8], [9]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    n = len(X)

    # 2. sigmoid 函数
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # 3.init the parameter
    w = 0
    b = 0
    lr = 0.1
    epochs = 2000

    losses = []
    ws = []
    bs = []
    # 4. train using gradient descend
    for epoch in range(epochs):
        z = w * X + b
        y_pred = sigmoid(z)

        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        losses.append(loss)

        dw = np.mean((y_pred - y) * X.reshape(-1))
        db = np.mean(y_pred - y)

        w -= lr * dw
        b -= lr * db

        ws.append(w)
        bs.append(b)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f},w= {w:.4f},b = {b:.4f}")

    print("training is over")
    print(f"w= {w:.4f},b = {b:.4f}")

    def predict(x):
        prob = sigmoid(w * x + b)
        return 1 if prob >= 0.5 else 0

    text_x = 5.5
    pred_label = predict(text_x)
    print(f"predict a new value text_x:{text_x}，the label is {pred_label}")

    # 图 1：原始数据散点图
    plt.figure(figsize=(6, 4))
    plt.scatter(X[y == 0], y[y == 0], label="Class 0", marker="o")
    plt.scatter(X[y == 1], y[y == 1], label="Class 1", marker="x")
    plt.title("Training Data")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 图 2：Sigmoid 决策边界
    plt.figure(figsize=(6, 4))
    X_line = np.linspace(0, 10, 200)
    y_line = sigmoid(w * X_line + b)
    plt.plot(X_line, y_line)
    plt.scatter(X, y, c=y, cmap="bwr")
    plt.title("Logistic Regression Sigmoid Curve")
    plt.xlabel("X")
    plt.ylabel("Predicted Probability")
    plt.grid(True)
    plt.show()

    # 图 3：Loss 曲线（最经典）
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 图 4：z = wx + b 的线性变化（理解模型线性部分）
    plt.figure(figsize=(6, 4))
    plt.plot(ws, label="w")
    plt.plot(bs, label="b")
    plt.title("Parameter updates (w & b)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------
    # 6. Test a sample
    # -------------------------
    def predict(x):
        prob = sigmoid(w * x + b)
        return 1 if prob >= 0.5 else 0

    test_x = 5.5
    pred_label = predict(test_x)
    print(f"Prediction: X={test_x}, Predicted Label = {pred_label}, Prob={sigmoid(w * test_x + b):.4f}")


def main():
    logistic_regression_principle()


if __name__ == "__main__":
    main()
