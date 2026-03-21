#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: logistic_regression_classifier_example
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt


def sklearn_lr_example():
    # 1. prepare data
    np.random.seed(42)

    # class 1
    X0 = np.random.randn(50, 2) + np.array([2, 2])
    y0 = np.zeros(50)

    # class 2
    X1 = np.random.randn(50, 2) + np.array([6, 6])
    y1 = np.ones(50)

    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    # 2. split train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. create model and train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print(f"w (coefficient):{model.coef_}")
    print(f"b (intercept):{model.intercept_}")

    # 4. predict
    y_pred = model.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))

    # 5. visualization the decision edge
    plt.figure(figsize=(6, 5))

    # 绘制散点
    plt.scatter(X0[:, 0], X0[:, 1], color="blue", label="Class 0")
    plt.scatter(X1[:, 0], X1[:, 1], color="red", label="Class 1")

    # 决策边界：w1*x1 + w2*x2 + b = 0
    w1, w2 = model.coef_[0]
    b = model.intercept_[0]

    x_line = np.linspace(0, 8, 100)
    y_line = -(w1 * x_line + b) / w2

    plt.plot(x_line, y_line, 'k--', label="Decision Boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression Decision Boundary (sklearn)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    sklearn_lr_example()


if __name__ == "__main__":
    main()
