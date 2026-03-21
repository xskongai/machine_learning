#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: linear_regression_house_price_example
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression_house_price():
    #     1. define the data
    X = np.array([[50], [60], [70], [80], [90]], dtype=float)
    y = np.array([160, 190, 220, 250, 280], dtype=float)

    # 2. create model and fit
    model = LinearRegression()
    model.fit(X, y)

    # 3. print model parameters
    print("w (coefficient):", model.coef_)
    print("b (intercept):", model.intercept_)

    # 4. predict new value
    text_x = np.array([[120], [150]])
    pred_price = model.predict(text_x)
    print(f"new value :{text_x},predict prices are {pred_price}")

    # 6. predict all training data for plot
    y_pred = model.predict(X)

    # 5. evaluation
    print("MSE:", mean_squared_error(y, y_pred))
    print("R2 score:", r2_score(y, y_pred))

    # 7. visualization
    plt.scatter(X, y, color="blue", label="Training Data")
    plt.plot(X, y_pred, color="red", label="Fitted Line")
    plt.xlabel("Area (sq meters)")
    plt.ylabel("Price (10k RMB)")
    plt.title("Linear Regression with sklearn")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    linear_regression_house_price()


if __name__ == "__main__":
    main()
