#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: sigmoid_curve
"""
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_sigmoid():
    # 1.prepare data
    x = np.linspace(-10, 10, 400)
    y = sigmoid(x)

    #     2. plot
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid(True)
    plt.show()


def main():
    plot_sigmoid()


if __name__ == "__main__":
    main()
