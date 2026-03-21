#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/13/25
Description: sigmoid_function
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def show_sigmoid():
    z = np.linspace(-10,10, 200)
    s = sigmoid(z)

    plt.figure(figsize=(8,5))
    plt.plot(z,s,label = "Sigmoid(z)",color = "blue")

    plt.scatter(0,sigmoid(0),color = 'red',label = "z=0,Sigmoid =0.5")

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    plt.axhline(1, color="gray", linestyle="--", linewidth=0.7)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.7)

    plt.title("Sigmoid function curve")
    plt.xlabel("z")
    plt.ylabel("Sigmoid(z)")
    plt.legend()
    plt.grid(True)

    plt.show()



def main():
    show_sigmoid()


if __name__ == "__main__":
    main()
