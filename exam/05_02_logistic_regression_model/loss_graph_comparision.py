#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: loss_graph_comparision
"""

import numpy as np
import matplotlib.pyplot as plt


def loss_function_graph():
    # x 当成误差或 logit 输入
    x = np.linspace(-3, 3, 400)

    # 1. MSE
    mse = x ** 2

    # 2. MAE
    mae = np.abs(x)

    # 3. Huber Loss（δ = 1）
    delta = 1
    huber = np.where(np.abs(x) <= delta,
                     0.5 * x ** 2,
                     delta * (np.abs(x) - 0.5 * delta))

    # 4. Cross Entropy（针对 y = 1）
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    ce = -np.log(sigmoid(x))

    # CE 会爆炸，剪裁到可视化区间
    ce_clipped = np.clip(ce, 0, 6)

    # ------------------ 绘图 ------------------
    plt.figure(figsize=(8, 8))  # 正方形画布更适合 1:1 显示

    plt.plot(x, mse, label="MSE Loss", linewidth=2)
    plt.plot(x, mae, label="MAE Loss", linewidth=2)
    plt.plot(x, huber, label="Huber Loss (δ=1)", linewidth=2)
    plt.plot(x, ce_clipped, label="Cross Entropy (clipped)", linewidth=2)

    # 设置坐标范围一致
    plt.xlim(-3, 3)
    plt.ylim(0, 6)

    # ★★★ 保持 1:1 比例 ★★★
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title("Loss Functions Comparison (1:1 Scale)", fontsize=16)
    plt.xlabel("Error / Logit", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.show()


def main():
    loss_function_graph()


if __name__ == "__main__":
    main()
