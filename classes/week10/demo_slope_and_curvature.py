#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/16/25
Description: demo_slope_and_curvature
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False


def demo_slope_and_curvature():
    """
       演示斜率和曲率的基本概念
       """
    # 模拟 Elbow 曲线数据
    K = np.array([1, 2, 3, 4, 5, 6, 7])
    WCSS = np.array([1000, 400, 200, 150, 140, 135, 133])

    # 计算一阶导数 (斜率)
    first_derivative = np.diff(WCSS)

    # 计算二阶导数 (曲率)
    second_derivative = np.diff(first_derivative)

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    # ===== 子图1: 原始曲线 =====
    axes[0].plot(K, WCSS, 'bo-', linewidth=2, markersize=10, label='WCSS曲线')
    axes[0].set_xlabel('K值', fontsize=12)
    axes[0].set_ylabel('WCSS (误差)', fontsize=12)
    axes[0].set_title('原始曲线: WCSS vs K', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # 标注每个点的值
    for i, (k, w) in enumerate(zip(K, WCSS)):
        axes[0].annotate(f'{w}', xy=(k, w), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)

    # ===== 子图2: 一阶导数 (斜率) =====
    K_slope = K[1:]  # 斜率对应的K值
    axes[1].plot(K_slope, first_derivative, 'go-', linewidth=2,
                 markersize=10, label='斜率 (一阶导数)')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='y=0')
    axes[1].set_xlabel('K值', fontsize=12)
    axes[1].set_ylabel('斜率 (WCSS变化)', fontsize=12)
    axes[1].set_title('一阶导数: 下降速度 (斜率越负 = 下降越快)',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    # 标注斜率值
    for i, (k, s) in enumerate(zip(K_slope, first_derivative)):
        axes[1].annotate(f'{s:.0f}', xy=(k, s), xytext=(5, -15),
                         textcoords='offset points', fontsize=9)

    # ===== 子图3: 二阶导数 (曲率) =====
    K_curve = K[2:]  # 曲率对应的K值
    axes[2].plot(K_curve, second_derivative, 'ro-', linewidth=2,
                 markersize=10, label='曲率 (二阶导数)')

    # 找到肘部 (曲率最大值)
    elbow_idx = np.argmax(second_derivative)
    elbow_k = K_curve[elbow_idx]
    elbow_curvature = second_derivative[elbow_idx]

    axes[2].plot(elbow_k, elbow_curvature, 'y*', markersize=25,
                 label=f'肘部位置 (K≈{elbow_k})')
    axes[2].axvline(x=elbow_k, color='y', linestyle='--', alpha=0.5)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('K值', fontsize=12)
    axes[2].set_ylabel('曲率 (斜率变化)', fontsize=12)
    axes[2].set_title('二阶导数: 弯曲程度 (曲率最大 = 肘部)',
                      fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=11)

    # 标注曲率值
    for i, (k, c) in enumerate(zip(K_curve, second_derivative)):
        axes[2].annotate(f'{c:.0f}', xy=(k, c), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 打印数值结果
    print("=" * 60)
    print("数值分析:")
    print("=" * 60)
    print(f"{'K值':<8} {'WCSS':<12} {'斜率':<15} {'曲率':<10}")
    print("-" * 60)

    for i in range(len(K)):
        k_str = f"{K[i]}"
        wcss_str = f"{WCSS[i]:.0f}"

        if i < len(first_derivative):
            slope_str = f"{first_derivative[i]:.0f}"
        else:
            slope_str = "-"

        if i < len(second_derivative) + 1:
            if i >= 2:
                curve_str = f"{second_derivative[i - 2]:.0f}"
            else:
                curve_str = "-"
        else:
            curve_str = "-"

        print(f"{k_str:<8} {wcss_str:<12} {slope_str:<15} {curve_str:<10}")

    print("=" * 60)
    print(f"推荐的K值: {elbow_k + 1} (曲率最大处)")
    print("=" * 60)


def main():
    demo_slope_and_curvature()


if __name__ == "__main__":
    main()
