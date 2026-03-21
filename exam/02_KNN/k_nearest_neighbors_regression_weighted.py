#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: k_nearest_neighbors_regression_weighted
"""

import math

from fontTools.misc.bezierTools import epsilon


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def knn_regression_weighted(X_train, y_train, x_new, k=3):
    # 1. compute distance from new point with samples
    distances = [(euclidean_distance(x, x_new), label) for x, label in zip(X_train, y_train)]

    # 2. sort by distance
    distances.sort(key=lambda pair: pair[0])

    # 3. select k nearest neighbors
    k_neighbors = distances[:k]

    # 4. compute the weighted  average y of k nearest neighbors
    numerator = sum(1 / (d + epsilon) * y for d, y in k_neighbors)
    denominator = sum(1 / (d + epsilon) for d, _ in k_neighbors)

    prediction = numerator / denominator

    return prediction


def main():
    print("--------------------------------test knn regressions,weighted-------------------------------------------")
    # 示例数据（二维特征）
    X_train = [
        [1, 2],
        [2, 3],
        [3, 1],
        [5, 4],
        [6, 1]
    ]
    y_train = [3.0, 3.5, 2.0, 5.0, 4.5]

    # 新数据点
    x_new = [3, 2]

    predicted_value = knn_regression_weighted(X_train, y_train, x_new, k=3)

    print(f"x_new :{x_new} predictions:{predicted_value} ")


if __name__ == "__main__":
    main()
