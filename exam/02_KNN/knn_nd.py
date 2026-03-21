#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: knn_nd
"""
from collections import Counter


def minkowski_distance(point1, point2, p=2):
    return sum([abs(a - b) ** p for a, b in zip(point1, point2)]) ** (1 / p)


def knn_predict(X_train, y_train, x_new, k=3, p=2):
    distances = []
    for x, label in zip(X_train, y_train):
        d = minkowski_distance(x, x_new)
        distances.append((d, label))

    distances.sort(key=lambda pair: pair[0])
    k_neighbors = distances[:k]

    votes = [label for _, label in k_neighbors]

    prediction = Counter(votes).most_common(1)[0][0]

    return prediction


def main():
    # 训练数据（3维）
    X_train = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 3.0, 3.0],
        [8.0, 7.0, 6.0],
        [7.0, 8.0, 5.0]
    ]

    y_train = [0, 0, 0, 1, 1]

    x_new = [2.5, 3.0, 3.5]

    for p in [1, 2, 3, 4]:
        pred = knn_predict(X_train, y_train, x_new, k=3, p=p)
        print(f"Minkowski p={p} -> predictions: {pred}")


if __name__ == "__main__":
    main()
