#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: knn_sklean_demo
"""
from sklearn.neighbors import KNeighborsRegressor


def knn_sklean_demo(X_train, y_train, x_new):
    print("--------------------------normal knn sklearn--------------------------------")
    knn_plain = KNeighborsRegressor(n_neighbors=3, weights='uniform')
    knn_plain.fit(X_train, y_train)
    pred_plain = knn_plain.predict(x_new)
    print(f"uniform weight,knn prediction:{pred_plain}")

    print("--------------------------weighted knn sklearn--------------------------------")
    knn_weighted = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn_weighted.fit(X_train, y_train)
    pred_weighted = knn_weighted.predict(x_new)
    print(f"distance weighted,knn prediction:{pred_weighted}")


def main():
    # ---------------- 示例数据 ----------------
    X_train = [
        [1, 2],
        [2, 3],
        [3, 1],
        [5, 4],
        [6, 1]
    ]
    y_train = [3.0, 3.5, 2.0, 5.0, 4.5]

    x_new = [[3, 2]]  # 注意 scikit-learn 需要二维列表
    knn_sklean_demo(X_train, y_train, x_new)


if __name__ == "__main__":
    main()
