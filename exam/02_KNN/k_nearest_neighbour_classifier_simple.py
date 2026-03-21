#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: k_nearest_neighbour_simple
"""
from sklearn.neighbors import KNeighborsClassifier


def knn_simple_examle():
    #     1. dataset2
    X = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    y = [0, 0, 1, 1, 1]
    # 2. create knn classifier and train data
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # 3. predict new data
    new_sample = [[1.5, 1.5]]
    pred = knn.predict(new_sample)

    print("new sample,classify result:", pred)


def main():
    print("-----------------KNN 测试--------------------")
    knn_simple_examle()


if __name__ == "__main__":
    main()
