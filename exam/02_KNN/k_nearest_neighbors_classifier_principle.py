#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: k_nearest_neighbors_priciple
"""
from collections import Counter


def eu_distance(P, data):
    distances = [(((P[0] - d[0]) ** 2 + (P[1] - d[1]) ** 2) ** 0.5, d[2]) for d in data]
    distances.sort(key=lambda pair: pair[0])
    return distances


def mh_distance(P, data):
    distances = [(abs(P[0] - d[0]) + abs(P[1] - d[1]), d[2]) for d in data]
    distances.sort(key=lambda pair: pair[0])
    return distances


def minkowski_distance(P, data, p):
    distances = [((abs(P[0] - d[0]) ** p + abs(P[1] - d[1]) ** p) ** (1 / p), d[2]) for d in data]
    distances.sort(key=lambda pair: pair[0])
    return distances


def knn_principle_demo():
    #  format:[x1,x2, category]
    data = [
        [0, 0, 0],
        [1, 1, 0],
        [2, 2, 1],
        [3, 3, 1]
    ]

    P = [1, 5, 1.5]
    k = 3

    # 1. compute distances between P and points with labels

    # distances = eu_distance(P, data)
    # distances = mh_distance(P, data)
    distances = minkowski_distance(P, data, p=1)
    distances = minkowski_distance(P, data, p=2)
    # 2. select k nearest distance  neighbors
    neighbors = distances[:k]

    # 3. vote them,get the label
    votes = [label for _, label in neighbors]
    prediction = Counter(votes).most_common(1)[0][0]
    return prediction


def main():
    print("------------------------test KNN priciple demo-------------------------------------")
    print("new data's label:", knn_principle_demo())


if __name__ == "__main__":
    main()
