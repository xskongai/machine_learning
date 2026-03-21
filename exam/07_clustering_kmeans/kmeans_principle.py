#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: kmeans_principle
"""

import numpy as np

import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centers = None

    def fit(self, X):
        # 1. initial k centers randomly
        n_samples = X.shape[0]
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centers = X[random_idx]

        for _ in range(self.max_iters):
            # 2. assign every sample to its nearest center
            labels = self._assign_clusters(X)

            # 3. update the center point
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers
        return labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:,np.newaxis] - self.centers, axis=2)
        # find the nearest center,
        return np.argmin(distances, axis=1)


def kmeans_example():
    # 1. prepare data
    np.random.seed(42)
    cluster_1 = np.random.randn(100, 2) + np.array([2, 2])
    cluster_2 = np.random.randn(100, 2) + np.array([8, 3])
    cluster_3 = np.random.randn(100, 2) + np.array([5, 8])

    X = np.vstack((cluster_1, cluster_2, cluster_3))

    # 2. create model ,train data
    kmeans = KMeans(k=3)
    labels = kmeans.fit(X)

    # 3. plot
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1],
                c='red', s=200, marker='X', label="Centroids")
    plt.title("K-Means Clustering (Handwritten Implementation)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    kmeans_example()


if __name__ == "__main__":
    main()
