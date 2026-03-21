#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: kmeans_comparison_with_sklearn
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as SKMeans


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centers = None

    def fit(self, X):
        n_samples = X.shape[0]

        # Step 1: 随机初始化中心点
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centers = X[random_idx]

        for _ in range(self.max_iters):
            # Step 2: 分配
            labels = self._assign_clusters(X)

            # Step 3: 更新
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Step 4: 收敛判断
            if np.allclose(self.centers, new_centers):
                break

            self.centers = new_centers

        return labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)


def compare_kmeans():
    # ------------------------------
    # 数据生成
    # ------------------------------
    np.random.seed(42)

    cluster_1 = np.random.randn(100, 2) + np.array([2, 2])
    cluster_2 = np.random.randn(100, 2) + np.array([8, 3])
    cluster_3 = np.random.randn(100, 2) + np.array([5, 8])

    X = np.vstack((cluster_1, cluster_2, cluster_3))

    my_kmeans = KMeans(k=3)
    labels_my = my_kmeans.fit(X)

    sk_kmeans = SKMeans(n_clusters=3, random_state=42)
    labels_sk = sk_kmeans.fit_predict(X)

    # ------------------------------
    # 可视化对比
    # ------------------------------

    plt.figure(figsize=(14, 6))

    # ---- 左图：手写 KMeans ----
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels_my, cmap='viridis', s=30)
    plt.scatter(my_kmeans.centers[:, 0], my_kmeans.centers[:, 1],
                c='red', marker='X', s=200, label='Centroids')
    plt.title("Handwritten KMeans")
    plt.legend()
    plt.grid(True)

    # ---- 右图：sklearn KMeans ----
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels_sk, cmap='viridis', s=30)
    plt.scatter(sk_kmeans.cluster_centers_[:, 0], sk_kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, label='SKLearn Centroids')
    plt.title("sklearn KMeans")
    plt.legend()
    plt.grid(True)

    plt.show()


def main():
    compare_kmeans()


if __name__ == "__main__":
    main()
