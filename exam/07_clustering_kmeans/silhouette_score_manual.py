#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/7/25
Description: silhouette_score_manual
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def silhouette_score_manual(X, labels):
    n = X.shape[0]

    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k <= 1 or k >= n:
        raise ValueError("Number of clusters must be between 2 and n-1.")

    # 1. compute distance matrix
    dist_matrix = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

    silhouettes = np.zeros(n)

    for i in range(n):
        own_cluster = labels[i]

        # -2. a(i): within clusters mean distance
        same_cluster_idx = np.where(labels == own_cluster)[0]
        same_cluster_idx = same_cluster_idx[same_cluster_idx != i]

        if len(same_cluster_idx) > 0:
            a_i = np.mean(dist_matrix[i, same_cluster_idx])
        else:
            a_i = 0

        # -3. b(i): outside clusters mean distance
        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue

            other_idx = np.where(labels == other_cluster)[0]
            avg_dist = np.mean(dist_matrix[i, other_idx])

            if avg_dist < b_i:
                b_i = avg_dist

        silhouettes[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(silhouettes), silhouettes


# -------------------------
# 手写 silhouette plot
# -------------------------
def plot_silhouette(X, labels):

    # ❗❗❗ 修复 1：拆包
    avg_s, s = silhouette_score_manual(X, labels)

    unique_clusters = np.unique(labels)

    plt.figure(figsize=(8, 6))

    y_lower = 0

    for c in unique_clusters:

        # ❗❗❗ 修复 2：s 是数组，因此需要 s[...] 而不是 tuple
        cluster_s = s[labels == c]

        cluster_s.sort()  # 从低到高画条形

        size = len(cluster_s)
        y_upper = y_lower + size

        plt.barh(
            range(y_lower, y_upper),
            cluster_s,
            edgecolor='none',
            label=f"Cluster {c}"
        )

        y_lower = y_upper

    # ❗❗❗ 修复 3：用 avg_s，而不是 np.mean(s)
    plt.axvline(avg_s, color='red', linestyle='--', label="Average Silhouette")

    plt.title("Silhouette Plot (Manual Implementation)")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Samples")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    np.random.seed(0)

    # 三簇测试数据
    X1 = np.random.randn(150, 2) + np.array([2, 2])
    X2 = np.random.randn(150, 2) + np.array([8, 3])
    X3 = np.random.randn(150, 2) + np.array([5, 8])
    X = np.vstack((X1, X2, X3))

    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(X)

    plot_silhouette(X, labels)


if __name__ == "__main__":
    main()
