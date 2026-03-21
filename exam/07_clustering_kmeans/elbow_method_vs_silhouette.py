#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: elbow_method_vs_silhouette
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compare_elbow_method_silhouette():
    # 1. prepare data
    np.random.seed(42)

    cluster_1 = np.random.randn(200, 2) + np.array([2, 2])
    cluster_2 = np.random.randn(200, 2) + np.array([8, 3])
    cluster_3 = np.random.randn(200, 2) + np.array([5, 8])

    X = np.vstack((cluster_1, cluster_2, cluster_3))

    # 2. elbow and silhouette compute
    K_range = range(2, 10)

    wcss = []
    silhouette_vals = []
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)

        wcss.append(model.inertia_)
        silhouette_vals.append(silhouette_score(X, labels, random_state=42))

    # 3. visualization
    plt.figure(figsize=(14, 5))

    # ---- Elbow 曲线 ----
    plt.subplot(1, 2, 1)
    plt.plot(K_range, wcss, 'o--', color='blue')
    plt.title("Elbow Method (WCSS)")
    plt.xlabel("Number of clusters K")
    plt.ylabel("WCSS")
    plt.grid(True)

    # ---- Silhouette 曲线 ----
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_vals, 'o--', color='green')
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    compare_elbow_method_silhouette()


if __name__ == "__main__":
    main()
