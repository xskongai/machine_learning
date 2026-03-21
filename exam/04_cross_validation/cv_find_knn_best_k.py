#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: cv_find_knn_best_k
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


def find_knn_best_k():
    # 1. load data set  using iris
    data = load_iris()
    X, y = data.data, data.target

    # 2. define k-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 3. define the the candidates of hyper parameters
    candidate_k = [1, 3, 5, 7, 9, 11]

    # 4. define a dict to record the mean accuracy
    mean_scores = []

    # 5. run the model  and  generate  different scores of  candidates
    for k in candidate_k:
        model = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

        print(f"k = {k}, every fold result = {scores}, mean score = {scores.mean():.4f}")
        mean_scores.append(scores.mean())

    # 6. select the best mean score and the k
    best_k = candidate_k[np.argmax(mean_scores)]
    print(f"the best k:{best_k}")

    # 7. train a new model using the best k  on  all datasets
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X, y)

    print("the final model has trained the entire dataset, we can use it to predict ")

    # 8. NEW: plot K vs mean accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(candidate_k, mean_scores, marker='o', linestyle='-', linewidth=2)
    plt.title("KNN: Mean Accuracy vs K")
    plt.xlabel("K value")
    plt.ylabel("Mean Accuracy")
    plt.grid(True)
    plt.xticks(candidate_k)
    plt.show()


def main():
    find_knn_best_k()


if __name__ == "__main__":
    main()
