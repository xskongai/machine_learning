#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: cv_find_dt_best_max_depth
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def find_dt_best_max_depth():
    # 1. load ddata iris
    data = load_iris()
    X, y = data.data, data.target

    # 2. create k fold object
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 3. create candidates of hyperparameter max_depth
    #     None means without limitation
    candidate_depth = [1, 2, 3, 4, 5, None]

    # 4. iterate candidate ,generate model, get scores using cross_val_scores, add mean score
    mean_scores = []
    for depth in candidate_depth:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        print(f"max_depth = {depth},every fold :{scores},mean score:{scores.mean()}")

        mean_scores.append(scores.mean())

    # 5. find the best max_depth
    best_max_depth = candidate_depth[np.argmax(mean_scores)]
    print(f"the best max_depth:{best_max_depth}")
    # 6. train a new model using the best max_depth on the entire  dataset
    model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    model.fit(X, y)

    print("the final model has been trained, it can be used to predict")

    #     7.visualize the candidates and scores
    # convert None to string for plotting
    x_labels = [str(d) for d in candidate_depth]

    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, mean_scores, marker='o', linewidth=2)
    plt.title("Decision Tree: Max Depth vs Mean Accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Mean Accuracy")
    plt.grid(True)

    # highlight best depth
    best_index = np.argmax(mean_scores)
    plt.scatter(x_labels[best_index], mean_scores[best_index], s=150, edgecolors='red', facecolors='none', linewidths=2)
    plt.text(x_labels[best_index], mean_scores[best_index] + 0.01,
             f"best: {best_max_depth}", color="red", fontsize=12, ha='center')

    plt.show()


def main():
    find_dt_best_max_depth()


if __name__ == "__main__":
    main()
