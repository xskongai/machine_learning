#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: decision_tree_irish
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree


def dt_iris():
    # 1. load data sets
    iris = load_iris()
    X = iris.data
    y = iris.target

    #   2. train decsion tree model
    clf = DecisionTreeClassifier(max_depth=3, random_state=43)
    clf.fit(X, y)

    #     3. visualize the decision tree model
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.title("iris decision tree classifier")
    plt.show()


def main():
    print("-------------------decision tree: classifier,iris demo--------------------------------")
    dt_iris()


if __name__ == "__main__":
    main()
