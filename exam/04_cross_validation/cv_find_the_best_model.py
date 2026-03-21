#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: cv_find_the_best_model
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def find_the_best_model():
    # 1. load datasets
    data = load_iris()
    X, y = data.data, data.target
    # 2. define  KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 3. define candidates of model
    candidate_models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM ( linear kernel)": SVC(kernel='linear'),
        "SVM ( RBF kernel)": SVC(kernel='rbf')

    }

    # 4. iterate with different model ,record the  mean scores
    mean_scores = {}
    for name, model in candidate_models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        mean_scores[name] = scores.mean()

        print(f"{name}: every fold scores:{scores}, mean_scores:{scores.mean()}")

        # 5. fine best model
    best_model_name = max(mean_scores, key=mean_scores.get)
    print(f"the best model:{best_model_name}, average accuracy = {mean_scores[best_model_name]}")

    # 6.  create a new model with the best model
    best_model = candidate_models[best_model_name]
    best_model.fit(X, y)
    print("the best model has been trained ")

    # 7. visualize

    plt.figure(figsize=(10, 5))
    bars = plt.bar(mean_scores.keys(), mean_scores.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("Mean Accuracy")
    plt.title("Model Comparison (10-fold CV)")

    # highlight best model
    best_index = list(mean_scores.keys()).index(best_model_name)
    bars[best_index].set_color("orange")

    # ---- NEW PART: add score text above each bar ----
    for i, (model_name, score) in enumerate(mean_scores.items()):
        plt.text(i, score + 0.005, f"{score:.3f}",
                 ha='center', fontsize=10)

    # label best model
    plt.text(best_index,
             mean_scores[best_model_name] + 0.03,
             f"BEST: {mean_scores[best_model_name]:.3f}",
             ha='center', color='red', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def main():
    find_the_best_model()


if __name__ == "__main__":
    main()
