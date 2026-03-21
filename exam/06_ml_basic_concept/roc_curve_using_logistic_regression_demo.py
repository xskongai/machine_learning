#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/6/25
Description: roc_curve_using_logistic_regression_demo
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from unsupervised_learning.roc_curve import thresholds


def roc_curve():
    # --------------------------
    # 1. Create binary dataset
    # --------------------------
    np.random.seed(0)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # simple linear boundary

    # --------------------------
    # 2. Train-test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --------------------------
    # 3. Train logistic regression
    # --------------------------
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # --------------------------
    # 4. Get predicted probabilities
    # --------------------------
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    # --------------------------
    # 5. Compute ROC curve + AUC
    # --------------------------

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print("AUC:", roc_auc)

    # --------------------------
    # 6. Plot ROC curve
    # --------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
