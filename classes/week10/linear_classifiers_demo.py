#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/13/25
Description: Demo of Perceptron vs Logistic Regression
"""


import numpy as np
from sklearn.linear_model import Perceptron,LogisticRegression

def train_perceptron(X,y):
    """
    训练感知机分类器
    输入X，特征矩阵，y 标签
    输出 训练好的模型，预测结果
    """
    model= Perceptron()
    model.fit(X,y)
    predictions = model.predict(X)
    return model,predictions

def train_logistic_regression(X,y):
    """
    训练逻辑回归分类器
    输入X,y(标签)
    输出：训练好的横暴，概率预测，分类预测
    """
    model = LogisticRegression()
    model.fit(X,y)
    probs = model.predict_proba(X)[:,1]
    predictions =model.predict(X)
    return model, probs,predictions

def main():
    X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [5, 5], [6, 6]])
    y = np.array([0, 0, 0, 0, 1, 1])  # 前四个点是类别0，后两个点是类别1

    # 感知机
    perc_model, perc_preds = train_perceptron(X, y)
    print("Perceptron predictions:", perc_preds)

    # 逻辑回归
    log_model, log_probs, log_preds = train_logistic_regression(X, y)

    print("Logistic Regression probabilities:", np.round(log_probs, 3))
    print("Logistic Regression predictions:", log_preds)


if __name__ == "__main__":
    main()
