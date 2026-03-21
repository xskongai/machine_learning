#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/13/25
Description: linear_classifier_comprehension_demo
"""

import numpy as np

def hard_threshold(scores,threshold):
    scores = np.array(scores)
    return np.where(scores >= threshold,1,0)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def soft_threshold(scores,threshold = 60,prob_threshold = 0.5,scale = 0.1):
    scores = np.array(scores)
    z = (scores - threshold)*scale
    probs = sigmoid(z)
    labels = (probs >= prob_threshold).astype(int)
    return probs, labels

def main():
    scores = [55,62,70,48,90]

    labels_hard = hard_threshold(scores,60)
    print("hard thresholds result:",labels_hard)

    probs,labels_soft = soft_threshold(scores)
    print("soft thresholds probability:",np.round(probs,3))
    print("soft thresholds results:",labels_soft)

    pass


if __name__ == "__main__":
    main()
