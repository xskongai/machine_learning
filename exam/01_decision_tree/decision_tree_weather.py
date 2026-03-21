#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: decision_tree_weather
"""
from xml.sax.handler import feature_namespaces

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt


def decision_tree_weather_demo():
    #  1. construct  datasets
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy',
                    'Sunny',
                    'Overcast', 'Overcast', 'Rainy'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'High',
                     'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    print("Raw Data:\n", df)

    # 2. label encoding : convert the char to the num
    encoder = LabelEncoder()
    for col in ['Outlook', 'Humidity', 'Play']:
        df[col] = encoder.fit_transform(df[col])
    print("\nEncoded Data:\n", df)

    # 3. prepare training set
    X = df[['Outlook', 'Humidity', 'Windy']]
    y = df['Play']  # 修正这里

    # 4. fit decision tree with Entropy

    clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01, random_state=42)
    clf.fit(X, y)

    # 5. visualize the decision tree
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf,
                   feature_names=X.columns,
                   class_names=['No', 'Yes'],
                   filled=True)
    plt.title("Decision Tree for Weather → Play Tennis")
    plt.show()
    # 6. make predictions
    #       Sunny = 2, high = 0,windy = False
    test_sample = [[2, 0, False]]
    pred = clf.predict(test_sample)
    # 7. print decision tree rules

    rules = export_text(clf, feature_names=list(X.columns))
    print("\n Decision Tree Rules:\n", rules)


def main():
    decision_tree_weather_demo()


if __name__ == "__main__":
    main()
