#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 12/5/25
Description: decision_tree_priciple_demo_3features
"""


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 特征索引
        self.threshold = threshold  # 阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶子节点的类别


def build_three_feature_tree():
    #  height <170
    root = DecisionTreeNode(feature=0, threshold=170)
    root.left = DecisionTreeNode(value="不适合打球")
    #  weight <60
    weight_node = DecisionTreeNode(feature=1, threshold=60)
    root.right = weight_node
    weight_node.left = DecisionTreeNode(value="不适合打球")
    # age >30
    age_node = DecisionTreeNode(feature=2, threshold=30)
    weight_node.right = age_node
    age_node.right = DecisionTreeNode(value="不适合打球")
    age_node.left = DecisionTreeNode(value="适合打球")

    return root


def predict(tree, x):
    if tree.value is not None:
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)


def main():
    print("-------------------------test simple decision tree:with three features ------------------------------------")
    tree = build_three_feature_tree()
    samples = [
        [165, 55, 25],
        [180, 58, 28],
        [185, 75, 25],
        [190, 80, 35]
    ]
    for s in samples:
        result = predict(tree, s)
        print(f" sample:{s},decision tree result:{result}")


if __name__ == "__main__":
    main()
