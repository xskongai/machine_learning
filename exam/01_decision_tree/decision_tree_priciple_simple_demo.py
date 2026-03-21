"""
本质上，决策树就是把“复杂问题拆成一系列简单的条件判断”，最终得到一个结果。它是机器学习里最直观、最像人类思维的算法之一。
"""


#  一个简单的决策树实现（递归）
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 特征索引
        self.threshold = threshold  # 阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶子节点的类别


def build_simple_tree():
    # 构建一个简单的决策树
    # 条件：身高 >170 适合打篮球，否则不适合
    root = DecisionTreeNode(feature=0, threshold=170)
    root.left = DecisionTreeNode(value='不适当打球')
    root.right = DecisionTreeNode(value='适合打球')
    return root


def predict(tree, x):
    # 递归预测
    if tree.value is not None:
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)


def main():
    print("-------------------------test simple decision tree ------------------------------------")
    tree = build_simple_tree()
    samples = [[165], [185], [175], [160]]

    for s in samples:
        value = predict(tree, s)
        print(f"height:{s},", value)


if __name__ == "__main__":
    main()
