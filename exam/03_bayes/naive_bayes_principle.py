#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, Counter
import math


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}             # P(C)
        self.word_counts = {}              # count(word, C)
        self.class_total_words = {}        # total words in class
        self.vocab = set()                 # all words

    def fit(self, X, y):
        """
        X: list of documents (each is a list of tokens)
        y: list of labels
        """
        # 先统计每个类的文档数量（用于计算先验）
        class_counts = Counter(y)
        total_docs = len(y)

        # 存储先验概率 P(C)
        self.class_priors = {c: class_counts[c] / total_docs for c in class_counts}

        # 初始化 count(word, C)
        self.word_counts = {c: defaultdict(int) for c in class_counts}
        self.class_total_words = {c: 0 for c in class_counts}

        # 统计词频
        for tokens, c in zip(X, y):
            for w in tokens:
                self.word_counts[c][w] += 1
                self.class_total_words[c] += 1
                self.vocab.add(w)

    def predict(self, tokens):
        """
        对单个文档进行预测
        返回最高后验概率的类别
        """
        vocab_size = len(self.vocab)

        log_posteriors = {}

        for c in self.class_priors:
            # log(先验)
            log_prob = math.log(self.class_priors[c])

            for w in tokens:
                # 拉普拉斯平滑: (count + 1) / (total + V)
                count_wc = self.word_counts[c].get(w, 0)
                total_c = self.class_total_words[c]

                cond_prob = (count_wc + 1) / (total_c + vocab_size)
                log_prob += math.log(cond_prob)

            log_posteriors[c] = log_prob

        # 返回最高后验概率类别
        return max(log_posteriors, key=log_posteriors.get)


# ====================== 示例 ======================

X = [
    ["free", "win", "money"],             # Spam
    ["free", "free", "offer"],            # Spam
    ["meeting", "schedule", "project"],   # Ham
    ["project", "update", "meeting"],     # Ham
]

y = ["spam", "spam", "ham", "ham"]

# 训练模型
clf = NaiveBayesClassifier()
clf.fit(X, y)

# 预测新邮件
doc = ["free", "meeting"]
pred = clf.predict(doc)

print("Document:", doc)
print("Predicted Class:", pred)
