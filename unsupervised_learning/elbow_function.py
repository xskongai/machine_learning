import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1️⃣ 生成模拟数据：二维样本点，分布在3个中心附近
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# 2️⃣ 记录不同簇数下的 WCSS（簇内误差平方和）
wcss = []
k_values = range(1, 11)  # 尝试从1到10个簇

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ 就是 WCSS

# 3️⃣ 绘制 Elbow 图
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o')
plt.title("Elbow 方法选择最佳簇数")
plt.xlabel("簇数 k")
plt.ylabel("簇内误差平方和（WCSS）")
plt.xticks(k_values)
plt.grid(True)
plt.show()
