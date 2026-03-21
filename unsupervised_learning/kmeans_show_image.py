# 导入必要的库
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Step 1: 生成模拟数据（300个点，分布在4个簇中）
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 2: 使用 KMeans 聚类（设定簇数为4）
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Step 3: 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')  # 每个点根据所属簇着色
centers = kmeans.cluster_centers_  # 获取聚类中心
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6, marker='x')  # 显示中心点
plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()  # 显示图像而不是保存
