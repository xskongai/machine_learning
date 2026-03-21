
import numpy as np
import random

# step 1 import data ,and set parameter
print("step 1")
X = [
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
]

K = 2
max_iter = 100


# step 2 define euclidean distance
print("step 2")
def euclidean_distance(x1, x2):
    return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5


# step 3 initial the central point (random selection)
print("step 3")
random.seed(42)
centroids = random.sample(X, K)

# step 4 main loop(partition+update)
print("step 4")
for _ in range(max_iter):
    # 4.1 分配每个点到最近的中心
    clusters = [[] for _ in range(K)]
    for point in X:
        distances = [euclidean_distance(point, center) for center in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)

    # 4.2 更新中心点
    new_centrolids = []
    for cluster in clusters:
        if cluster:
            dim = len(cluster[0])
            mean = [sum(p[i] for p in cluster) / len(cluster) for i in range(dim)]
            new_centrolids.append(mean)
        else:
            new_centrolids.append(random.choice(X))
        # 4.3 判断是否收敛
        if all(euclidean_distance(c1, c2) < 1e-4 for c1, c2 in zip(centroids, new_centrolids)):
            break
    print("centroids", centroids)
    print("new_centrolids", new_centrolids)
    centroids = new_centrolids

# step 5 print output
print("step 5")
for i, cluster in enumerate(clusters):
    print("cluster", i)
    for point in cluster:
        print(" point", point)
print("final centroids", centroids)
