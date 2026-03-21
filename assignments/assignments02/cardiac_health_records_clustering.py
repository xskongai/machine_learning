#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 11/13/25
Description: cardiac_health_records_clustering
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_selection import f_oneway
from sklearn.mixture import GaussianMixture

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway

def explore_and_clean_data():
    #  1. read file
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    file_path = "Modified_Heart_failure_data.csv"
    df = pd.read_csv(file_path)
    print(df)

    # 2. compute missing data ratio of every column
    missing_info = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_ratio": df.isnull().mean() * 100
    })
    print(missing_info)  # the missing ratio is so regular ! 29,  9.698997%

    # 3.compute missing data ratio of every row
    row_missing_info = pd.DataFrame({
        "row_missing_count": df.isnull().sum(axis=1),
        "row_missing_ratio": df.isnull().mean(axis=1) * 100
    }).sort_values(by="row_missing_ratio", ascending=False)

    # 3.1. print the row missing info
    print(row_missing_info)  # the largest 5,

    # 3.2. print the row missing info  which has larger than 50 percent data shortage
    rows_with_many_missing = df[row_missing_info["row_missing_ratio"] > 50]
    print(rows_with_many_missing)

    # 3.3 group by row missing count
    row_missing_info_grouped = row_missing_info.groupby("row_missing_count").size()
    print(row_missing_info_grouped)

    # 4. remove invalid data entries that have 4 or more missing values
    df_cleaned = df[df.isnull().sum(axis=1) < 4]  # remove 5 records, left 295

    return df_cleaned


def pre_process_handle_missing_values(df):
    # compute

    # binary variables: Mode imputation
    mode_imputer = SimpleImputer(strategy="most_frequent")
    for col in ['anaemia', 'high_blood_pressure', 'smoking', 'diabetes', 'sex']:
        df[[col]] = mode_imputer.fit_transform(df[[col]])

    # data distribution is skewed,median imputation
    median_imputer = SimpleImputer(strategy="median")
    df[['ejection_fraction']] = median_imputer.fit_transform(df[['ejection_fraction']])

    # distribution close to normal, mean imputation
    mean_imputer = SimpleImputer(strategy="mean")
    df[['serum_sodium']] = mean_imputer.fit_transform(df[['serum_sodium']])

    # distribution is skewed , knn imputation based on similarity of samples
    # alter n_neighbors to 6,7,8,9 ,observe the results
    knn_imputer = KNNImputer(n_neighbors=5)
    df[['creatinine_phosphokinase']] = knn_imputer.fit_transform(df[['creatinine_phosphokinase']])

    num_cols = ["age", "platelets", "serum_creatinine"]
    df[num_cols] = median_imputer.fit_transform(df[num_cols])
    return df


def select_features(df, mode="all"):
    # drop irrelevant columns
    drop_cols = ['patient_id', 'time']
    df = df.drop(columns=drop_cols, errors="ignore")

    # all:select all features except from id and time
    if mode == "all":
        selected_df = df.copy()
    # core: select key features
    elif mode == "core":
        core_features = [
            "age",
            "ejection_fraction",
            "serum_creatinine",
            "serum_sodium",
            "diabetes",
            "high_blood_pressure"
        ]
        selected_df = df[core_features].copy()
    else:
        raise ValueError("mode must be 'all' or 'core'")

    return selected_df


def pre_process_feature_scaling(df):
    # Log transform for skewed features
    for col in ["creatinine_phosphokinase", "platelets"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])  # log(1+x)，

    # Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    #  return scaler for new data
    # return df_scaled, scaler
    # 转回 DataFrame，保持列名和索引
    df_scaled_new = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    # 返回 DataFrame 和 scaler
    return df_scaled_new, scaler


def remove_outliers_by_distance(df, labels, threshold=1):
    outlier_indices = []
    # iterate through each cluster
    for k in np.unique(labels):
        # extract all points in cluster k
        cluster_points = df[labels == k]

        # compute cluster centroid
        center = cluster_points.mean(axis=0)

        # compute Euclidean distance of each point to centroid
        distances = np.linalg.norm(cluster_points - center, axis=1)

        #  compute mean and std of intra-cluster distances
        mean_dist = distances.mean()
        std_dist = distances.std()

        # identify outliers
        for i, d in enumerate(distances):
            # if distance >mean+threshold *std ,mark as outlier
            if d > mean_dist + threshold * std_dist:
                idx = np.where(labels == k)[0][i]
                outlier_indices.append(idx)

    print(f"outliers detected:{len(outlier_indices)}")

    # remove outliers and return cleaned data
    # df_cleaned = np.delete(df, outlier_indices, axis=0)
    df_cleaned = df.drop(df.index[outlier_indices]).reset_index(drop=True)

    return df_cleaned, outlier_indices


def find_optimal_clusters(data, method="kmeans", max_k=10):
    """
    Find the optimal number of clusters using Silhouette Score (and WCSS for KMeans).
    使用轮廓系数（以及 KMeans 的 WCSS）寻找最佳簇数
    """

    wcss = []  # 保存簇内平方和 (仅 KMeans)
    silhouette_scores = []  # 保存轮廓系数

    for k in range(2, max_k + 1):
        if method == "kmeans":
            # Apply KMeans clustering
            # 使用 KMeans 聚类
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(data)
            wcss.append(model.inertia_)

        elif method == "hierarchical":
            # Apply Agglomerative clustering
            # 使用层次聚类
            model = AgglomerativeClustering(n_clusters=k, linkage="average")
            labels = model.fit_predict(data)
            wcss.append(np.nan)  # 层次聚类没有 WCSS，用 NaN 占位

        elif method == "gmm":
            # Apply Gaussian Mixture Models
            # 使用高斯混合模型
            model = GaussianMixture(n_components=k, random_state=42)
            labels = model.fit_predict(data)
            wcss.append(np.nan)  # GMM 没有 inertia，用 NaN 占位

        else:
            raise ValueError("method must be 'kmeans', 'hierarchical', or 'gmm'")

        # Compute Silhouette Score
        # 计算轮廓系数
        sil_score = silhouette_score(data, labels)
        silhouette_scores.append(sil_score)

        # Print results
        # 打印结果
        if method == "kmeans":
            print(f"k={k}, WCSS={model.inertia_:.2f}, Silhouette={sil_score:.3f}")
        else:
            print(f"k={k}, Silhouette={sil_score:.3f}")

    # Visualization
    # 可视化结果
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(2, max_k + 1), wcss, marker='o', label="WCSS (Elbow)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("WCSS (Elbow Method)")
    ax1.set_title(f"Elbow & Silhouette ({method})")

    ax2 = ax1.twinx()
    ax2.plot(range(2, max_k + 1), silhouette_scores, marker='s', color='red', label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score")

    fig.legend(loc="upper right")
    plt.show()

    # Select k with maximum Silhouette Score
    # 选择轮廓系数最大的簇数
    best_k = range(2, max_k + 1)[silhouette_scores.index(max(silhouette_scores))]
    print(f"最佳簇数 / Optimal number of clusters ({method}): {best_k}")
    return best_k


def apply_dimensionality_reduction(df, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(df)
        title_suffix = 'PCA'

    elif method == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=20
        )
        reduced = reducer.fit_transform(df)
        title_suffix = "t-SNE"

    elif method == "umap":
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        reduced = reducer.fit_transform(df)
        title_suffix = "UMAP"
    else:
        raise ValueError("method must be 'pca','tsne' or 'umap' ")

    return reduced, title_suffix


def kmeans_clustering(df, dim_reduce="pca", plot_name=""):
    best_k = find_optimal_clusters(df, method="kmeans")
    print(f"KMeans best k: {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(df)

    print("KMeans Silhouette Score:", silhouette_score(df, labels))
    print("KMeans Davies-Bouldin Index:", davies_bouldin_score(df, labels))

    # 降维
    reduced, title_suffix = apply_dimensionality_reduction(df, method=dim_reduce)

    # 可视化
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set1")
    plt.title(f"[{plot_name}]KMeans Clusters ({title_suffix} 2D)")
    plt.show()
    return labels


def kmeans_clustering_optimized(df, dim_reduce="pca", plot_name=""):
    """
    KMeans clustering after dimensionality reduction
    先降维再进行 KMeans 聚类
    """

    # Step 1: Dimensionality Reduction
    # 第一步：降维处理
    reduced, title_suffix = apply_dimensionality_reduction(df, method=dim_reduce)
    # Step 2: Find optimal number of clusters on reduced data
    # 第二步：在降维后的数据上寻找最佳簇数
    best_k = find_optimal_clusters(reduced, method="kmeans")
    print(f"KMeans best k (after {title_suffix}): {best_k}")

    # Step 3: Apply KMeans clustering on reduced data
    # 第三步：在降维后的数据上进行 KMeans 聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(reduced)

    # Step 4: Evaluate clustering performance
    # 第四步：评估聚类效果
    print("KMeans Silhouette Score (after reduction):", silhouette_score(reduced, labels))
    print("KMeans Davies-Bouldin Index (after reduction):", davies_bouldin_score(reduced, labels))

    # Step 5: Visualization
    # 第五步：可视化聚类结果
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set1")
    plt.title(f"[{plot_name}]KMeans Clusters optimized ({title_suffix} 2D)")
    plt.show()

    return labels


def hierarchical_clustering(df, dim_reduce="pca", plot_name=""):
    """
    Hierarchical clustering (Agglomerative) after dimensionality reduction
    先降维再进行层次聚类（使用平均连接法 Average Linkage）
    """

    reduced, title_suffix = apply_dimensionality_reduction(df, method=dim_reduce)
    # Step 2: Find optimal number of clusters on reduced data
    # 第二步：在降维后的数据上寻找最佳簇数
    best_k = find_optimal_clusters(reduced, method="hierarchical")
    print(f"Hierarchical best k (after {title_suffix}): {best_k}")

    # Step 3: Apply Agglomerative clustering with Average Linkage
    # 第三步：在降维后的数据上进行层次聚类，使用平均连接法
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="average")
    labels = hier.fit_predict(reduced)

    # Step 4: Evaluate clustering performance
    # 第四步：评估聚类效果
    print("Hierarchical Silhouette Score (after reduction):", silhouette_score(reduced, labels))
    print("Hierarchical Davies-Bouldin Index (after reduction):", davies_bouldin_score(reduced, labels))

    # Step 5: Visualization
    # 第五步：可视化聚类结果
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2")
    plt.title(f"[{plot_name}]Hierarchical Clusters ({title_suffix} 2D, Average Linkage)")
    plt.show()

    return labels


def dbscan_clustering(df, dim_reduce="pca", plot_name=""):
    """
    dbscan clustering after dimensionality reduction
    :param df: 
    :param dim_reduce: 
    :return: 
    """
    # step 1: dimensionality reduction
    reduced, title_suffix = apply_dimensionality_reduction(df, dim_reduce)

    # steop 2:apply dbscan clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(reduced)

    if len(set(labels)) > 1 and -1 not in set(labels):
        print("DBSCAN silhouette Score:", silhouette_score(reduced, labels, random_state=42))
        print("DBSCAN Davies-Bouldin Index:", davies_bouldin_score(reduced, labels))
    else:
        print("DBSCAN produced noise or only one cluster, metrics are not available")
        print("DBSCAN unique labels:", set(labels))

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='Set3')
    plt.title(f"[{plot_name}] DBSCAN Clusters ({title_suffix} 2D)")
    plt.show()

    return labels


def gmm_clustering(df, dim_reduce="pca", plot_name=""):
    """
    Gaussian Mixture Models(GMM) clustering after dimensionality reduction
    """
    reduced, title_suffix = apply_dimensionality_reduction(df, dim_reduce)
    best_k = find_optimal_clusters(reduced, method="gmm")
    print(f"GMM best k (after {title_suffix}): {best_k}")

    gmm = GaussianMixture(n_components=best_k, random_state=42)
    labels = gmm.fit_predict(reduced)

    print("GMM silhouette score", silhouette_score(reduced, labels, random_state=42))
    print("GMM Davies-Bouldin Index", davies_bouldin_score(reduced, labels))

    # Step 5: Visualization
    # 第五步：可视化聚类结果
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="Set2")
    plt.title(f"[{plot_name}]GMM Clusters ({title_suffix} 2D)")
    plt.show()

    return labels






def analyze_and_visualize_clusters(df, labels, clinical_vars=None):
    """
    功能 / Function:
    Analyze and visualize clustering results, including:
    - Add cluster labels
    - Compute mean and median summaries
    - Perform ANOVA tests
    - Plot boxplots and radar chart
    """

    # 1. 添加簇标签 / Add cluster labels
    df_clusters = df.copy()
    labels = np.array(labels).reshape(-1)  # 强制转为一维 / Force reshape to 1D
    if len(labels) != len(df_clusters):
        raise ValueError(f"Length mismatch: labels ({len(labels)}) vs df ({len(df_clusters)})")
    df_clusters["cluster"] = labels

    # 2. 计算簇均值和中位数 / Compute mean and median per cluster
    summary_mean = df_clusters.groupby("cluster").mean()
    summary_median = df_clusters.groupby("cluster").median()

    print("=== Mean features per cluster ===")
    print(summary_mean)
    print("\n=== Median features per cluster ===")
    print(summary_median)

    # 3. ANOVA 显著性检验 / ANOVA significance test
    print("\n=== ANOVA test between clusters ===")
    for var in clinical_vars:
        groups = [df_clusters[df_clusters["cluster"] == c][var].dropna().values
                  for c in df_clusters["cluster"].unique()]
        valid_groups = [g for g in groups if len(g) > 1 and np.std(g) > 0]

        if len(valid_groups) >= 2:
            f_stat, p_val = f_oneway(*valid_groups)
            print(f"{var}: F={f_stat:.3f}, p={p_val:.4f}")
        else:
            print(f"{var}: Not enough data or zero variance, skipping ANOVA")

    # 4. 绘制组合箱线图 / Draw combined boxplots
    n_vars = len(clinical_vars)
    n_cols = 3   # 每行放3个图 / 3 plots per row
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, var in enumerate(clinical_vars):
        sns.boxplot(x="cluster", y=var, data=df_clusters, palette="Set2", ax=axes[i])
        axes[i].set_title(f"{var} distribution across clusters")

    # 删除多余的空子图 / Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # 5. 绘制雷达图 / Draw radar chart
    categories = clinical_vars
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合 / Close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for cluster_id in summary_mean.index:
        values = summary_mean.loc[cluster_id, categories].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {cluster_id}")
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar chart of clinical features across clusters")
    ax.legend()
    plt.show()

    # 6. 返回结果字典 / Return results dictionary
    results = {
        "mean_summary": summary_mean,
        "median_summary": summary_median
    }
    return results




def main():
    #  1. explore and preprocess
    df_cleaned = explore_and_clean_data()
    df_imputed = pre_process_handle_missing_values(df_cleaned)
    df_selected = select_features(df_imputed, mode="core")
    df_scaled, _ = pre_process_feature_scaling(df_selected)

    # 2. remove expectional point (when the score is too low, add after scoring)
    kmeans = KMeans(n_clusters=4, random_state=42)
    initial_labels = kmeans.fit_predict(df_scaled)
    df_no_outliers, _ = remove_outliers_by_distance(df_scaled, initial_labels)

    # 3. clustering and evaluation
    # try different approaches about dimensionality reduction: cpa->t-sne->umap
    # try different approaches about clustering : kmeans,hierarchical,dbscan,gmm

    dim_reduce = "tsne"

    clinical_vars = [
        "age",
        "ejection_fraction",
        "serum_creatinine",
        "serum_sodium",
        "diabetes",
        "high_blood_pressure"
    ]

    labels = kmeans_clustering(df_no_outliers, dim_reduce=dim_reduce, plot_name="Plot 1")

    # 4. analyze and visualize clusters
    results = analyze_and_visualize_clusters(df_no_outliers, labels, clinical_vars)

    labels2 = kmeans_clustering_optimized(df_no_outliers, dim_reduce=dim_reduce, plot_name="Plot 2")
    results2 = analyze_and_visualize_clusters(df_no_outliers, labels2, clinical_vars)

    labels3 = hierarchical_clustering(df_no_outliers, dim_reduce=dim_reduce, plot_name="Plot 3")
    results3 = analyze_and_visualize_clusters(df_no_outliers, labels3, clinical_vars)

    labels4 = dbscan_clustering(df_no_outliers, dim_reduce=dim_reduce, plot_name="Plot 4")
    results4 = analyze_and_visualize_clusters(df_no_outliers, labels4, clinical_vars)

    labels5 = gmm_clustering(df_no_outliers, dim_reduce=dim_reduce, plot_name="Plot 5")
    results5 = analyze_and_visualize_clusters(df_no_outliers, labels5, clinical_vars)


if __name__ == "__main__":
    main()
