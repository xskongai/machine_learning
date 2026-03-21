# 📦 导入必要的库（Import required libraries）
from sklearn.datasets import load_breast_cancer  # 加载乳腺癌数据集
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import roc_curve, roc_auc_score  # ROC 曲线和 AUC
import matplotlib.pyplot as plt  # 绘图工具

# 📊 加载数据（Load dataset）
data = load_breast_cancer()
X = data.data  # 特征数据（Features）
y = data.target  # 标签数据（Labels）

# ✂️ 划分训练集和测试集（Split into train and test sets）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🤖 训练逻辑回归模型（Train logistic regression model）
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 📈 获取预测概率（Get predicted probabilities）
y_scores = model.predict_proba(X_test)[:, 1]  # 只取正类的概率

# 📐 计算 ROC 曲线数据（Compute ROC curve data）
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc_score = roc_auc_score(y_test, y_scores)

# 🎨 绘制 ROC 曲线（Plot ROC curve）
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')  # 随机猜测参考线
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
