import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_excel(r"C:\Users\sennn\OneDrive\桌面\大创数据\Multi_Type_HERs-main\Multi_Type_HERs-main\10features_for_ML.xlsx", sheet_name=0)

# 提取特征和目标列
X = data.iloc[1:200, 7:-1].values  # 特征（第8列到倒数第2列）
y = data.iloc[1:200, -1].values    # 目标（最后一列）

# 划分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建ETR模型
etr = ExtraTreesRegressor(
    n_estimators=100,  # 树的数量
    max_depth=10,    # 树的最大深度
    min_samples_split=2, # 分裂内部节点所需的最小样本数
    min_samples_leaf=1,
    random_state=6     # 随机种子
)

# 训练模型
etr.fit(X_train, y_train)

# 在训练集和测试集上预测
y_train_pred = etr.predict(X_train)
y_test_pred = etr.predict(X_test)

# 计算指标
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 打印结果
print("=== 训练集 ===")
print(f"R²: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}\n")

print("=== 测试集 ===")
print(f"R²: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")

# 可视化真实值 vs 预测值
plt.figure(figsize=(12, 6))

# 训练集
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6, label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--r', label='Perfect Fit')
plt.xlabel('Actual (Train)')
plt.ylabel('Predicted (Train)')
plt.title(f'Train Set: R² = {train_r2:.2f}')
plt.legend()

# 测试集
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.6, label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label='Perfect Fit')
plt.xlabel('Actual (Test)')
plt.ylabel('Predicted (Test)')
plt.title(f'Test Set: R² = {test_r2:.2f}')
plt.legend()

plt.tight_layout()
plt.show()

# 特征重要性
feature_importance = etr.feature_importances_
feature_names = data.columns[7:-1].tolist()

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Extra Trees Regressor Feature Importance')
plt.tight_layout()
plt.show()
