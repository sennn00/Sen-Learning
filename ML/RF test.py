import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_excel(r"C:\Users\sennn\OneDrive\桌面\大创数据\Multi_Type_HERs-main\Multi_Type_HERs-main\10features_for_ML.xlsx", sheet_name=0)

# 提取特征和目标列
X = data.iloc[1:500, 7:-1].values  # 特征
y = data.iloc[1:500, -1].values    # 目标

# 划分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=6)

# 在训练集上训练模型
rf_regressor.fit(X_train, y_train)

# 在训练集上预测（检查拟合情况）
y_train_pred = rf_regressor.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

# 在测试集上预测（评估泛化能力）
y_test_pred = rf_regressor.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 输出结果
print("=== 训练集 ===")
print(f"R²: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}\n")

print("=== 测试集 ===")
print(f"R²: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")

# 绘制训练集和测试集的真实值 vs 预测值
plt.figure(figsize=(12, 6))

# 训练集散点图
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6, label='Train')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--r', label='Perfect Fit')
plt.xlabel('Actual (Train)')
plt.ylabel('Predicted (Train)')
plt.title(f'Train Set: R² = {train_r2:.2f} MSE = {train_mse:.2f}')
plt.legend()

# 测试集散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.6, label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label='Perfect Fit')
plt.xlabel('Actual (Test)')
plt.ylabel('Predicted (Test)')
plt.title(f'Test Set: R² = {test_r2:.2f}  MSE = {test_mse:.2f}')
plt.legend()

plt.tight_layout()
plt.show()

# 特征重要性
feature_importance = rf_regressor.feature_importances_
feature_names = data.columns[7:-1].tolist()  # 对齐特征名称

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='skyblue')  # 横向条形图更易读
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
