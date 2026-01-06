import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_excel(r"C:\Users\sennn\OneDrive\桌面\大创数据\Multi_Type_HERs-main\Multi_Type_HERs-main\10features_for_ML.xlsx", sheet_name=0)
print(data)

# 提取特征和目标列
X = data.iloc[1:100, 7:-1].values
y = data.iloc[1:100, -1].values    # Column index 14 (15th column)
print(X)
# 创建随机森林回归模型
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=6)

# 训练模型并进行预测
rf_regressor.fit(X, y)
y_pred = rf_regressor.predict(X)

# 计算 R^2
r2 = r2_score(y, y_pred)

# 使用 LOOCV 计算 MSE
loocv = LeaveOneOut()
mse_scores = -cross_val_score(rf_regressor, X, y, cv=loocv, scoring='neg_mean_squared_error')

# 计算特征重要度
feature_importance = rf_regressor.feature_importances_
feature_names=data.columns[7:-1].tolist()

# 输出 R^2 和 LOOCV 的 MSE
print("R^2:", r2)
print("Mean LOOCV MSE:", np.mean(mse_scores))
print("Feature Importance:", feature_importance)

# 绘制散点图和拟合线
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red', label='Perfect Fit')
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs. Predicted Performance (R^2: {:.2f})'.format(r2_score(y, y_pred)))
plt.legend()
plt.show()

# 绘制特征重要度条形图
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance, tick_label=feature_names)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # 调整布局防止标签重叠
plt.show()
