import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_excel(r"C:\Users\sennn\OneDrive\桌面\大创数据\Multi_Type_HERs-main\Multi_Type_HERs-main\10features_for_ML.xlsx", sheet_name=0)
print(data)

# 提取特征和目标列
X = data.iloc[1:100, 7:-1].values
y = data.iloc[1:100, -1].values    # Column index 14 (15th column)
print(X)


# 创建神经网络回归模型
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=6)

# 训练模型并进行预测
mlp_regressor.fit(X, y)
y_pred = mlp_regressor.predict(X)

# 计算 R^2
r2 = r2_score(y, y_pred)

# 使用 LOOCV 计算 MSE
loocv = LeaveOneOut()
mse_scores = -cross_val_score(mlp_regressor, X, y, cv=loocv, scoring='neg_mean_squared_error')

# 注意：MLPRegressor 不支持特征重要度，所以这个部分我们会省略

# 输出 R^2 和 LOOCV 的 MSE
print("R^2:", r2)
print("Mean LOOCV MSE:", np.mean(mse_scores))

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
# MLPRegressor 不支持特征重要度，所以这部分会被省略
