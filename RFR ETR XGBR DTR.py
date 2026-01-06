import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 加载数据
data = pd.read_excel(
    r"C:\Users\sennn\OneDrive\桌面\大创数据\Multi_Type_HERs-main\Multi_Type_HERs-main\10features_for_ML.xlsx",
    sheet_name=0)
X_O = data.iloc[1:2000, 7:-1].values  # 特征
y_O = data.iloc[1:2000, -1].values  # 目标

def xgbr():
    """
    XGBoost回归模型训练和评估
    """
    # 数据分割（只进行一次）
    x_train, x_test, y_train, y_test = train_test_split(X_O, y_O, test_size=0.2, random_state=68)

    # 创建XGBoost回归模型
    xgbr = XGBRegressor(random_state=31, n_jobs=-1)

    # 定义XGBoost超参数网格
    xgbr_grid = {
        'n_estimators': [50, 100],  # 树的数量
        'max_depth': [3, 5, 7, 9],  # 树的最大深度
        'learning_rate': [0.01, 0.1, 0.2],  # 学习率
        'subsample': [0.8, 0.9, 1.0],  # 样本采样比例
        'colsample_bytree': [0.8, 0.9, 1.0]  # 特征采样比例
    }

    # 网格搜索调优
    xgbr_grid_cv = GridSearchCV(
        estimator=xgbr,
        cv=10,
        param_grid=xgbr_grid,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        return_train_score=True
    )

    # 模型训练
    model = xgbr_grid_cv.fit(x_train, y_train)

    # 输出最佳参数
    print("最佳参数:", model.best_params_)
    print("最佳交叉验证分数:", -model.best_score_)  # 转换为正数

    # 使用最佳模型进行预测
    xgbr_model = xgbr_grid_cv.best_estimator_
    xgbr_preds_train = xgbr_model.predict(x_train)
    xgbr_preds_test = xgbr_model.predict(x_test)

    # 训练集评估指标
    mae_train = mean_absolute_error(xgbr_preds_train, y_train)
    mse_train = mean_squared_error(y_train, xgbr_preds_train)
    R2_train = r2_score(y_train, xgbr_preds_train)

    print('\n=== 训练集性能 ===')
    print('Training R² = {:.3f}'.format(R2_train))
    print('Training MAE = {:.3f}'.format(mae_train))
    print('Training MSE = {:.3f}'.format(mse_train))
    print('Training RMSE = {:.3f}'.format(np.sqrt(mse_train)))

    # 测试集评估指标
    mae_test = mean_absolute_error(xgbr_preds_test, y_test)
    mse_test = mean_squared_error(y_test, xgbr_preds_test)
    R2_test = r2_score(y_test, xgbr_preds_test)

    print('\n=== 测试集性能 ===')
    print('Test R² = {:.3f}'.format(R2_test))
    print('Test MAE = {:.3f}'.format(mae_test))
    print('Test MSE = {:.3f}'.format(mse_test))
    print('Test RMSE = {:.3f}'.format(np.sqrt(mse_test)))

    # 计算过拟合程度
    overfitting_ratio = mse_test / mse_train if mse_train > 0 else float('inf')
    print(f'\n过拟合程度 (Test MSE / Train MSE): {overfitting_ratio:.3f}')

    plot_combined_results(y_train, xgbr_preds_train, y_test, xgbr_preds_test)

    return xgbr_model, xgbr_preds_train, xgbr_preds_test

def plot_combined_results(y_train_true, y_train_pred, y_test_true, y_test_pred):
    plt.figure(figsize=(8, 6))

    # 绘制训练集和测试集的散点
    plt.scatter(y_train_true, y_train_pred, c='blue', alpha=0.6, label='Train', s=50)
    plt.scatter(y_test_true, y_test_pred, c='red', alpha=0.6, label='Test', s=50, marker='x')

    # 绘制理想拟合线（y=x）
    max_val = max(np.max(y_train_true), np.max(y_test_true))
    min_val = min(np.min(y_train_true), np.min(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], '--k', linewidth=1, label='Ideal Fit')

    # 计算指标
    train_r2 = r2_score(y_train_true, y_train_pred)
    train_mse = mean_squared_error(y_train_true, y_train_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)
    test_mse = mean_squared_error(y_test_true, y_test_pred)

    # 添加指标标注
    plt.text(0.25, 0.85,
             f'Train R² = {train_r2:.3f}\nTrain MSE = {train_mse:.3f}',
             transform=plt.gca().transAxes, color='blue')
    plt.text(0.25, 0.75,
             f'Test R² = {test_r2:.3f}\nTest MSE = {test_mse:.3f}',
             transform=plt.gca().transAxes, color='red')

    # 图表装饰
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs Predicted Values (Train & Test)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return train_r2, test_r2, train_mse, test_mse

def etr_model_tuning(X_O, y_O):
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X_O, y_O, test_size=0.2, random_state=68)

    # 定义ETR模型和超参数网格
    parameters = {
        'n_estimators': [50, 51, 1],  # 树的数量
        'max_depth': [10, 20, 1],  # 树深度
        'min_samples_split': [2, 5, 10],  # 分裂内部节点所需最小样本数
        'min_samples_leaf': [1, 2, 4]  # 叶节点最小样本数
    }

    # 网格搜索调优
    grid = GridSearchCV(
        estimator=ExtraTreesRegressor(random_state=178),
        param_grid=parameters,
        scoring='r2',  # 优化目标为R²
        cv=10,  # 10折交叉验证
        n_jobs=-1  # 使用所有CPU核心
    )
    grid.fit(x_train, y_train)

    # 输出最佳参数
    print("最佳超参数组合:", grid.best_params_)
    print("最佳交叉验证R²:", grid.best_score_)

    # 使用最佳模型预测
    best_etr = grid.best_estimator_
    y_train_pred = best_etr.predict(x_train)
    y_test_pred = best_etr.predict(x_test)

    # 计算训练集和测试集指标
    def print_metrics(y_true, y_pred, label):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{label} R² = {r2:.3f}")
        print(f"{label} MAE = {mae:.3f}")
        print(f"{label} RMSE = {np.sqrt(mse):.3f}\n")

    print_metrics(y_train, y_train_pred, "训练集")
    print_metrics(y_test, y_test_pred, "测试集")

    # 可视化真实值 vs 预测值
    plot_combined_results(y_train, y_train_pred, y_test, y_test_pred)

    # 特征重要性分析
    feature_importance = best_etr.feature_importances_
    feature_names = data.columns[7:-1].tolist()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance, color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('ETR Feature Importance')
    plt.tight_layout()
    plt.show()

    return best_etr

def dtr_model(X_O, y_O):
    """决策树回归模型"""

    x_train, x_test, y_train, y_test = train_test_split(X_O, y_O, test_size=0.2, random_state=68)
    dtr = DecisionTreeRegressor(random_state=31)

    # 定义DTR超参数网格
    dtr_grid = {
        'max_depth': [5,10,15,33, 34,None, 1],
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
    }

    dtr_grid_cv = GridSearchCV(
        estimator=dtr,
        cv=10,
        param_grid=dtr_grid,
        n_jobs=-1,
        scoring='r2',
        verbose=1,
        return_train_score=True
    )

    model = dtr_grid_cv.fit(x_train, y_train)
    print("最佳参数:", model.best_params_)

    dtr_model = dtr_grid_cv.best_estimator_
    dtr_preds_train = dtr_model.predict(x_train)
    dtr_preds_test = dtr_model.predict(x_test)

    # 训练集评估指标
    mae_train = mean_absolute_error(dtr_preds_train, y_train)
    mse_train = mean_squared_error(y_train, dtr_preds_train)
    R2_train = r2_score(y_train, dtr_preds_train)

    print('\n=== 训练集性能 ===')
    print('Training R² = {:.3f}'.format(R2_train))
    print('Training MAE = {:.3f}'.format(mae_train))
    print('Training MSE = {:.3f}'.format(mse_train))
    print('Training RMSE = {:.3f}'.format(np.sqrt(mse_train)))

    # 测试集评估指标
    mae_test = mean_absolute_error(dtr_preds_test, y_test)
    mse_test = mean_squared_error(y_test, dtr_preds_test)
    R2_test = r2_score(y_test, dtr_preds_test)

    print('\n=== 测试集性能 ===')
    print('Test R² = {:.3f}'.format(R2_test))
    print('Test MAE = {:.3f}'.format(mae_test))
    print('Test MSE = {:.3f}'.format(mse_test))
    print('Test RMSE = {:.3f}'.format(np.sqrt(mse_test)))

    # 绘制结果
    plot_combined_results(y_train, dtr_preds_train, y_test, dtr_preds_test)

    return

def rfr_model(X_O, y_O):
    """随机森林回归模型"""
    print("=" * 50)
    print("随机森林回归模型 (RFR)")
    print("=" * 50)

    x_train, x_test, y_train, y_test = train_test_split(X_O, y_O, test_size=0.2, random_state=188)

    rfr = RandomForestRegressor(random_state=167)

    # 定义RFR超参数网格
    rfr_grid = {
        'n_estimators': list(range(173, 174, 1)),  # 树的数量
        'max_depth': list(range(27, 28, 1)),  # 树的最大深度
        # 'max_leaf_nodes': list(range(28, 30, 2)),  # 可选的叶节点参数
    }

    # 网格搜索调优
    rfr_grid_cv = GridSearchCV(
        estimator=rfr,
        param_grid=rfr_grid,
        cv=10,
        n_jobs=-1,
        scoring='r2',
        verbose=1
    )

    # 模型训练
    rfr_grid_cv.fit(x_train, y_train)
    print("最佳参数:", rfr_grid_cv.best_params_)

    # 使用最佳模型
    rfr_model = rfr_grid_cv.best_estimator_
    rfr_preds_train = rfr_model.predict(x_train)
    rfr_preds_test = rfr_model.predict(x_test)

    # 训练集评估指标
    mae_train = mean_absolute_error(y_train, rfr_preds_train)
    mse_train = mean_squared_error(y_train, rfr_preds_train)
    R2_train = r2_score(y_train, rfr_preds_train)

    print('\n=== 训练集性能 ===')
    print('Training R² = {:.4f}'.format(R2_train))
    print('Training MAE = {:.3f}'.format(mae_train))
    print('Training MSE = {:.3f}'.format(mse_train))
    print('Training RMSE = {:.3f}'.format(np.sqrt(mse_train)))

    # 测试集评估指标
    mae_test = mean_absolute_error(y_test, rfr_preds_test)
    mse_test = mean_squared_error(y_test, rfr_preds_test)
    R2_test = r2_score(y_test, rfr_preds_test)

    print('\n=== 测试集性能 ===')
    print('Test R² = {:.4f}'.format(R2_test))
    print('Test MAE = {:.3f}'.format(mae_test))
    print('Test MSE = {:.3f}'.format(mse_test))
    print('Test RMSE = {:.3f}'.format(np.sqrt(mse_test)))

    # 绘制结果
    train_r2, test_r2, train_mse, test_mse = plot_combined_results(
        y_train, rfr_preds_train, y_test, rfr_preds_test
    )

    # 特征重要性分析（可选）
    feature_names = data.columns[7:-1].tolist()
    feature_importance = rfr_model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance, color='lightgreen')
    plt.xlabel('Importance Score')
    plt.title('RFR Feature Importance')
    plt.tight_layout()
    plt.show()

    return {
        'model': rfr_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse
    }
# 运行函数
#etr_model_tuning(X_O, y_O)
#xgbr()
#dtr_model(X_O, y_O)
rfr_model(X_O, y_O)
