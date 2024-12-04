# from pyswarm import pso
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import GPy
import streamlit as st
from PSO import pso

def optimize_gaussian_multikernel(X_train, y_train, X_test, y_test, lb, ub, swarmsize=40, maxiter=30):
 
    # 使用粒子群优化对高斯多核回归的核权重进行优化。

    # 参数:
    # - X_train: ndarray, 训练集特征
    # - Y_train: ndarray, 训练集标签
    # - X_test: ndarray, 测试集特征
    # - Y_test: ndarray, 测试集标签
    # - lb: list, 粒子群优化的下界
    # - ub: list, 粒子群优化的上界
    # - swarmsize: int, 粒子群大小
    # - maxiter: int, 最大迭代次数

    # 返回:
    # - Y_pred: ndarray, 测试集预测值
    # - best_r2: float, 最优模型的 R² 值
    # - best_rmse: float, 最优模型的 RMSE 值
    # - best_mape: float, 最优模型的 MAPE 值
    # - best_decision_variables: list, 最优模型的决策变量（核权重）
    # - fitness_history: list, 适应度历史
   
    
    
    # 确保 y_train 和 y_test 是二维数组
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    
    best_mape = float('inf')
    best_r2 = None
    best_rmse = None
    best_decision_variables = None
    best_model = None
    fitness_history = []
    best_per_iter = []  # 用于记录每一代的最佳适应度值
    def create_gp_model(weights):
        """创建高斯过程回归模型，使用多核函数组合。"""
        kernel = (
            GPy.kern.RBF(input_dim=X_train.shape[1], variance=weights[0], lengthscale=1.0) +
            GPy.kern.Matern32(input_dim=X_train.shape[1], variance=weights[1], lengthscale=1.0) +
            GPy.kern.Exponential(input_dim=X_train.shape[1], variance=weights[2], lengthscale=1.0)
        )
        mean_function = GPy.mappings.Linear(input_dim=X_train.shape[1], output_dim=1)
        model = GPy.models.GPRegression(X_train, y_train, kernel, mean_function=mean_function)
        model.Gaussian_noise.variance = 1e-8
        model.optimize()
        return model

    def objective_function(weights):
        """目标函数，用于优化 MAPE。"""
        nonlocal best_mape, best_r2, best_rmse, best_decision_variables, best_model, fitness_history
        try:
            model = create_gp_model(weights)
            y_pred, _ = model.predict(X_test)
            # 计算评价指标
            mape = np.mean(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # 记录适应度历史
            fitness_history.append(mape)

            # 更新最佳模型
            if mape < best_mape:
                best_mape = mape
                best_r2 = r2
                best_rmse = rmse
                best_decision_variables = np.array(weights)
                best_model = model
            return mape
        except np.linalg.LinAlgError:
            return float('inf')

    # 调用粒子群优化
    optimal_weights, _, fitness = pso(objective_function, lb, ub, swarmsize=swarmsize, maxiter=maxiter, debug=True)

    # 使用最佳模型进行预测
    y_pred, _ = best_model.predict(X_test)
  
    # 返回最佳模型的指标和参数以及训练好的模型
    return y_pred.flatten(), best_r2, best_rmse, best_mape, best_decision_variables, fitness, best_model
    
    
