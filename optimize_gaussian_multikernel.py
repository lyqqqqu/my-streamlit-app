# from pyswarm import pso
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import GPy
import streamlit as st
from PSO import pso
from GA import ga_optimize
from deap import base, creator, tools, algorithms

def  PSO_gaussian_multikernel(X_train, y_train, X_test, y_test, lb, ub, swarmsize=40, maxiter=30):
 
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
    # #  调用粒子群优化
    # for iter in range(maxiter):
    #     st.write(f"开始迭代 {iter+1}/{maxiter}")
    #     # 记录本代所有粒子的适应度
    #     iter_fitness = []
    #     for _ in range(swarmsize):
    #         weights = np.random.uniform(low=lb, high=ub)
    #         fitness = objective_function(weights)
    #         iter_fitness.append(fitness)
    #     # 记录本代的最佳适应度
    #     best_iter_fitness = min(iter_fitness)
    #     best_per_iter.append(best_iter_fitness)
    #     st.write(f"迭代 {iter+1} 的最佳适应度 (MAPE): {best_iter_fitness}")
    
    # # 使用最佳模型进行预测
    # y_pred, _ = best_model.predict(X_test)

    # # 返回最佳模型的指标和参数以及训练好的模型
    # return y_pred.flatten(), best_r2, best_rmse, best_mape, best_decision_variables.tolist(), best_per_iter, best_model

import numpy as np
import GPy
import GPyOpt
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import GPy
import GPyOpt
from sklearn.metrics import r2_score, mean_squared_error

def bayesian_gaussian_multikernel(X_train, y_train, X_test, y_test, lb, ub, max_iter=30):
    """
    使用SMBO框架中的贝叶斯优化对高斯多核回归的核权重进行优化，并记录每一代的最优值。

    参数:
    - X_train: ndarray, 训练集特征 (N x d)
    - Y_train: ndarray, 训练集标签 (N x 1)
    - X_test: ndarray, 测试集特征 (M x d)
    - Y_test: ndarray, 测试集标签 (M x 1)
    - lb: list, 权重搜索空间下界
    - ub: list, 权重搜索空间上界
    - max_iter: int, 贝叶斯优化迭代次数

    返回:
    - y_pred: ndarray, 测试集预测值 (M,)
    - best_r2: float, 最优模型的 R² 值
    - best_rmse: float, 最优模型的 RMSE 值
    - best_mape: float, 最优模型的 MAPE 值
    - best_decision_variables: ndarray, 最优模型的核权重参数
    - best_model: GPy.models.GPRegression, 最优训练好的模型
    - iteration_bests: list, 每一次迭代后的全局最佳MAPE记录（用于绘制适应度曲线）
    """

    # 确保 y_train 和 y_test 是二维数组
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # 定义创建高斯过程多核模型的函数
    def create_gp_model(weights):
        kernel = (GPy.kern.RBF(input_dim=X_train.shape[1], variance=weights[0], lengthscale=1.0) +
                  GPy.kern.Matern32(input_dim=X_train.shape[1], variance=weights[1], lengthscale=1.0) +
                  GPy.kern.Exponential(input_dim=X_train.shape[1], variance=weights[2], lengthscale=1.0))
        mean_function = GPy.mappings.Linear(input_dim=X_train.shape[1], output_dim=1)
        model = GPy.models.GPRegression(X_train, y_train, kernel, mean_function=mean_function)
        model.Gaussian_noise.variance = 1e-8
        model.optimize()
        return model

    # 定义目标函数（SMBO需要的目标函数形式）
    def objective_function(weights_array):
        # weights_array是形如(1, 3)的二维数组，这里取第一行作为参数
        weights = weights_array[0]
        try:
            model = create_gp_model(weights)
            y_pred, _ = model.predict(X_test)
            mape = np.mean(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) * 100
            # 返回目标值（贝叶斯优化想要最小化MAPE）
            return mape
        except np.linalg.LinAlgError:
            # 若出现数值问题，返回一个很大的值作为惩罚
            return 1e10

    # 定义搜索空间domain
    domain = []
    for i in range(len(lb)):
        domain.append({'name': f'var_{i}', 'type': 'continuous', 'domain': (lb[i], ub[i])})

    # 使用GPyOpt进行贝叶斯优化（SMBO）
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=domain,
        model_type='GP',
        acquisition_type='EI',  # 使用EI采集函数
        evaluator_type='sequential',
        maximize=False
    )

    # 运行优化
    optimizer.run_optimization(max_iter=max_iter)

    # 获取最优解
    best_decision_variables = optimizer.X[optimizer.Y.argmin()]
    best_model = create_gp_model(best_decision_variables)
    y_pred, _ = best_model.predict(X_test)

    best_mape = np.mean(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) * 100
    best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    best_r2 = r2_score(y_test, y_pred)

    # 记录每一代的全局最优
    iteration_bests = []
    current_best = float('inf')
    # optimizer.X和optimizer.Y中存储了所有评估过的点和目标值（MAPE）
    # 第0到第(max_iter-1)次评估分别对应每一代
    # 注意：GPyOpt默认初始会有一些初始点，如果有初始点，则总评估次数>max_iter
    # 您可以通过设置 initial_design_numdata=0 或根据需要调整初始设计大小
    # 假设初始设计点数量为0(即不会多评估初始点)
    # 若不是0，请根据initial_design_numdata减掉初始点个数后处理
    for i in range(optimizer.X.shape[0]):
        val = optimizer.Y[i][0]
        if val < current_best:
            current_best = val
        iteration_bests.append(current_best)

    return y_pred.flatten(), best_r2, best_rmse, best_mape, best_decision_variables, iteration_bests, best_model 

def GA_gaussian_multikernel(X_train, y_train, X_test, y_test, lb, ub, max_gen=30, pop_size=20, cxpb=0.5, mutpb=0.2):
    """
    使用GA优化高斯多核回归模型的核权重，并记录每一代的全局最优。

    参数:
    - X_train: ndarray, 训练集特征 (N x d)
    - Y_train: ndarray, 训练集标签 (N x 1)
    - X_test: ndarray, 测试集特征 (M x d)
    - Y_test: ndarray, 测试集标签 (M x 1)
    - lb: list, 权重搜索空间下界
    - ub: list, 权重搜索空间上界
    - max_gen: int, GA最大进化代数
    - pop_size: int, 种群大小
    - cxpb: float, 交叉概率
    - mutpb: float, 变异概率

    返回:
    - y_pred: ndarray, 测试集预测值 (M,)
    - best_r2: float, 最优模型的 R² 值
    - best_rmse: float, 最优模型的 RMSE 值
    - best_mape: float, 最优模型的 MAPE 值
    - best_decision_variables: ndarray, 最优模型的核权重参数
    - best_model: GPy.models.GPRegression, 最优训练好的模型
    - iteration_bests: list, 每一代的全局最佳MAPE记录
    """

    # 确保 y_train 和 y_test 是二维数组
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    def create_gp_model(weights):
        kernel = (GPy.kern.RBF(input_dim=X_train.shape[1], variance=weights[0], lengthscale=1.0) +
                  GPy.kern.Matern32(input_dim=X_train.shape[1], variance=weights[1], lengthscale=1.0) +
                  GPy.kern.Exponential(input_dim=X_train.shape[1], variance=weights[2], lengthscale=1.0))
        mean_function = GPy.mappings.Linear(input_dim=X_train.shape[1], output_dim=1)
        model = GPy.models.GPRegression(X_train, y_train, kernel, mean_function=mean_function)
        model.Gaussian_noise.variance = 1e-8
        model.optimize()
        return model

    def evaluate_fitness(weights):
        # weights: list or array of decision variables
        try:
            model = create_gp_model(weights)
            y_pred, _ = model.predict(X_test)
            mape = np.mean(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) * 100
            return mape
        except np.linalg.LinAlgError:
            return 1e10

    # 调用封装好的GA优化函数
    best_decision_variables, iteration_bests = ga_optimize(evaluate_fitness, lb, ub, max_gen=max_gen,
                                                           pop_size=pop_size, cxpb=cxpb, mutpb=mutpb)
    # 使用最优解建模
    best_model = create_gp_model(best_decision_variables)
    y_pred, _ = best_model.predict(X_test)

    best_mape = np.mean(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) * 100
    best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    best_r2 = r2_score(y_test, y_pred)

    return y_pred.flatten(), best_r2, best_rmse, best_mape, best_decision_variables,iteration_bests, best_model
