from pyswarm import pso
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import GPy

from pyswarm import pso
from sklearn.metrics import mean_squared_error
import numpy as np
import GPy

def optimize_gaussian_multikernel(X_train, Y_train, X_test, Y_test, lb=[0.1, 0.1, 0.1], ub=[10, 10, 10], swarmsize=40, maxiter=30):
    """
    使用粒子群优化对高斯多核回归的核权重进行优化。

    参数:
    - X_train: ndarray, 训练集特征
    - Y_train: ndarray, 训练集标签
    - X_test: ndarray, 测试集特征
    - Y_test: ndarray, 测试集标签
    - lb: list, 粒子群优化的下界
    - ub: list, 粒子群优化的上界
    - swarmsize: int, 粒子群大小
    - maxiter: int, 最大迭代次数

    返回:
    - best_r2: float, 最优模型的 R² 值
    - best_rmse: float, 最优模型的 RMSE 值
    - best_mape: float, 最优模型的 MAPE 值
    - best_decision_variables: list, 最优模型的决策变量（核权重）
    - best_fitness: float, 最优模型的适应度值（MAPE）
    """
    best_mape = float('inf')
    best_r2 = None
    best_rmse = None
    best_decision_variables = None
    best_model = None

    def create_gp_model(weights):
        """创建高斯过程回归模型，使用多核函数组合。"""
        kernel = (GPy.kern.RBF(input_dim=X_train.shape[1], variance=weights[0], lengthscale=1.0) +
                  GPy.kern.Matern32(input_dim=X_train.shape[1], variance=weights[1], lengthscale=1.0) +
                  GPy.kern.Exponential(input_dim=X_train.shape[1], variance=weights[2], lengthscale=1.0))
        model = GPy.models.GPRegression(X_train, Y_train, kernel)
        model.Gaussian_noise.variance = 1e-8
        model.optimize()
        return model

    def objective_function(weights):
        """目标函数，用于优化 MAPE。"""
        nonlocal best_mape, best_r2, best_rmse, best_decision_variables, best_model
        try:
            model = create_gp_model(weights)
            Y_pred, _ = model.predict(X_test)
            # 计算评价指标
            mape = np.mean(np.abs((Y_pred - Y_test) / Y_test)) * 100
            rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
            r2 = r2_score(Y_test, Y_pred)

            # 更新最佳模型
            if mape < best_mape:
                best_mape = mape
                best_r2 = r2
                best_rmse = rmse
                best_decision_variables =  np.array(weights)
                best_model = model
            return mape
        except np.linalg.LinAlgError:
            return np.inf

    # 调用粒子群优化
    _, _ = pso(objective_function, lb, ub, swarmsize=swarmsize, maxiter=maxiter, debug=True)

    # 使用最佳模型进行预测
    Y_pred, _ = best_model.predict(X_test)

    # 返回最佳模型的指标和参数
    return Y_pred, best_r2, best_rmse, best_mape, best_decision_variables, best_mape
