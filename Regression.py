import GPy
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import numpy as np
import GPy

def gaussian_multikernel_model(X_train, Y_train, X_test):
    """
    使用RBF、Matern和Exponential核函数构建高斯过程多核模型，并进行预测。

    参数:
        - X_train: 训练数据的输入特征 (二维数组)
        - Y_train: 训练数据的输出标签 (二维数组)
        - X_test: 测试数据的输入特征 (二维数组)

    返回:
        - Y_pred: 模型对测试数据的预测值
        - results: 每个测试点的预测均值和标准差
    """
    # X_train = X_train.values  # 转换为 numpy 数组
    # y_train = y_train.values  # 转换为 numpy 数组
    # X_test = X_test.values    # 转换为 numpy 数组

    # 默认各核函数权重为1
    weights = [1.0, 1.0, 1.0]

    # 创建多核函数，权重分别应用于RBF、Matern和Exponential核函数
    kernel = (GPy.kern.RBF(input_dim=X_train.shape[1], variance=weights[0], lengthscale=1.0) +
              GPy.kern.Matern32(input_dim=X_train.shape[1], variance=weights[1], lengthscale=1.0) +
              GPy.kern.Exponential(input_dim=X_train.shape[1], variance=weights[2], lengthscale=1.0))

    # 创建线性均值函数
    mean_function = GPy.mappings.Linear(input_dim=X_train.shape[1], output_dim=1)

    # 构建高斯过程回归模型
    model = GPy.models.GPRegression(X_train, Y_train, kernel, mean_function=mean_function)
    
    # 设置非常小的噪声方差，防止模型过拟合
    model.Gaussian_noise.variance = 1e-8
    # print("X_train shape:", X_train.shape)
    # print("Y_train shape:", Y_train.shape)
    # print("X_test shape:", X_test.shape)

    # 优化模型参数
    model.optimize()

    # 获取预测的均值和方差
    Y_pred, Y_pred_var = model.predict(X_test)
    Y_pred_std = np.sqrt(Y_pred_var)  # 标准差是方差的平方根

    # 保存每个测试点的预测均值和标准差
    results = {"mean": Y_pred.flatten(), "std": Y_pred_std.flatten()}

    return Y_pred.flatten(), results

