import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from Regression import gaussian_multikernel_model
from data_augmentation_regression import (bayes_bootstrap, Bootstrap, neighbor_based_interpolation,
    generate_regressdata_GAN, generate_regressdata_GMM,
    generate_regressdata_LSTM)
from smt.surrogate_models import RBF,QP,GENN

from optimize_gaussian_multikernel import optimize_gaussian_multikernel
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
matplotlib.rcParams['axes.unicode_minus'] = False

# 假设 X_train, y_train, X_test, y_test, feature_columns, output_column 已在其他部分定义
# 如果未定义，请确保在使用前定义这些变量

def augment_data(method, X, y, factor):
    if method == "Bayesian Bootstrap":
        return bayes_bootstrap(X, y, factor), 1
    elif method == "Bootstrap":
        return Bootstrap(X, y, factor), 1
    elif method == "最近邻插值":
        return neighbor_based_interpolation(X, y, factor, n_neighbors=5), 1
    elif method == "GAN生成对抗网络":
        return generate_regressdata_GAN(X, y, factor), 2
    elif method == "GMM":
        return generate_regressdata_GMM(X, y, factor), 2
    elif method == "LSTM":
        augmented_X, augmented_y = generate_regressdata_LSTM(X, y, factor)
        if augmented_X.shape[0] != augmented_y.shape[0]:
            raise ValueError(f"augmented_X 和 augmented_y 的样本数不匹配：{augmented_X.shape[0]} != {augmented_y.shape[0]}")
        return (augmented_X, augmented_y), 2
    else:
        raise ValueError("未知的数据增强方法")



def train_model(model_type, X_train, y_train, X_test, y_test, options=None):
    if model_type == "线性回归":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        extra = {"trained_model": model}
        return y_pred, extra
    elif model_type == "高斯过程":
        model = GaussianProcessRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        extra = {"trained_model": model}
        return y_pred, extra
    elif model_type == "SVM模型":
        model = SVR()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        extra = {"trained_model": model}
        return y_pred, extra
    elif model_type == "RBF神经网络":
        model = RBF(d0=5)
        model.set_training_values(X_train, y_train)
        model.train()
        y_pred = model.predict_values(X_test)
        extra = {"trained_model": model}
        return y_pred, extra
    elif model_type == "QP二次多项式":
        model = QP()
        model.set_training_values(X_train, y_train)
        model.train()
        y_pred = model.predict_values(X_test)
        extra = {"trained_model": model}
        return y_pred, extra
    elif model_type == "GENN":
        model = GENN()
        for key, value in options.items():
            model.options[key] = value
        model.set_training_values(X_train, y_train)
        model.train()
        y_pred = model.predict_values(X_test)
        extra = {"trained_model": model}
        return y_pred, extra
    elif model_type == "高斯多核回归":
            X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_train = y_train.values.reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.values.reshape(-1, 1) if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test.reshape(-1, 1)

            y_pred,_, model = gaussian_multikernel_model(X_train, y_train.reshape(-1, 1), X_test)
            extra = {"trained_model": model}
            return y_pred, extra
    else:
        raise ValueError("未知的模型类型")


def plot_fitness_curve(best_per_iter):
    """
    绘制适应度曲线，每次迭代显示最佳（最低）适应度值。
    
    参数:
        - best_per_iter: 每一代的最佳适应度值列表
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(best_per_iter) + 1), best_per_iter, marker='o', linestyle='-')
    ax.set_title("适应度曲线",fontsize=16)
    ax.set_xlabel("迭代次数",fontsize=12)
    ax.set_ylabel("最佳适应度 (MAPE)",fontsize=12)
    ax.grid(True)
    st.pyplot(fig)


def plot_fitted_curve(y_test, y_pred):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(range(len(y_test)), y_test, '-', label="真实值", color="purple", alpha=0.7)
    ax1.plot(range(len(y_pred)), y_pred, '--', label="拟合值", color="#4682B4", alpha=0.9, marker='*')
    ax1.set_xlabel("样本索引", fontsize=12)
    ax1.set_ylabel("值", fontsize=12)
    ax1.set_title("测试集拟合曲线图", fontsize=16)
    ax1.legend(loc="upper right", fontsize=12)
    ax1.grid(alpha=0.5)
    st.pyplot(fig1)

def plot_scatter(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={"s": 80, "alpha": 0.6, "edgecolor": "k"}, 
                line_kws={"color": "purple", "lw": 2}, ax=ax)
    ax.set_title("测试集散点图", fontsize=16)
    ax.set_xlabel("真实值 (True Data)", fontsize=12)
    ax.set_ylabel("预测值 (Predicted Data)", fontsize=12)
    ax.legend(["真实和预测值", "拟合线"], loc="upper left", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

def genetic_algorithm_optimize_model(X_train, y_train, model_type, population_size, max_iterations):
    a = 1

def bayesian_optimize_model(X_train, y_train, model_type, bayes_iterations):
    a = 2

