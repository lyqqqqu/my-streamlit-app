import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
matplotlib.rcParams['axes.unicode_minus'] = False
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

from optimize_gaussian_multikernel import PSO_gaussian_multikernel

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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)
import lightgbm as lgb
import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def Classification_Model(model_type, X_train, y_train, X_test, y_test, options=None):
    """
    训练指定类型的分类模型，并进行预测。
    
    参数：
    - model_type: 模型类型（字符串）
    - X_train, y_train: 训练数据和标签
    - X_test, y_test: 测试数据和标签
    - options: 其他可选参数（未使用）
    
    返回：
    - model: 训练好的模型
    - y_pred: 预测标签
    - y_prob: 预测概率（如果适用）
    """
    model = None
    y_pred = None
    y_prob = None
    
    if model_type == "逻辑回归":
        model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "决策树":
        model = tree.DecisionTreeClassifier().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "bagging":
        model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "随机森林":
        model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=12, random_state=0).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "Extra Trees":
        model = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "AdaBoost": 
        model = AdaBoostClassifier(n_estimators=10).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "GBDT": 
        model = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            
    elif model_type == "VOTE模型":
        # 定义基模型
        Cmodel1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
        Cmodel2 = RandomForestClassifier(n_estimators=50, random_state=1)
        Cmodel3 = GaussianNB()

        # 创建VotingClassifier
        model = VotingClassifier(estimators=[
            ('lr', Cmodel1), 
            ('rf', Cmodel2), 
            ('gnb', Cmodel3)
        ], voting='soft')  # 使用'soft'投票以便于概率计算

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        
    
        
    elif model_type == "xgboost":
         model = XGBClassifier(
            booster='gbtree',
            objective='binary:logistic',
            eval_metric='logloss',
            gamma=1,
            min_child_weight=1.5,
            max_depth=5,
            reg_lambda=10,  # 对应 'lambda' 参数
            subsample=0.7,
            colsample_bytree=0.7,
            learning_rate=0.03,
            tree_method='hist',
            random_state=2017,
            verbosity=0,
            n_estimators=1000
        )
         model.fit(X_train, y_train)
         y_pred = model.predict(X_test)
         if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

    elif model_type == "lightgbm":
       # 创建LGBM多分类模型
        model = LGBMClassifier(
            boosting_type='gbdt', 
            objective='multiclass',  # 设置为多分类任务
            metric='multi_logloss',  # 多分类的评估标准
            # num_class=3,  # 目标类别数为3
            num_leaves=31, 
            learning_rate=0.05, 
            feature_fraction=0.9
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)  # 获取所有类别的概率

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    计算常用的分类评价指标，并返回为字典。
    
    参数：
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_prob: 预测概率（可选，主要用于计算ROC-AUC）
    
    返回：
    - metrics: 包含各项指标的字典
    """
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if y_prob is not None:
        # 判断是二分类还是多分类
        if y_prob.shape[1] == 2:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob[:,1])
        else:
            metrics['ROC-AUC (OvR)'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    return metrics


def plot_classification_results(y_true, y_pred, y_prob=None, title_suffix=""):
    """
    创建混淆矩阵和ROC曲线的Matplotlib图形对象。
    
    参数：
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_prob: 预测概率（可选，主要用于绘制ROC曲线）
    - title_suffix: 图表标题的后缀（可选）
    
    返回：
    - fig_cm: 混淆矩阵图形对象
    - fig_roc: ROC曲线图形对象（如果适用）
    """
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
    ax_cm.set_xlabel('预测标签')
    ax_cm.set_ylabel('真实标签')
    ax_cm.set_title(f'混淆矩阵 {title_suffix}')
    
    # ROC曲线（仅限二分类且提供预测概率）
    fig_roc = None
    if y_prob is not None and y_prob.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:,1])
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(6,5))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('假阳性率 (FPR)')
        ax_roc.set_ylabel('真阳性率 (TPR)')
        ax_roc.set_title(f'ROC曲线 {title_suffix}')
        ax_roc.legend(loc="lower right")
    
    return fig_cm, fig_roc