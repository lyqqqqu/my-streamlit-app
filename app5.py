import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from data_augmentation_regression import bayes_bootstrap, generate_regressdata_SMOTE, generate_regressdata_GAN, generate_regressdata_GMM, generate_regressdata_LSTM,Bootstrap,neighbor_based_interpolation
from Regression import gaussian_multikernel_model
import time;
from smt.surrogate_models import RBF,QP,GENN
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from data_processing import load_data,convert_data,clean_data
from main_function import (train_model,plot_fitted_curve,plot_scatter,
                           plot_scatter,bayesian_optimize_model,genetic_algorithm_optimize_model)

from main_function import plot_fitness_curve,augment_data
from optimize_gaussian_multikernel import PSO_gaussian_multikernel,bayesian_gaussian_multikernel,GA_gaussian_multikernel
from main_function import Classification_Model,plot_classification_results,compute_metrics
# 页面设置
import matplotlib
# matplotlib.font_manager.fontManager.addfont('SimHei.ttf') #临时注册新的全局字体

matplotlib.font_manager.fontManager.addfont('SimHei.ttf') #临时注册新的全局字体

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

plt.rcParams['axes.unicode_minus']=False#用来正常显示负号
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

# plt.rcParams['axes.unicode_minus']=False#用来正常显示负号

st.set_page_config(page_title="多元回归预测和分类模块", layout="wide")
# 标志：仅在第一次加载页面时触发气球
if "balloons_shown" not in st.session_state:
    st.session_state["balloons_shown"] = False

if not st.session_state["balloons_shown"]:
    st.balloons()  # 显示气球
    st.session_state["balloons_shown"] = True  # 设置标志，确保只显示一次

# 在侧边栏显示注意事项
# Sidebar Title
# Sidebar Title
st.sidebar.title("碳纤维相关背景")

# Initialize session state variables for toggling display
if "show_process" not in st.session_state:
    st.session_state["show_process"] = False
if "show_data" not in st.session_state:
    st.session_state["show_data"] = False

# Toggle button for carbon fiber process image
if st.sidebar.button("碳纤维生产流程"):
    st.session_state["show_process"] = not st.session_state["show_process"]

# Toggle button for carbon fiber sample data and feature target image
if st.sidebar.button("碳纤维样例数据展示"):
    st.session_state["show_data"] = not st.session_state["show_data"]

# Display content based on session state
if st.session_state["show_process"]:
    st.write("### 碳纤维生产流程")
    st.image("碳纤维全流程.png", caption="碳纤维生产流程", width=1000)

if st.session_state["show_data"]:
    st.write("### 碳纤维样例数据展示")
    # Path to the fixed Excel file
    excel_file_path = "样例数据展示.xlsx"

    # Read the Excel file
    df = pd.read_excel(excel_file_path)

    # Display the first 5 rows of the data
    st.dataframe(df.head())

    # Display the feature target image with reduced width
    st.image("特征图.png", caption="特征目标示意图", width=600)
# 页面标题
st.title("多元回归预测和分类模块")

# 将页面分为三列布局
col1, col2, col3 = st.columns([3, 3, 3])

# 初始化 session_state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False
if 'converted' not in st.session_state:
    st.session_state.converted = False
# 左侧栏：数据输入和预处理
with col1:
     
        st.header("数据输入与处理")
        
        # 文件上传模块
        uploaded_file = st.file_uploader("选择一个 Excel 或 CSV 文件上传", type=["csv", "xlsx"])
        
        if uploaded_file:
                data = load_data(uploaded_file)
                # 使用 Markdown 显示黑体加大标题
              # 使用 Markdown 改变选择框标题的字体大小和加粗
                st.write("### 数据清洗")

                # 数据清洗选项
                clean_option = st.selectbox("是否进行数据清洗", ["否", "是"])  # 选择框的标签设置为空
                
                if clean_option == "是":
                    data = clean_data(data)
                
            # 数据预处理选项
                st.write("### 数据转换")
                preprocess_option = st.selectbox("### 是否进行数据转换:", ["否", "标准化", "归一化"])
                
                if preprocess_option in ["标准化", "归一化"]:
                    data = convert_data(st.session_state.data, preprocess_option)
                
                # 显示处理后的数据
                st.write("(处理好的)数据集预览：")
                st.dataframe(data.head())
                
                # 特征和标签选择
                st.subheader("选择特征和输出列")
                all_columns = data.columns.tolist()
                output_column = st.selectbox(
                    "选择输出列（默认最后一列）",
                    options=all_columns,
                    index=len(all_columns) - 1
                )
                feature_columns = st.multiselect(
                    "选择特征列（默认除输出列外所有列）",
                    options=[col for col in all_columns if col != output_column],
                    default=[col for col in all_columns if col != output_column]
                )
                
                if not feature_columns:
                    st.warning("请至少选择一个特征列进行建模！")
                    st.stop()  # 使用 st.stop() 停止脚本执行
                
                # 数据集划分比例选择
                st.subheader("数据集划分比例")
                train_ratio = st.number_input("训练集比例（如 0.7 表示 70%）", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                test_ratio = 1 - train_ratio
                st.write(f"当前设置：训练集比例为 {train_ratio:.2f}，测试集比例为 {test_ratio:.2f}")
                
                # 随机种子选择
                use_random_seed = st.checkbox("是否使用随机种子", value=False)
                if use_random_seed:
                    random_seed = st.number_input("输入随机种子", min_value=1, max_value=10000, value=42, step=1)
                else:
                    random_seed = None  # 不设置随机种子
                
                # 数据划分
                X = data[feature_columns]
                y = data[output_column]

                if not X.empty and not y.empty:
                # 判断是否为分类问题（标签为分类类型）
                    if y.dtype == 'object' or len(np.unique(y)) <= 20:
                        is_classification = True
                        # 编码标签
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y)
                        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        st.write("### 标签编码")
                        st.write(label_mapping)
                    else:
                        is_classification = False
                        y_encoded = y  # 对于回归任务，不需要编码
                
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_ratio, random_state=random_seed, stratify=y_encoded if is_classification else None
                    )
                # if not X.empty and not y.empty:
                #     X_train, X_test, y_train, y_test = train_test_split(
                #         X, y, test_size=test_ratio, random_state=random_seed
                #     )
                #     st.success("训练集和测试集划分完成。")
                #     # st.write("训练集示例：")
                #     # st.dataframe(X_train.head())
                #     # st.write("测试集示例：")
                #     # st.dataframe(X_test.head())
                else:
                    st.warning("特征列或输出列无效，无法划分数据集。")
    
def merge_data(X, y, augmented_X, augmented_y, flag):
    if flag == 1:
        augmented_y = augmented_y.values.reshape(-1, 1)
        data_train = pd.concat([X, y], axis=1)
        augmented_data = pd.concat([augmented_X, pd.DataFrame(augmented_y, columns=[output_column])], axis=1)
        data_train = pd.concat([data_train, augmented_data], ignore_index=True)
        return data_train[feature_columns].values, data_train[output_column].values
    elif flag == 2:
        augmented_y = augmented_y.reshape(-1, 1)
        y_numpy = y.to_numpy() if isinstance(y, pd.Series) else y
        data_train = np.hstack([X, y_numpy.reshape(-1, 1)])
        augmented_data = np.hstack([augmented_X, augmented_y])
        data_train = np.vstack([data_train, augmented_data])
        return data_train[:, :-1], data_train[:, -1]
    else:
        raise ValueError("无效的flag值")
        
# 中间栏：数据增强 -> 模型选择 -> 优化 -> 训练
with col2:
    st.header("数据增强、模型选择和优化")
    st.write("### 问题类型")
    problem_type = st.selectbox("选择问题类型", ["回归", "分类"])
    
    if problem_type == "回归":
        
        st.subheader("数据增强选项")
        
        # 是否启用数据增强，默认设置为“否”
        use_augmentation = st.radio("是否选择数据增强", ["否", "是"], index=0, key="use_augmentation")
        
        # 当选择“是”时，显示数据增强方法和参数
        if use_augmentation == "是":
            augmentation_method = st.selectbox("数据增强方法", options=["Bayesian Bootstrap", "Bootstrap","最近邻插值","GAN生成对抗网络", "GMM", "LSTM"], key="augment_method")
            augment_factor = st.slider("增强倍数", 1, 10, 3, key="augment_factor")
            flag_augmentation = 1
        else:
            st.write("数据增强未启用")
            flag_augmentation = 0  # 修改为0表示未启用
        
        # 模型选择和训练
        st.subheader("模型选择")
        model_type = st.selectbox("选择回归模型", options=["线性回归", "高斯过程", "SVM模型", "高斯多核回归","RBF神经网络","GENN","QP二次多项式"], key="model_select")
        
        # 特定模型参数输入
        model_options = {}
        if model_type == "GENN":
            hidden_layer_sizes = st.text_input("隐藏层结构（用英文逗号分隔，例如：10,10）", value="10,10")
            try:
                hidden_layer_sizes = [int(size.strip()) for size in hidden_layer_sizes.split(",")]
            except ValueError:
                st.error("隐藏层结构输入不合法，请使用逗号分隔的整数列表（例如：10,10）")
                hidden_layer_sizes = [10, 10]
            alpha = st.number_input("学习率 (alpha)", min_value=0.0001, max_value=1.0, step=0.0001, value=0.01, format="%.4f")
            num_epochs = st.number_input("训练迭代次数 (num_epochs)", min_value=1, max_value=1000, value=10)
            model_options = {
                "hidden_layer_sizes": hidden_layer_sizes,
                "alpha": alpha,
                "num_epochs": num_epochs,
                "is_normalize": False
            }
        
        # 记录开始时间
        start_time = time.time()

        st.subheader("模型优化")
        optimize_model = st.radio("是否优化模型", ["否", "是"])
        optimize_method = None  # 默认优化方法为空
        fitness_history = []  # 用于存储适应度值

        if optimize_model == "是":
            optimize_method = st.selectbox("选择优化方法", ["贝叶斯优化", "遗传算法", "粒子群算法"])
            if optimize_method == "贝叶斯优化":
                bayes_iterations = st.number_input("贝叶斯优化迭代次数", min_value=1, max_value=100, value=5)
            elif optimize_method in ["遗传算法", "粒子群算法"]:
                population_size = st.number_input("种群大小", min_value=1, max_value=100, value=10)
                max_iterations = st.number_input("最大迭代次数", min_value=1, max_value=100, value=20)
        
        if st.button("训练回归模型"):
            # try:
                # 显示训练状态
                status_text = st.empty()
                status_text.text("模型正在训练中...")
                
                # 创建进度条
                progress_bar = st.progress(0)  #1
                
                # 数据增强
                if flag_augmentation == 1:
                    progress_bar.progress(0.1)
                    augmented_data, flag = augment_data(augmentation_method, X_train, y_train, augment_factor)
                    augmented_X, augmented_y = augmented_data
                    X_train_aug, y_train_aug = merge_data(X_train, y_train, augmented_X, augmented_y, flag)
                    X_train, y_train = X_train_aug, y_train_aug
                    progress_bar.progress(0.3)
                else:
                    X_train_aug, y_train_aug = X_train, y_train  # 无数据增强
                
                  # 数据类型转换
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.to_numpy().astype(float)
                if isinstance(y_train, (pd.Series, pd.DataFrame)):
                    y_train = y_train.to_numpy().astype(float)
                if isinstance(X_test, pd.DataFrame):
                    X_test = X_test.to_numpy().astype(float)
                if isinstance(y_test, (pd.Series, pd.DataFrame)):
                    y_test = y_test.to_numpy().astype(float)
               
                # 模型优化和训练
                progress_bar.progress(0.4)  #2
                if optimize_model == "是":
                    if optimize_method == "贝叶斯优化":
                        # 针对特定模型的贝叶斯优化
                        # if model_type in ["线性回归", "高斯过程", "SVM模型", "RBF神经网络", "GENN", "QP二次多项式"]:
                        #     # 假设我们有一个通用的贝叶斯优化函数，可优化这些模型的超参数
                        #     best_params, fitness_history = bayesian_optimize_model(
                        #         X_train, y_train, model_type, bayes_iterations
                        #     )
                        #     model_options.update(best_params)
                        # else:
                        #     st.error("当前模型不支持贝叶斯优化")
                        #     raise NotImplementedError("贝叶斯优化尚未为该模型实现")
                        if model_type == "高斯多核回归":
                            lb = [0.1, 0.1, 0.1]  # 根据实际情况定义
                            ub = [1.0, 1.0, 1.0]
                            y_pred, best_r2, best_rmse, best_mape, best_decision_variables,fitness_history, best_model = bayesian_gaussian_multikernel(
                                 X_train, y_train, X_test, y_test, lb, ub, bayes_iterations)
                    
                    elif optimize_method == "遗传算法":
                        # 针对特定模型的遗传算法优化
                        # if model_type in ["线性回归", "高斯过程", "SVM模型", "RBF神经网络", "GENN", "QP二次多项式"]:
                        #     best_params, fitness_history = genetic_algorithm_optimize_model(
                        #         X_train, y_train, model_type, population_size, bayes_iterations
                        #     )
                        #     model_options.update(best_params)
                        # else:
                        #     st.error("当前模型不支持遗传算法优化")
                        #     raise NotImplementedError("遗传算法优化尚未为该模型实现")
                        if model_type == "高斯多核回归":
                            y_pred, best_r2, best_rmse, best_mape, best_decision_variables,fitness_history, best_model=GA_gaussian_multikernel(
                                                    X_train, y_train, X_test, y_test, 
                                                    lb=[0.1,0.1,0.1], ub=[10,10,10], 
                                                    max_gen=max_iterations, pop_size=population_size, 
                                                    cxpb=0.5, mutpb=0.2)
                     
                    elif optimize_method == "粒子群算法":
                        if model_type == "高斯多核回归":
                            # 调用您的 PSO 优化函数
                            # 确保训练和测试数据是 numpy 数组
                            X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
                            y_train = y_train.values.reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)
                            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                            y_test = y_test.values.reshape(-1, 1) if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test.reshape(-1, 1)

                            y_pred, best_r2, best_rmse, best_mape, best_decision_variables, fitness_history, best_model = PSO_gaussian_multikernel(
                                X_train, y_train, X_test, y_test,
                                lb=[0.1, 0.1, 0.1], ub=[10, 10, 10],
                                swarmsize=population_size,
                                maxiter=max_iterations
                            )
                           
                        else:
                            st.error("粒子群算法仅支持高斯多核回归模型")
                            raise NotImplementedError("粒子群算法仅实现于高斯多核回归模型")
                      # 将优化结果传递给 extra
                      
                    extra = {
                        "r2": best_r2,
                        "rmse": best_rmse,
                        "mape": best_mape,
                        "decision_variables": best_decision_variables,
                        "fitness_history": fitness_history,
                        "trained_model": best_model
                    }
                    if "trained_model" in extra and extra["trained_model"] is not None:
                        st.session_state['trained_model'] = extra["trained_model"]
                    else:
                        st.session_state['trained_model'] = None  # 只有在没有训练模型时设置为 None
                else:
                    # 没有优化的模型训练
                    y_pred, extra = train_model(model_type, X_train, y_train, X_test, y_test, model_options)
                
                progress_bar.progress(0.6)  #3
               

                
                
                progress_bar.progress(0.8)
                
                # 模型评估
                if  optimize_model == "是":
                    r2 = extra["r2"]
                    rmse = extra["rmse"]
                    mape = extra["mape"]
                    decision_variables = extra.get("decision_variables")
                else:
                    if extra and "r2" in extra:
                        r2 = extra["r2"]
                        rmse = extra["rmse"]
                        mape = extra["mape"]
                        decision_variables = extra.get("decision_variables")
                    else:
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # 更新进度条至100%
                progress_bar.progress(1.0)
                status_text.text("模型训练完成")
                
                # 展示模型结果
                st.write("回归模型评估结果:")
                st.write("R² 值:", r2)
                st.write("均方根误差 (RMSE):", rmse)
                st.write("平均绝对百分比误差 (MAPE):", mape)
                if model_type == "高斯多核回归" and optimize_method == "粒子群算法":
                    st.write("最佳决策变量 (核权重):", decision_variables)
                
                # 记录结束时间并计算总时长
                end_time = time.time()
                total_time = end_time - start_time
                st.write(f"总时长: {total_time:.2f} 秒")
                
                # 可视化结果
                plot_fitted_curve(y_test, y_pred)
                plot_scatter(y_test, y_pred)
                # 如果有适应度历史，绘制适应度曲线
                if optimize_model == "是" and fitness_history:                    
                     plot_fitness_curve(fitness_history)
            # except NotImplementedError as nie:
            #     st.error(f"功能未实现: {nie}")
            #     progress_bar.progress(0)
            #     status_text.text("模型训练失败")
            # except Exception as e:
            #     st.error(f"模型训练失败: {e}")
            #     progress_bar.progress(0)
            #     status_text.text("模型训练失败")

    elif problem_type == "分类":
        st.write("### 模型选择")
        model_type = st.selectbox("选择分类模型", options=["逻辑回归", "KNN", "决策树","bagging","随机森林","Extra Trees","AdaBoost","GBDT","VOTE模型","xgboost","lightgbm"])
          # 按钮 - 训练模型
        if st.button("训练模型"):
            with st.spinner("正在训练模型..."):
                model, y_pred, y_prob = Classification_Model(
                    model_type, X_train, y_train, X_test, y_test
                )
            st.success("模型训练完成！")
            
            # 计算评价指标
            metrics = compute_metrics(y_test, y_pred, y_prob)
            
            # 展示评价指标
            st.subheader("评价指标")
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['值'])
            st.table(metrics_df)
            
            # 绘制并展示混淆矩阵和ROC曲线
            fig_cm, fig_roc = plot_classification_results(y_test, y_pred, y_prob, title_suffix=f"({model_type})")
            
            st.subheader("混淆矩阵")
            st.pyplot(fig_cm)
            
            if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                st.subheader("ROC曲线")
                st.pyplot(fig_roc)
            
            # 可选：展示特征重要性（仅适用于树模型）
            if model_type in ["随机森林", "Extra Trees", "GBDT", "lightgbm", "xgboost", "VOTE模型"]:
                st.subheader("特征重要性")
                feature_imp = None
                if model_type == "lightgbm":
                    importance = model.feature_importances_
                    feature_imp = pd.DataFrame({
                        'feature': feature_columns,
                        'importance': importance
                    }).sort_values(by='importance', ascending=False)
                elif model_type == "xgboost":
                    importance = model.feature_importances_
                    
                    xgb.plot_importance(model)
                    feature_imp = pd.DataFrame({
                        'feature': feature_columns,
                        'importance': importance
                    }).sort_values(by='importance', ascending=False)
                elif model_type == "VOTE模型":
                    # 对于VotingClassifier，可以提取各个基模型的特征重要性并取平均
                    importances = []
                    for estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            importances.append(estimator.feature_importances_)
                    if importances:
                        avg_importance = np.mean(importances, axis=0)
                        feature_imp = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': avg_importance
                        }).sort_values(by='importance', ascending=False)
                else:
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        feature_imp = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': importance
                        }).sort_values(by='importance', ascending=False)
                
                if feature_imp is not None:
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='importance', y='feature', data=feature_imp.head(20), ax=ax_imp)
                    ax_imp.set_title('特征重要性')
                    st.pyplot(fig_imp)
                else:
                    st.write("当前模型不支持特征重要性展示。")
# 右侧栏：预测模块
with col3:
    st.header("预测模块")
    
    # 上传预测文件
    predict_file = st.file_uploader("选择预测数据文件 (CSV/Excel)", type=["csv", "xlsx"])
    
    if predict_file:
        if predict_file.name.endswith('.csv'):
            predict_data = pd.read_csv(predict_file)
        else:
            predict_data = pd.read_excel(predict_file)
        
        st.write("预测数据预览：")
        st.dataframe(predict_data.head())
        
        # 检查是否已经训练了模型
        if 'trained_model' in st.session_state and st.session_state['trained_model'] is not None:
            trained_model = st.session_state['trained_model']
            
            try:
                # 确保预测数据包含必要的特征
                missing_features = set(feature_columns) - set(predict_data.columns)
                if missing_features:
                    st.error(f"预测数据缺少以下特征列：{missing_features}")
                else:
                    # 进行预测
                    prediction = trained_model.predict(predict_data[feature_columns])
                    st.write("预测结果：")
                    predict_data['预测结果'] = prediction.flatten()  # 确保预测结果为一维
                    st.dataframe(predict_data)
                    
                    # 提供下载预测结果的功能
                    csv = predict_data.to_csv(index=False)
                    st.download_button(label="下载预测结果", data=csv, file_name='prediction_results.csv', mime='text/csv')
            except Exception as e:
                st.error(f"预测过程中出现错误: {e}")
        else:
            st.warning("请先训练模型，然后上传预测数据文件。")
