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
from optimize_gaussian_multikernel import optimize_gaussian_multikernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# 页面设置
st.set_page_config(page_title="多元回归预测和分类模块", layout="wide")
# 标志：仅在第一次加载页面时触发气球
if "balloons_shown" not in st.session_state:
    st.session_state["balloons_shown"] = False

if not st.session_state["balloons_shown"]:
    st.balloons()  # 显示气球
    st.session_state["balloons_shown"] = True  # 设置标志，确保只显示一次

# 在侧边栏显示注意事项
st.sidebar.title("注意事项")
st.sidebar.write("""
1. 请确保正确输入数据集之后，才能点击训练模型按钮。

2. 目前该程序仍存在很多问题，优化目前只有高斯多核回归和粒子群结合是有效的，后续将不断完善。
                 
3. 高斯多核回归和粒子群结合的决策变量是权重，适应度是误差mape

4. 如果遇到问题，可多尝试不同方法的组合解决，或反馈给我。
""")
# 页面标题
st.title("多元回归预测和分类模块")

# 将页面分为三列布局
col1, col2, col3 = st.columns([3, 3, 3])

# 左侧栏：数据输入和预处理
with col1:
    st.header("数据输入与处理")
      
    # 文件上传模块
    uploaded_file = st.file_uploader("选择一个 Excel 或 CSV 文件上传", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # 添加是否进行预处理的选择按钮
        preprocess_option = st.radio("是否进行数据预处理:", ["无", "是"])
        
        if preprocess_option == "是":
            # 用户选择进行预处理后，进一步选择是否进行标准化或归一化
            method_option = st.selectbox("数据预处理方法:", options=["标准化", "归一化"])

            # 自动选择所有数值型列进行处理
            numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns #对所有数值列进行预处理 
            if numeric_columns.empty:
                st.warning("数据集中没有数值型列，无法进行标准化或归一化处理！")
            else:
                if method_option == "标准化":
                    # 标准化处理
                    scaler = StandardScaler()
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                elif method_option == "归一化":
                    # 归一化处理 (0-1)
                    scaler = MinMaxScaler()
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        st.write("(处理好的)数据集预览：")
        st.dataframe(data.head())

        # 特征和标签选择
        st.subheader("选择特征和输出列")
        all_columns = data.columns.tolist()
        output_column = st.selectbox("选择输出列（默认最后一列）", options=all_columns, index=len(all_columns) - 1)
        feature_columns = st.multiselect("选择特征列（默认除输出列外所有列）", options=all_columns, default=[col for col in all_columns if col != output_column])

        # 确保用户选择了特征列
        if not feature_columns:
            st.warning("请至少选择一个特征列进行建模！")

        # 数据集划分比例选择
        st.subheader("数据集划分比例")
        default_train_ratio = 0.7
        train_ratio = st.number_input("训练集比例（如 0.7 表示 70%）", min_value=0.0, max_value=1.0, value=default_train_ratio)
        test_ratio = 1 - train_ratio
        st.write(f"当前设置：训练集比例为 {train_ratio:.2f}，测试集比例为 {test_ratio:.2f}")

        # 随机种子选择
        use_random_seed = st.checkbox("是否使用随机种子", value=False)
        if use_random_seed:
            random_seed = st.number_input("输入随机种子", min_value=1, max_value=10000, value=42)
        else:
            random_seed = 42  # 给定一个默认种子值

        # 数据划分
        X = data[feature_columns]
        y = data[output_column]
        
        # 确保 X 和 y 是有效的
        if len(X) > 0 and len(y) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed)
            st.write("训练集和测试集划分完成。")
        else:
            st.warning("特征列或输出列无效，无法划分数据集。")

    

         
       

        
# 中间栏：数据增强 -> 模型选择 -> 优化 -> 训练
with col2:
    st.header("数据增强、模型选择和优化")
    problem_type = st.selectbox("选择问题类型", ["回归", "分类"])
    
    if problem_type == "回归":
        
        st.subheader("数据增强选项")
        
        # 是否启用数据增强，默认设置为“否”
        use_augmentation = st.radio("是否选择数据增强", ["否", "是"], index=0, key="use_augmentation")
        
        # 当选择“是”时，显示数据增强方法和参数
        if use_augmentation == "是":
                    augmentation_method = st.selectbox("数据增强方法", options=["Bayesian Bootstrap", "Bootstrap","最近邻插值","GAN生成对抗网络", "GMM"], key="augment_method")
                    augment_factor = st.slider("增强倍数", 1, 10, 3, key="augment_factor")
                    flag_augmentation = 1
        else:
                    st.write("数据增强未启用")
                    flag_augmentation = 2
        # 模型选择和训练
        st.subheader("模型选择")
        model_type = st.selectbox("选择回归模型", options=["线性回归", "高斯过程", "SVM模型", "高斯多核回归","RBF神经网络","GENN","QP二次多项式"], key="model_select")
        
        if model_type=="GENN":
                hidden_layer_sizes = st.text_input("隐藏层结构（用英文逗号分隔，例如：10,10）", value="10,10")
                # status_text.text("如10,10表示有两个隐藏层，每层10个神经元")
                hidden_layer_sizes = [int(size.strip()) for size in hidden_layer_sizes.split(",")]
                 # 自定义学习率
                alpha = st.number_input("学习率 (alpha)", min_value=0.0001, max_value=1.0, step=0.0001, value=0.01, format="%.4f")
                # 自定义训练迭代次数
                num_epochs = st.number_input("训练迭代次数 (num_epochs)", min_value=1, max_value=100, value=10)
        # 记录开始时间
        start_time = time.time()

        st.subheader("模型优化")
        optimize_model = st.radio("是否优化模型", ["否", "是"])
        optimize_method = None  # 默认优化方法为空
        if optimize_model == "是":
            optimize_method = st.selectbox("选择优化方法", ["贝叶斯优化", "遗传算法", "粒子群算法"])
            
        # 如果用户选择贝叶斯优化
            if optimize_method == "贝叶斯优化":
                bayes_iterations = st.number_input("贝叶斯优化迭代次数", min_value=1, max_value=100, value=5)
                flag_optimization = 0
        # 如果用户选择遗传算法或粒子群算法
            else:
                population_size = st.number_input("种群大小", min_value=1, max_value=100, value=10)
                max_iterations = st.number_input("最大迭代次数", min_value=1, max_value=100, value=20)
                


        if st.button("训练回归模型"):
        #    try:
                # 显示训练状态
                status_text = st.empty()  # 创建一个空的文本占位符
                status_text.text("模型正在训练中...")  # 训练开始前显示提示
                
                # 增强数据的执行按钮
                if flag_augmentation == 1:
                    if augmentation_method == "Bayesian Bootstrap":
                        augmented_X, augmented_y = bayes_bootstrap(X_train, y_train, augment_factor)
                        flag = 1
                    elif augmentation_method == "Bootstrap":
                        augmented_X, augmented_y =Bootstrap (X_train, y_train, augment_factor)
                        flag = 1
                    elif augmentation_method == "最近邻插值":
                        augmented_X, augmented_y = neighbor_based_interpolation(X_train, y_train, augment_factor,n_neighbors=5)
                        flag = 1
                    elif augmentation_method == "GAN生成对抗网络":
                        augmented_X, augmented_y = generate_regressdata_GAN(X_train, y_train, augment_factor)
                        flag = 2
                    # elif augmentation_method == "SMOTE":
                    #     augmented_X, augmented_y = generate_regressdata_SMOTE(X_train, y_train, augment_factor)
                    #     flag = 1
                    elif augmentation_method == "GMM":
                        augmented_X, augmented_y = generate_regressdata_GMM(X_train, y_train, augment_factor)
                        flag = 2
                    elif augmentation_method == "LSTM":
                        augmented_X, augmented_y = generate_regressdata_LSTM(X_train, y_train, augment_factor)
                        if augmented_X.shape[0] != augmented_y.shape[0]:
                            raise ValueError(f"augmented_X 和 augmented_y 的样本数不匹配：{augmented_X.shape[0]} != {augmented_y.shape[0]}")
                        flag = 2

                    if flag == 1:
                        # 合并增强后的数据到原数据
                        # 将增强后的 y 转换为二维 numpy 数组
                    

                        augmented_y = augmented_y.values.reshape(-1, 1)
                        # augmented_y = augmented_y.reshape(-1, 1)

                        # 将训练集的特征和标签合并
                        data_train = pd.concat([X_train, y_train], axis=1)
                        # 将增强后的特征和目标标签合并
                        augmented_data = pd.concat([augmented_X, pd.DataFrame(augmented_y, columns=[output_column])], axis=1)
                        # 合并原始训练集和增强后的数据
                        data_train = pd.concat([data_train, augmented_data], ignore_index=True)
                        # 从合并后的 data_train 中提取训练特征和标签
                        X_train = data_train[feature_columns].values  # 转换为 numpy 数组
                        y_train = data_train[output_column].values

                        # data = pd.concat([data, augmented_data], ignore_index=True)
                        # st.write("数据增强完成，增强后的数据集预览:")
                        # st.write("数据增强完成!")
                        # st.write(data.head())
                        flag = 0

                    if flag == 2:
                    
                        # 合并增强后的数据到原数据

                        # 将增强后的 y 转换为二维 numpy 数组
                        augmented_y = augmented_y.reshape(-1, 1)  # 直接使用 reshape

                        # 确保 y_train 是 numpy 数组，转换为 numpy ndarray
                        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train  # 如果是 Series，转换为 ndarray

                        # 将训练集的特征和标签合并
                        data_train = np.hstack([X_train, y_train.reshape(-1, 1)])  # 将 y_train 转换为二维数组合并

                        # 将增强后的特征和目标标签合并
                        augmented_data = np.hstack([augmented_X, augmented_y])  # 使用 numpy 的 hstack 合并

                        # 合并原始训练集和增强后的数据
                        data_train = np.vstack([data_train, augmented_data])  # 使用 numpy 的 vstack 合并

                        # 从合并后的 data_train 中提取训练特征和标签
                        X_train = data_train[:, :-1]  # 获取特征列
                        y_train = data_train[:, -1]   # 获取标签列

                        flag = 0
                    flag_augmentation = 0
                    
                    if model_type == "线性回归":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_type == "高斯过程":
                        model = GaussianProcessRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_type == "SVM模型":
                        model = SVR()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_type == "RBF神经网络":
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.to_numpy().astype(float)
                        if isinstance(y_train, (pd.Series, pd.DataFrame)):
                            y_train = y_train.to_numpy().astype(float).reshape(-1, 1)
                        if isinstance(X_test, pd.DataFrame):
                            X_test = X_test.to_numpy().astype(float)
                        if isinstance(y_test, (pd.Series, pd.DataFrame)):
                            y_test = y_test.to_numpy().astype(float)
                        sm = RBF(d0=5)
                        sm.set_training_values(X_train, y_train)
                        sm.train()
                        # 预测
                    
                        y_pred = sm.predict_values(X_test)
                    elif model_type == "QP二次多项式":
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.to_numpy().astype(float)
                        if isinstance(y_train, (pd.Series, pd.DataFrame)):
                            y_train = y_train.to_numpy().astype(float).reshape(-1, 1)
                        if isinstance(X_test, pd.DataFrame):
                            X_test = X_test.to_numpy().astype(float)
                        if isinstance(y_test, (pd.Series, pd.DataFrame)):
                            y_test = y_test.to_numpy().astype(float)
                        # 调用 QP 模型
                        sm = QP()
                        sm.set_training_values(X_train, y_train)  # 设置训练数据
                        sm.train()  # 训练模型
                        # 使用模型预测
                        y_pred = sm.predict_values(X_test)
                    elif model_type == "GENN":
                        # 定义模型参数
                       
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.to_numpy().astype(float)
                        if isinstance(y_train, (pd.Series, pd.DataFrame)):
                            y_train = y_train.to_numpy().astype(float).reshape(-1, 1)
                        if isinstance(X_test, pd.DataFrame):
                            X_test = X_test.to_numpy().astype(float)
                        if isinstance(y_test, (pd.Series, pd.DataFrame)):
                            y_test = y_test.to_numpy().astype(float)
                    
                        options = {
                            "hidden_layer_sizes": hidden_layer_sizes,  # 隐藏层的结构
                            "alpha": alpha,  # 学习率
                            "num_epochs": num_epochs,  # 训练迭代次数
                            "is_normalize": True,  # 是否对训练数据归一化
                        }

                        # 初始化模型
                        sm = GENN()
                        for key, value in options.items():
                            sm.options[key] = value

                        # 设置训练数据
                        sm.set_training_values(X_train, y_train)

                        # 训练模型
                        sm.train()

                        # 预测测试数据
                        y_pred = sm.predict_values(X_test)

                    elif model_type == "高斯多核回归":
                        y_train = np.array(y_train).reshape(-1, 1)
                        y_test = y_test.values.reshape(-1, 1)
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.values
                        X_test = X_test.values
                        y_pred, results = gaussian_multikernel_model(X_train, y_train, X_test)

                elif flag_augmentation == 2:   #flag_augmentation==2为没有经历数据增强
                    if model_type == "线性回归":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_type == "高斯过程":
                        model = GaussianProcessRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_type == "SVM模型":
                        model = SVR()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    elif model_type == "RBF神经网络":
                         # 确保 X_train 和 y_train 是 NumPy 数组
                       # 确保 X_train 和 y_train 是 NumPy 数组，并转换为 float 类型
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.to_numpy().astype(float)
                        if isinstance(y_train, (pd.Series, pd.DataFrame)):
                            y_train = y_train.to_numpy().astype(float).reshape(-1, 1)

                        sm = RBF(d0=5)
                        sm.set_training_values(X_train, y_train)
                        sm.train()
                        # 预测
                         
                        if isinstance(X_test, pd.DataFrame):
                            X_test = X_test.to_numpy().astype(float)
                        if isinstance(y_test, (pd.Series, pd.DataFrame)):
                            y_test = y_test.to_numpy().astype(float)

                        y_pred = sm.predict_values(X_test)
                    elif model_type == "QP二次多项式":
                        # 调用 QP 模型
                        # 确保 X_train 和 y_train 是 NumPy 数组，并转换为 float 类型
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.to_numpy().astype(float)
                        if isinstance(y_train, (pd.Series, pd.DataFrame)):
                            y_train = y_train.to_numpy().astype(float).reshape(-1, 1)
                        if isinstance(X_test, pd.DataFrame):
                            X_test = X_test.to_numpy().astype(float)
                        if isinstance(y_test, (pd.Series, pd.DataFrame)):
                            y_test = y_test.to_numpy().astype(float)

                        sm = QP()
                        sm.set_training_values(X_train, y_train)  # 设置训练数据
                        sm.train()  # 训练模型
                        # 使用模型预测
                        y_pred = sm.predict_values(X_test)

                    elif model_type == "GENN":
                        # 定义模型参数
                         # 自定义隐藏层结构
                            # 确保 X_train 和 y_train 是 NumPy 数组，并转换为 float 类型
                        if isinstance(X_train, pd.DataFrame):
                            X_train = X_train.to_numpy().astype(float)
                        if isinstance(y_train, (pd.Series, pd.DataFrame)):
                            y_train = y_train.to_numpy().astype(float).reshape(-1, 1)
                        if isinstance(X_test, pd.DataFrame):
                            X_test = X_test.to_numpy().astype(float)
                        if isinstance(y_test, (pd.Series, pd.DataFrame)):
                            y_test = y_test.to_numpy().astype(float)
                        
        
                        options = {
                            "hidden_layer_sizes": hidden_layer_sizes,  # 隐藏层的结构
                            "alpha": alpha,  # 学习率
                            "num_epochs": num_epochs,  # 训练迭代次数
                            "is_normalize": True,  # 是否对训练数据归一化
                        }

                        # 初始化模型
                        sm = GENN()
                        for key, value in options.items():
                            sm.options[key] = value

                        # 设置训练数据
                        sm.set_training_values(X_train, y_train)

                        # 训练模型
                        sm.train()

                        # 预测测试数据
                        y_pred = sm.predict_values(X_test)

                    elif model_type == "高斯多核回归":
                        if optimize_model == "是" and optimize_method == "粒子群算法":
                             # 确保训练和测试数据是 numpy 数组
                            X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
                            Y_train = y_train.values.reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)
                            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                            Y_test = y_test.values.reshape(-1, 1) if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test.reshape(-1, 1)

                            # 调用 PSO 优化函数
                            st.write("正在运行粒子群优化高斯多核回归，请稍候...")
                            y_pred, best_r2, best_rmse, best_mape, best_decision_variables, best_mape = optimize_gaussian_multikernel(
                                    X_train, Y_train, X_test, Y_test,
                                    lb=[0.1, 0.1, 0.1], ub=[10, 10, 10],
                                    swarmsize=population_size, maxiter=max_iterations)
                        else:
                            y_train = np.array(y_train).reshape(-1, 1)
                            y_test = y_test.values.reshape(-1, 1)
                            if isinstance(X_train, pd.DataFrame):
                                X_train = X_train.values
                            X_test = X_test.values
                            y_pred, results = gaussian_multikernel_model(X_train, y_train, X_test)

                    flag_augmentation = 0
                
                if optimize_model=="是":
                    st.write("回归模型评估结果:")
                    st.write("R^2 值:", best_r2)
                    st.write("均方根误差 (RMSE):", best_rmse)
                    # st.write("平均绝对百分比误差 (MAPE):", mape)
                    st.write("平均绝对百分比误差 (MAPE):", best_mape)
                    st.write("决策变量最终值：",best_decision_variables)
                    # st.write("预测值示例:", y_pred[:5])
                    # 记录结束时间并计算总时长
                    end_time = time.time()
                    total_time = end_time - start_time
                    st.write(f"总时长: {total_time:.2f} 秒")
                    # 更新状态为“训练完成”
                    status_text.text("模型训练完成")
                else:#没有优化的时候
                # 计算评价指标

                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                    # 展示模型结果
                    st.write("回归模型评估结果:")
                    st.write("R^2 值:", r2)
                    st.write("均方根误差 (RMSE):", rmse)
                    st.write("平均绝对百分比误差 (MAPE):", mape)

                    # st.write("平均绝对百分比误差 (MAPE):", best_mape)
                    # st.write("预测值示例:", y_pred[:5])
                    # 记录结束时间并计算总时长
                    end_time = time.time()
                    total_time = end_time - start_time
                    st.write(f"总时长: {total_time:.2f} 秒")
                    # 更新状态为“训练完成”
                    status_text.text("模型训练完成")
            # except Exception as e:
            #     st.error(f"模型训练过程中发生错误: {e}")

    elif problem_type == "分类":
        model_type = st.sidebar.selectbox("选择分类模型", options=["SVM", "KNN", "决策树"])

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
        
        # 自动进行预测，移除“开始预测”按钮
        prediction = model.predict(predict_data[feature_columns])
        st.write("预测结果：")
        predict_data['预测结果'] = prediction
        st.dataframe(predict_data)
        
        # 提供下载预测结果的功能
        csv = predict_data.to_csv(index=False)
        st.download_button(label="下载预测结果", data=csv, file_name='prediction_results.csv', mime='text/csv')
