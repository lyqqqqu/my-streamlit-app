import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
def load_data(uploaded_file):
    """
    加载上传的 CSV 或 Excel 文件并返回 DataFrame。
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
           data = pd.read_excel(uploaded_file)
        
        st.session_state.data = data  # 存储到 session_state
        st.session_state.cleaned = False  # 重置清洗状态
        st.session_state.converted = False  # 重置转换状态
        st.success("文件上传并加载成功。")
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None
    return data

def convert_data(data, method):
    """
    对数值型列进行标准化或归一化处理，并显示进度指示器。
    """
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    if numeric_columns.empty:
        st.warning("数据集中没有数值型列，无法进行标准化或归一化处理！")
        return data
    if method == "标准化":
        scaler = StandardScaler()
        st.write("正在进行标准化处理...")
    elif method == "归一化":
        scaler = MinMaxScaler()
        st.write("正在进行归一化处理...")

    
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    st.success(f"{method}处理完成。")
    st.session_state.data = data  # 更新 session_state
    st.session_state.converted = True
    return data

def clean_data(data):
    """
    进行数据清洗，包括缺失值检测与填充、异常值检测与处理，并显示进度指示器。
    """
    
    
    # 缺失值检测与处理
    st.write("1、缺失值检测与处理")
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()
    st.write(f"总缺失值数量: {total_missing}")
    
    if total_missing > 0:
        st.write("缺失值按列分布:")
        st.write(missing_values[missing_values > 0])
        
        # 选择填充方式
        fill_option = st.selectbox("选择缺失值填充方式:", ["均值", "中位数", "众数"])
        
        if st.button("执行缺失值填充"):
            numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
            with st.spinner("正在填充缺失值..."):
                for col in numeric_columns:
                    if data[col].isnull().sum() > 0:
                        if fill_option == "均值":
                            fill_value = data[col].mean()
                        elif fill_option == "中位数":
                            fill_value = data[col].median()
                        else:
                            fill_value = data[col].mode()[0]
                        data[col].fillna(fill_value, inplace=True)
            st.success(f"缺失值已使用{fill_option}填充。")
            # st.write("填充后的数据预览：")
            #st.dataframe(data.head())
            st.session_state.data = data
            st.session_state.cleaned = True
    else:
        st.success("数据集中没有缺失值。")
        st.session_state.cleaned = True
    
    # 异常值检测与处理
    st.write("2、异常值检测与处理")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    if numeric_columns.empty:
        st.warning("数据集中没有数值型列，无法进行异常值检测！")
    else:
        z_scores = np.abs(stats.zscore(data[numeric_columns]))
        # 处理可能存在的 NaN 值
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        outliers = (z_scores > 3).sum()
        total_outliers = outliers.sum()
        st.write(f"总异常值数量（Z-Score > 3）: {total_outliers}")
        
        if total_outliers > 0:
            st.write("异常值按列分布:")
            st.write(outliers[outliers > 0])
            
            # 选择异常值处理方式
            outlier_option = st.selectbox("选择异常值处理方式:", ["替换为均值", "替换为中位数", "删除异常值"])
            
            if st.button("执行异常值处理"):
                with st.spinner("正在处理异常值..."):
                    for col in numeric_columns:
                        z_col = np.abs(stats.zscore(data[col].dropna()))
                        # 重新计算 Z-Score，排除 NaN 值
                        z_col = z_col.replace([np.inf, -np.inf], np.nan).dropna()
                        # 获取原始数据的索引
                        indices = data.index[z_col > 3].tolist()
                        if indices:
                            if outlier_option == "替换为均值":
                                replacement = data[col].mean()
                                data.loc[indices, col] = replacement
                            elif outlier_option == "替换为中位数":
                                replacement = data[col].median()
                                data.loc[indices, col] = replacement
                            else:
                                data.drop(indices, inplace=True)
                st.success(f"异常值已使用{outlier_option}处理。")
                # st.write("处理后的数据预览：")
                # st.dataframe(data.head())
                st.session_state.data = data 
                st.session_state.cleaned = True
        else:
            st.success("数据集中没有异常值。")
            st.session_state.cleaned = True
    
    return data