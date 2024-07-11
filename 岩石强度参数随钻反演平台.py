# -*- coding: UTF-8 -*-
"""
文件分析工具
"""
import streamlit as st
import pandas as pd
import altair as alt  # 导入Altair库
import pickle
import numpy as np

# 设置页面布局为宽模式
st.set_page_config(layout="wide")

# 初始化页面状态
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# 在会话状态中初始化数据变量
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None

def visualize_data(df):
    # 检查数据是否存在，如果存在处理和可视化
    if df is not None:
        # 横坐标的列名
        x_axis_column = df.columns[0]

        # 添加数据集标题
        st.write('### 随钻测量特征参数如下：')

        # 打印整个数据集以便于对照查看，这个操作会使表格尽可能充满容器宽度
        st.table(df.head(10))  # 只展示数据的前几行

        # 添加图表标题
        st.write('### 随钻数据可视化折线图如下：')

        # 将宽格式数据转换为长格式
        long_df = df.melt(id_vars=[x_axis_column])

        # 自定义颜色映射
        color_scale = alt.Scale(domain=long_df['variable'].unique().tolist(),
                                range=['blue', 'green', 'red', 'orange', 'purple', 'yellow'])  # 你可以自定义这个颜色列表

        # 基础图表
        base = alt.Chart(long_df).encode(
            alt.X(x_axis_column, title=x_axis_column),  # 使用第一列作为X轴，并设置标题
            alt.Y('value', title='Values'),  # 'value'是melt方法产生的默认值列
            color=alt.Color('variable', scale=color_scale)  # 使用自定义的颜色映射
        )

        # 绘出光滑的线条，不带点
        line_chart = base.mark_line(interpolate='monotone').encode(
            tooltip=[x_axis_column, 'value', 'variable']  # 鼠标悬停时的提示信息
        )

        # 设置整个图表的属性，使其宽高与Streamlit的表格插件对齐
        combined_chart = line_chart.properties(
            width='container',  # 使用'container'来使图表宽度自适应
            height=400  # 图表高度，你可能需要根据实际情况调整
        ).interactive()

        # 显示图表
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.sidebar.write('请上传一个Excel文件。')

# 主页面内容
def main_page():
    # 设置侧边栏
    st.sidebar.title('岩石力学参数随钻反演平台V1.0')

    # 创建围岩岩性复选框
    options1 = ['碳质板岩', '砂岩', '花岗岩']
    rock_types = st.sidebar.selectbox('围岩岩性', options1)

    # 创建围岩等级复选框
    options2 = ['Ⅰ级', 'Ⅱ级', 'Ⅲ级', 'Ⅳ级', 'Ⅴ级']
    rock_grades = st.sidebar.selectbox('围岩等级', options2)

    # 创建岩石风化程度状态复选框
    options3 = ['微风化', '全风化', '强风化', '中风化', '未风化']
    rock_fh = st.sidebar.selectbox('围岩风化程度', options3)

    # 文件上传器，用于导入Excel文件
    uploaded_file = st.sidebar.file_uploader('原始钻探数据导入', type=['xlsx'])

    # 所有的侧边栏设置完成之后，添加按钮
    if st.sidebar.button('岩石单轴抗压强度预测'):
        st.session_state['page'] = 'ucs_analysis'

    if st.sidebar.button('岩石粘聚力预测'):
        st.session_state['page'] = 'c_analysis'

    if st.sidebar.button('岩石内摩擦角预测'):
        st.session_state['page'] = 'f_analysis'

    if st.sidebar.button('岩石弹性模量预测'):
        st.session_state['page'] = 'e_analysis'

    # 如果文件被上传，显示文件名并进行可视化
    if uploaded_file is not None:
        st.sidebar.write('已上传文件:', uploaded_file.name)

        # 读取Excel文件，假设数据的第一行是header
        df = pd.read_excel(uploaded_file)

        # 清理列名，移除前后的空白符和解决任何可能的问题字符
        df.columns = [col.strip().replace(':', '\:') for col in df.columns]

        # 保存到会话状态
        st.session_state['uploaded_data'] = df
    # 如果没有新的文件上传，但会话状态中有数据，使用会话状态中的旧数据
    elif st.session_state['uploaded_data'] is not None:
        df = st.session_state['uploaded_data']
    else:
        st.sidebar.write('请上传一个Excel文件。')

# 加载归一化参数函数
def load_scaler(scaler_filename):
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# 数据分析函数
def analyze_data(prediction_type):
    st.title(f"岩石{prediction_type}预测")
    if prediction_type == "单轴抗压强度":
        if "uploaded_data" in st.session_state and st.session_state['uploaded_data'] is not None:
            df = st.session_state['uploaded_data']

            # 加载归一化参数
            scaler_x = load_scaler('scaler_x.pkl1')
            scaler_y = load_scaler('scaler_y.pkl1')

            # 确保选取的特征与训练时一致
            # 这里假设你需要的特征是前6列
            feature_columns = df.columns[:6]
            features = df[feature_columns]


            # 检查并确保特征数量与scaler_x期望的一致
            if features.shape[1] != scaler_x.n_features_in_:
                st.error(f"数据列数与模型期望不匹配。期望 {scaler_x.n_features_in_} 列，但得到了 {features.shape[1]} 列。")
                return

            # 使用归一化参数对特征进行归一化处理
            features_scaled = scaler_x.transform(features)

            # 使用pickle加载模型
            with open('rock_ucs_model.pkl1', 'rb') as model_file:
                model = pickle.load(model_file)

            # 对特征进行预测
            predictions_scaled = model.predict(features_scaled)
            # 确保 predictions_scaled 是一个 NumPy 数组
            predictions_scaled = np.array(predictions_scaled)

            # 使用切片，选择第22列的预测值（索引为21）
            predictions_required = predictions_scaled[:, :, 21]

            # 将预测结果进行反归一化处理
            predictions = scaler_y.inverse_transform(predictions_required.reshape(-1, 1))

            # 将反归一化后的预测结果添加到原始数据框中
            df['UCS_预测值（MPa）'] = predictions

            # 打印 UCS 预测值
            st.write("UCS预测值:")
            # 使用 st.columns 创建并列布局
            col1, col2 = st.columns([4, 1])  # 调整列宽比例

            with col1:
                # 可视化预测结果
                chart = alt.Chart(df).mark_line().encode(
                    x='深度（m）',  # 假设你的X轴是某个特征
                    y='UCS_预测值（MPa）',  # Y轴是预测结果
                    tooltip=['深度（m）', 'UCS_预测值（MPa）']
                ).properties(
                    width='container',
                    height=600  # 图表高度
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            with col2:
                # 打印表格
                st.dataframe(df[['深度（m）', 'UCS_预测值（MPa）']], height=550)  # 假设 "深度（mm）" 是你的深度列名

    elif prediction_type == "内摩擦角":
        if "uploaded_data" in st.session_state and st.session_state['uploaded_data'] is not None:
            df = st.session_state['uploaded_data']

            # 加载归一化参数
            scaler_x = load_scaler('scaler_x.pkl2')
            scaler_y = load_scaler('scaler_y.pkl2')

            # 确保选取的特征与训练时一致
            # 这里假设你需要的特征是前6列
            feature_columns = df.columns[:6]
            features = df[feature_columns]


            # 检查并确保特征数量与scaler_x期望的一致
            if features.shape[1] != scaler_x.n_features_in_:
                st.error(f"数据列数与模型期望不匹配。期望 {scaler_x.n_features_in_} 列，但得到了 {features.shape[1]} 列。")
                return

            # 使用归一化参数对特征进行归一化处理
            features_scaled = scaler_x.transform(features)

            # 使用pickle加载模型
            with open('rock_f_model.pkl1', 'rb') as model_file:
                model = pickle.load(model_file)

            # 对特征进行预测
            predictions_scaled = model.predict(features_scaled)

            # 确保 predictions_scaled 是一个 NumPy 数组
            predictions_scaled = np.array(predictions_scaled)

            # 使用切片，选择第22列的预测值（索引为21）
            predictions_required = predictions_scaled[:, :, 21]

            # 将预测结果进行反归一化处理
            predictions = scaler_y.inverse_transform(predictions_required.reshape(-1, 1))

            # 将反归一化后的预测结果添加到原始数据框中
            df['内摩擦角_预测值（°）'] = predictions

            # 打印 内摩擦角 预测值
            st.write("内摩擦角预测值:")
            # 使用 st.columns 创建并列布局
            col1, col2 = st.columns([4, 1])  # 调整列宽比例

            with col1:
                # 可视化预测结果
                chart = alt.Chart(df).mark_line().encode(
                    x='深度（m）',  # 假设你的X轴是某个特征
                    y='内摩擦角_预测值（°）',  # Y轴是预测结果
                    tooltip=['深度（m）', '内摩擦角_预测值（°）']
                ).properties(
                    width='container',
                    height=600  # 图表高度
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            with col2:
                # 打印表格
                st.dataframe(df[['深度（m）', '内摩擦角_预测值（°）']], height=550)  # 假设 "深度（m）" 是你的深度列名

    elif prediction_type == "粘聚力":
        if "uploaded_data" in st.session_state and st.session_state['uploaded_data'] is not None:
            df = st.session_state['uploaded_data']

            # 加载归一化参数
            scaler_x = load_scaler('scaler_x.pkl3')
            scaler_y = load_scaler('scaler_y.pkl3')

            # 确保选取的特征与训练时一致
            # 这里假设你需要的特征是前6列
            feature_columns = df.columns[:6]
            features = df[feature_columns]


            # 检查并确保特征数量与scaler_x期望的一致
            if features.shape[1] != scaler_x.n_features_in_:
                st.error(f"数据列数与模型期望不匹配。期望 {scaler_x.n_features_in_} 列，但得到了 {features.shape[1]} 列。")
                return

            # 使用归一化参数对特征进行归一化处理
            features_scaled = scaler_x.transform(features)

            # 使用pickle加载模型
            with open('rock_c_model.pkl1', 'rb') as model_file:
                model = pickle.load(model_file)

            # 对特征进行预测
            predictions_scaled = model.predict(features_scaled)

            # 确保 predictions_scaled 是一个 NumPy 数组
            predictions_scaled = np.array(predictions_scaled)

            # 使用切片，选择第22列的预测值（索引为21）
            predictions_required = predictions_scaled[:, :, 21]

            # 将预测结果进行反归一化处理
            predictions = scaler_y.inverse_transform(predictions_required.reshape(-1, 1))

            # 将反归一化后的预测结果添加到原始数据框中
            df['粘聚力_预测值（MPa）'] = predictions

            # 打印 内摩擦角 预测值
            st.write("粘聚力预测值:")
            # 使用 st.columns 创建并列布局
            col1, col2 = st.columns([4, 1])  # 调整列宽比例

            with col1:
                # 可视化预测结果
                chart = alt.Chart(df).mark_line().encode(
                    x='深度（m）',  # 假设你的X轴是某个特征
                    y='粘聚力_预测值（MPa）',  # Y轴是预测结果
                    tooltip=['深度（m）', '粘聚力_预测值（MPa）']
                ).properties(
                    width='container',
                    height=600  # 图表高度
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            with col2:
                # 打印表格
                st.dataframe(df[['深度（m）', '粘聚力_预测值（MPa）']], height=550)  # 假设 "深度（m）" 是你的深度列名


    # 利用会话状态中保存的数据进行分析
    if st.session_state['uploaded_data'] is not None:
        df = st.session_state['uploaded_data']
        st.write("分析结果和可视化内容...")
        if st.button('返回主页面'):
            st.session_state['page'] = 'home'
    else:
        st.error("请先上传数据文件。")

# 根据当前页面状态决定渲染哪个页面的内容
if st.session_state['page'] == 'home':
    main_page()
    if st.session_state['uploaded_data'] is not None:
        visualize_data(st.session_state['uploaded_data'])
elif st.session_state['page'] == 'ucs_analysis':
    analyze_data("单轴抗压强度")
elif st.session_state['page'] == 'c_analysis':
    analyze_data("粘聚力")
elif st.session_state['page'] == 'f_analysis':
    analyze_data("内摩擦角")
elif st.session_state['page'] == 'e_analysis':
    analyze_data("弹性模量")
