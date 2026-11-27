"""
温度偏差相关性分析脚本（相关系数矩阵与水平条形图）。

读取温湿度数据 CSV，把中文列名映射为带标准符号的英文缩写（如温度偏差→TD），
计算各数值型变量两两之间的 Pearson 相关系数矩阵并保存为 CSV；随后提取各变量与目标 TD 的相关系数，
按绝对值降序排列后另存，并绘制红色（正相关）/蓝色（负相关）水平条形图，
直观展示各因素与温度偏差的相关方向和强弱。
"""
import pandas as pd  # 导入 pandas 数据处理库，简写为 pd，用于读取 CSV 与计算相关性
import numpy as np  # 导入 numpy 数值计算库，简写为 np，用于判断数值型列
import matplotlib.pyplot as plt  # 导入 pyplot 绘图模块，用于绘制水平条形图
import seaborn as sns  # 导入 seaborn 统计绘图库（本脚本未直接调用其函数，保留备用）
from matplotlib import rcParams  # 导入 matplotlib 全局配置对象，用于统一设置字体

# Set English font（设置英文字体，保证英文标签正常显示）
rcParams['font.family'] = 'Arial'  # 把全局字体设为 Arial
rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号显示异常

# Read data（读取温湿度数据）
df = pd.read_csv('温湿度数据.csv', encoding='gbk')  # 以 GBK 编码读取温湿度数据 CSV

# Rename columns to standard notation with subscripts（把中文列名映射为带下标的英文标准符号）
column_mapping = {  # 定义中文列名到英文缩写的映射字典
    '日龄': 'Age',  # 日龄 → Age
    'CO2': 'CO2',  # CO2 列名保持不变
    '当前负压': 'Pin',  # 当前负压 → Pin
    '目标温度': 'Ttarget',  # 目标温度 → Ttarget
    '当前温度': 'Tin',  # 当前温度 → Tin
    '外界温度': 'Tout',  # 外界温度 → Tout
    '相对湿度': 'RHin',  # 相对湿度 → RHin
    '风速': 'Vin',  # 风速 → Vin
    '温度偏差': 'TD',  # 温度偏差 → TD（目标变量）
    '室外风速':'Vout',  # 室外风速 → Vout
    '室外湿度':'RHout'  # 室外湿度 → RHout
}
# Apply renaming for columns that exist（只对数据中实际存在的列执行重命名）
for old_name, new_name in column_mapping.items():  # 遍历映射字典中的每一对列名
    if old_name in df.columns:  # 仅当原列存在于数据中时
        df.rename(columns={old_name: new_name}, inplace=True)  # 就地重命名该列

print("Data Basic Information:")  # 打印数据基本信息标题
print(df.info())  # 打印 DataFrame 的索引、非空计数与 dtype 信息
print("\nFirst 5 Rows:")  # 打印前 5 行的标题
print(df.head())  # 打印前 5 行数据
print("\nStatistical Description:")  # 打印统计描述标题
print(df.describe())  # 打印各数值列的计数、均值、标准差与分位数等统计量

# Select numeric columns for correlation analysis（筛选数值型列用于相关性分析）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 选取 dtype 为数值的列名列表
print(f"\nNumeric Columns: {numeric_cols}")  # 打印参与分析的数值列

# Calculate correlation matrix（计算 Pearson 相关系数矩阵）
correlation_matrix = df[numeric_cols].corr()  # 对数值列计算两两 Pearson 相关系数

print("\nCorrelation Matrix:")  # 打印相关系数矩阵标题
print(correlation_matrix)  # 打印相关系数矩阵

# Save correlation matrix to CSV（把相关系数矩阵保存为 CSV 文件）
correlation_matrix.to_csv('correlation_matrix.csv', encoding='utf-8')  # 以 UTF-8 编码写出矩阵
print("\nCorrelation matrix saved to 'correlation_matrix.csv'")  # 打印保存提示

# Extract correlations with TD (temperature deviation)（提取各变量与温度偏差 TD 的相关系数）
td_col = 'TD'  # Target variable (use actual column name in data)  # 目标变量名：使用数据中实际的列名
if td_col in correlation_matrix.columns:  # 若相关矩阵中存在 TD 列
    # Get correlations with TD, excluding TD itself（取出与 TD 的相关，排除 TD 自身）
    td_correlations = correlation_matrix[td_col].drop(td_col)  # 取 TD 列并去掉与自身的相关（恒为 1）

    # Sort by absolute value of correlation (descending)（按相关系数绝对值降序排列）
    td_correlations_sorted = td_correlations.reindex(  # 按指定顺序重排 Series
        td_correlations.abs().sort_values(ascending=False).index  # 用绝对值降序后的索引作为新顺序
    )

    print(f"\nCorrelations with {td_col}:")  # 打印与 TD 的相关系数标题
    print(td_correlations_sorted)  # 打印排序后的相关系数

    # Save to CSV（保存为 CSV 文件）
    td_corr_df = pd.DataFrame({  # 构造含变量名和相关系数两列的 DataFrame
        'Variable': td_correlations_sorted.index,  # 变量名列取排序后的索引
        'Correlation': td_correlations_sorted.values  # 相关系数列取排序后的数值
    })
    td_corr_df.to_csv('td_correlations.csv', encoding='utf-8', index=False)  # 写出 CSV 且不包含行索引
    print(f"\nCorrelations with {td_col} saved to 'td_correlations.csv'")  # 打印保存提示

    # Create horizontal bar chart with more space for labels（创建水平条形图，为标签留出更多左侧空间）
    fig, ax = plt.subplots(figsize=(12, max(6, len(td_correlations_sorted) * 0.5)))  # 创建画布，高度随变量数自适应

    # Create color array: positive = red, negative = blue（构造颜色数组：正相关红色、负相关蓝色）
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in td_correlations_sorted.values]  # 按相关系数正负分配红/蓝

    # Create horizontal bar chart（绘制水平条形图）
    bars = ax.barh(range(len(td_correlations_sorted)),   # 以 0..n-1 作为纵轴条形位置
                   td_correlations_sorted.values,   # 条形长度为相关系数
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)  # 设置颜色、透明度与黑色描边

    # Set y-axis labels with larger font（设置纵轴标签，使用较大字号）
    ax.set_yticks(range(len(td_correlations_sorted)))  # 设定纵轴刻度位置
    ax.set_yticklabels(td_correlations_sorted.index, fontsize=13)  # 用变量名标注纵轴刻度

    # Set x-axis（设置横轴）
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=13, fontweight='bold')  # 横轴标签：Pearson 相关系数
    ax.set_xlim(-1, 1)  # 横轴范围限定在 -1 到 1
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')  # 绘制零相关参考竖线
    ax.grid(axis='x', alpha=0.3, linestyle='--')  # 显示横向虚线网格

    # Add value labels on bars（在条形末端添加数值标注）
    for i, (bar, value) in enumerate(zip(bars, td_correlations_sorted.values)):  # 遍历每个条形及其数值
        x_pos = value + (0.04 if value > 0 else -0.04)  # 依据正负把标注位置稍微外移
        ha = 'left' if value > 0 else 'right'  # 正值标注左对齐、负值右对齐
        ax.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')  # 绘制保留三位的数值文本

    # Set title（设置标题）
    ax.set_title('Correlation of Variables with TD (Temperature Deviation)',   # 标题：各变量与温度偏差的相关性
                 fontsize=15, fontweight='bold', pad=20)  # 设置字号、加粗与内边距

    # Adjust layout with more left margin for labels（调整布局，为左侧变量标签留出更多边距）
    plt.subplots_adjust(left=0.15)  # 把左边距设为 0.15
    plt.savefig('td_correlation_barplot.png', dpi=600, bbox_inches='tight')  # 保存条形图为图片
    print(f"Bar chart saved to 'td_correlation_barplot.png'")  # 打印保存提示
    plt.close()  # 关闭当前图形，释放内存
else:
    print(f"\nWarning: {td_col} column not found in correlation matrix")  # 未找到 TD 列时打印警告

print("\nCorrelation analysis completed!")  # 打印相关性分析全部完成标志
