"""
温湿度数据相关性分析脚本（备份版本，dropout=0.2 实验配套）。

主要流程：
1. 读取 GBK 编码的温湿度数据 CSV，并把中文列名统一重命名为英文缩写；
2. 输出数据基本信息、统计描述，筛选数值型列并计算 Pearson 相关系数矩阵；
3. 将相关系数矩阵保存为 correlation_matrix.csv；
4. 单独提取目标列 TD 与其他变量的相关性，按绝对值降序排序后保存为 td_correlations.csv；
5. 绘制水平柱状图（正相关红色、负相关蓝色）并保存为 td_correlation_barplot.png。
"""

import pandas as pd  # 导入 pandas 库并约定简写为 pd，用于读取 CSV 和操作 DataFrame
import numpy as np  # 导入 NumPy 并约定简写为 np，这里主要用来判断数值型数据类型
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 并约定简写为 plt，用于绘制柱状图
import seaborn as sns  # 导入 seaborn 并约定简写为 sns（本脚本未直接调用，保留原有导入）
from matplotlib import rcParams  # 从 matplotlib 导入全局配置对象 rcParams，用于统一设置字体

# Set English font（设置英文字体，保证图中英文标签正常显示）
rcParams['font.family'] = 'Arial'  # 将全局字体族设置为 Arial，保证图中英文文字使用无衬线字体
rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号在图中显示为方块

# Read data（读取原始温湿度数据）
df = pd.read_csv('温湿度数据.csv', encoding='gbk')  # 以 GBK 编码读取'温湿度数据.csv'，生成 DataFrame df

# Rename columns to standard notation with subscripts（把中文列名改为带标准下标的英文缩写）
column_mapping = {  # 定义中文列名到标准英文缩写（下标记号）的映射字典
    '日龄': 'Age',  # 把“日龄”列重命名为 Age
    'CO2': 'CO2',  # CO2 列名保持不变，仍记为 CO2
    '当前负压': 'Pin',  # 把“当前负压”列重命名为 Pin
    '目标温度': 'Ttarget',  # 把“目标温度”列重命名为 Ttarget
    '当前温度': 'Tin',  # 把“当前温度”列重命名为 Tin
    '外界温度': 'Tout',  # 把“外界温度”列重命名为 Tout
    '相对湿度': 'RHin',  # 把“相对湿度”列重命名为 RHin
    '风速': 'Vin',  # 把“风速”列重命名为 Vin
    'THI': 'TD'  # 把'THI'列重命名为 TD（本版本把 THI 作为温湿偏差目标列）
}  # 映射字典定义结束
# Apply renaming for columns that exist（仅对数据中实际存在的列执行重命名）
for old_name, new_name in column_mapping.items():  # 遍历映射字典中的每一个（原列名，新列名）键值对
    if old_name in df.columns:  # 仅当原列名确实存在于数据列中时才执行重命名
        df.rename(columns={old_name: new_name}, inplace=True)  # 原地修改 df 的列名，完成单个列的重命名

print("Data Basic Information:")  # 打印提示信息：即将输出数据基本信息
print(df.info())  # 调用 df.info() 输出行数、列类型、非空计数与内存占用
print("\nFirst 5 Rows:")  # 打印提示信息：即将输出前 5 行数据
print(df.head())  # 调用 df.head() 输出前 5 行，快速检查数据内容
print("\nStatistical Description:")  # 打印提示信息：即将输出统计描述
print(df.describe())  # 调用 df.describe() 输出计数、均值、标准差、最小/最大值与分位数

# Select numeric columns for correlation analysis（筛选数值型列用于相关性分析）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 选取所有数值型列，并把列名转换为 Python 列表
print(f"\nNumeric Columns: {numeric_cols}")  # 输出参与相关性分析的数值列清单

# Calculate correlation matrix（计算各数值变量间的 Pearson 相关系数矩阵）
correlation_matrix = df[numeric_cols].corr()  # 对数值型列调用 corr()，计算 Pearson 相关系数矩阵

print("\nCorrelation Matrix:")  # 打印提示信息：即将输出相关系数矩阵
print(correlation_matrix)  # 在控制台打印完整的相关系数矩阵

# Save correlation matrix to CSV（把相关系数矩阵导出为 CSV 文件）
correlation_matrix.to_csv('correlation_matrix.csv', encoding='utf-8')  # 把相关系数矩阵保存为 UTF-8 编码的 'correlation_matrix.csv'
print("\nCorrelation matrix saved to 'correlation_matrix.csv'")  # 打印提示：相关系数矩阵已成功保存

# Extract correlations with TD (temperature deviation)（提取各变量与目标变量 TD 的相关系数）
td_col = 'TD'  # Target variable (use actual column name in data)  # 指定目标列名为 'TD'（数据中的实际列名）
if td_col in correlation_matrix.columns:  # 判断目标列 TD 是否存在于相关系数矩阵的行列索引中
    # Get correlations with TD, excluding TD itself（取出与 TD 的相关系数，并剔除 TD 自身）
    td_correlations = correlation_matrix[td_col].drop(td_col)  # 取相关矩阵中 TD 所在列，并用 drop 去掉 TD 与自身的相关系数 1
    
    # Sort by absolute value of correlation (descending)（按相关系数绝对值从大到小排序）
    td_correlations_sorted = td_correlations.reindex(  # 按指定顺序重新排列相关系数序列（这里使用绝对值降序）
        td_correlations.abs().sort_values(ascending=False).index  # 先取相关系数的绝对值，降序排序后取其索引作为重排依据
    )  # reindex 调用结束，得到排序后的 Series
    
    print(f"\nCorrelations with {td_col}:")  # 打印提示信息：即将输出与 TD 的相关系数
    print(td_correlations_sorted)  # 打印按绝对值降序排列后的相关系数
    
    # Save to CSV（把排序后的相关系数保存为 CSV）
    td_corr_df = pd.DataFrame({  # 构造一个包含“变量名”和“相关系数”两列的 DataFrame
        'Variable': td_correlations_sorted.index,  # 第一列 Variable 取排序后相关系数的索引（变量名）
        'Correlation': td_correlations_sorted.values  # 第二列 Correlation 取排序后相关系数的值
    })  # DataFrame 构造结束
    td_corr_df.to_csv('td_correlations.csv', encoding='utf-8', index=False)  # 把结果保存为 'td_correlations.csv'，index=False 表示不写入行索引
    print(f"\nCorrelations with {td_col} saved to 'td_correlations.csv'")  # 打印提示：与 TD 的相关系数结果已保存
    
    # Create horizontal bar chart with more space for labels（绘制水平柱状图，并预留足够的标签空间）
    fig, ax = plt.subplots(figsize=(12, max(6, len(td_correlations_sorted) * 0.5)))  # 创建图形与坐标轴对象，宽度 12 英寸，高度随变量个数自适应且不小于 6 英寸
    
    # Create color array: positive = red, negative = blue（按相关系数正负生成颜色：正为红、负为蓝）
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in td_correlations_sorted.values]  # 列表推导式：系数大于 0 取红色 #d62728，否则取蓝色 #1f77b4
    
    # Create horizontal bar chart（绘制水平柱状图）
    bars = ax.barh(range(len(td_correlations_sorted)),   # 调用 barh 绘制水平柱状图，柱子的 y 位置为变量序号
                   td_correlations_sorted.values,   # 每个柱子的长度取该变量与 TD 的相关系数
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)  # 设置柱体颜色、透明度 0.8、黑色描边和 0.5 的线宽
    
    # Set y-axis labels with larger font（设置 y 轴刻度标签，并放大字号）
    ax.set_yticks(range(len(td_correlations_sorted)))  # 设置 y 轴刻度位置，使每个变量对应一个刻度
    ax.set_yticklabels(td_correlations_sorted.index, fontsize=13)  # 把 y 轴刻度标签替换为变量名，字号 13
    
    # Set x-axis（设置 x 轴标签、范围、零点参考线与网格）
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=13, fontweight='bold')  # 设置 x 轴标签为 Pearson Correlation Coefficient，加粗显示
    ax.set_xlim(-1, 1)  # 把 x 轴显示范围限定在 -1 到 1 之间（相关系数的理论范围）
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')  # 在 x=0 处绘制一条黑色实线，作为正负相关的分界参考线
    ax.grid(axis='x', alpha=0.3, linestyle='--')  # 沿 x 轴添加半透明虚线网格，便于读取数值
    
    # Add value labels on bars（在每个柱体末端标注具体相关系数数值）
    for i, (bar, value) in enumerate(zip(bars, td_correlations_sorted.values)):  # 遍历柱体对象与其对应的相关系数值（zip 一一配对）
        x_pos = value + (0.04 if value > 0 else -0.04)  # 计算数值标签的横向位置：正值右移 0.04，负值左移 0.04，避免遮挡柱体
        ha = 'left' if value > 0 else 'right'  # 正值标签左对齐、负值标签右对齐，使文字远离柱体
        ax.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')  # 在柱体末端标注保留三位小数的相关系数，垂直居中、水平按 ha 对齐
    
    # Set title（设置图表标题）
    ax.set_title('Correlation of Variables with TD (Temperature Deviation)',   # 设置图表标题，说明展示的是各变量与 TD（温度偏差）的相关性
                 fontsize=15, fontweight='bold', pad=20)  # 标题字号 15、加粗，并与图表保持 20 的间距
    
    # Adjust layout with more left margin for labels（调整布局，增大左边距防止标签被裁切）
    plt.subplots_adjust(left=0.15)  # 调整子图布局，把左边距设为 0.15，给长变量名留出空间
    plt.savefig('td_correlation_barplot.png', dpi=300, bbox_inches='tight')  # 以 300 dpi 保存图片为 'td_correlation_barplot.png'，bbox_inches='tight' 去除多余白边
    print(f"Bar chart saved to 'td_correlation_barplot.png'")  # 打印提示：柱状图已保存
    plt.close()  # 关闭当前图形，释放绘图内存
else:  # 当相关矩阵中不存在 TD 列时进入此分支
    print(f"\nWarning: {td_col} column not found in correlation matrix")  # 输出警告信息，提示未在相关矩阵中找到目标列 TD

print("\nCorrelation analysis completed!")  # 打印分析流程结束的提示信息

