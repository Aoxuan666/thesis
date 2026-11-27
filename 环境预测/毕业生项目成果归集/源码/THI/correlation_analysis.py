"""
温湿度数据相关性分析脚本（THI 目录版本）。

主要流程：
1. 读取 GBK 编码的温湿度数据 CSV，把中文列名统一映射为英文缩写；
2. 若数据中没有 TD（温度偏差）列，则按 TD = Tin - Ttarget 计算生成；
3. 输出数据基本信息与统计描述，筛选数值列并计算 Pearson 相关系数矩阵，保存为 correlation_matrix.csv（UTF-8 BOM，便于 Excel 打开）；
4. 复用 plot_target_correlation 函数，分别输出 THI 与 TD 两个目标变量的相关系数排序结果和水平柱状图。
"""

import pandas as pd  # 导入 pandas 并简写为 pd，用于读取 CSV 与操作 DataFrame
import numpy as np  # 导入 NumPy 并简写为 np，这里主要用于 select_dtypes 筛选数值列
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 并简写为 plt，用于绘制水平柱状图
from matplotlib import rcParams  # 从 matplotlib 导入全局配置对象 rcParams，用于设置字体

# Set English font（设置英文字体，保证图中英文标签正常显示）
rcParams['font.family'] = 'Arial'  # 把全局字体族设置为 Arial，保证英文标签使用无衬线字体
rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号在图中显示为方块

# Read data（读取原始温湿度数据）
df = pd.read_csv('温湿度数据.csv', encoding='gbk')  # 以 GBK 编码读取'温湿度数据.csv'，生成 DataFrame df

# Rename columns to standard notation（把中文列名统一改为标准英文缩写）
column_mapping = {  # 定义中文列名到标准英文缩写的映射字典
    '日龄': 'Age',  # 把“日龄”重命名为 Age
    'CO2': 'CO2',  # CO2 列名保持不变
    '当前负压': 'Pin',  # 把“当前负压”重命名为 Pin
    '目标温度': 'Ttarget',  # 把“目标温度”重命名为 Ttarget
    '当前温度': 'Tin',  # 把“当前温度”重命名为 Tin
    '外界温度': 'Tout',  # 把“外界温度”重命名为 Tout
    '相对湿度': 'RHin',  # 把“相对湿度”重命名为 RHin
    '风速': 'Vin',  # 把“风速”重命名为 Vin
    'THI': 'THI',  # THI 列名保持不变（本版本同时分析 THI 与 TD）
    '温度偏差': 'TD',  # 把“温度偏差”重命名为 TD
    '室外风速':'Vout',  # 把“室外风速”重命名为 Vout
    '室外湿度':'RHout'  # 把“室外湿度”重命名为 RHout
}  # 映射字典定义结束

# Apply renaming for columns that exist（仅对数据中实际存在的列执行重命名）
for old_name, new_name in column_mapping.items():  # 遍历映射字典中的每个（原列名，新列名）键值对
    if old_name in df.columns:  # 仅当原列名确实存在时才执行重命名
        df.rename(columns={old_name: new_name}, inplace=True)  # 原地修改列名，完成单个列的重命名

# If TD does not exist, calculate it（若数据中尚无 TD 列，则根据温度与设定温度计算）
if 'TD' not in df.columns and 'Tin' in df.columns and 'Ttarget' in df.columns:  # 若 TD 列不存在，且当前温度与目标温度列都存在，则进入计算分支
    df['TD'] = df['Tin'] - df['Ttarget']  # 按“当前温度 - 目标温度”计算温度偏差 TD 并新增该列
    print("TD column created: TD = Tin - Ttarget")  # 打印提示：TD 列已根据公式生成

print("Data Basic Information:")  # 打印提示：即将输出数据基本信息
print(df.info())  # 调用 df.info() 输出列类型、非空计数与内存占用
print("\nFirst 5 Rows:")  # 打印提示：即将输出前 5 行数据
print(df.head())  # 调用 df.head() 输出前 5 行，快速检查数据内容
print("\nStatistical Description:")  # 打印提示：即将输出统计描述
print(df.describe())  # 调用 df.describe() 输出各数值列的计数、均值、标准差与分位数

# Select numeric columns for correlation analysis（筛选数值型列用于相关性分析）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 筛选所有数值型列并把列名转为列表，作为相关性分析对象
print(f"\nNumeric Columns: {numeric_cols}")  # 输出参与相关性分析的数值列清单

# Calculate correlation matrix（计算各数值变量之间的 Pearson 相关系数矩阵）
correlation_matrix = df[numeric_cols].corr()  # 对数值列调用 corr()，计算 Pearson 相关系数矩阵

print("\nCorrelation Matrix:")  # 打印提示：即将输出相关系数矩阵
print(correlation_matrix)  # 在控制台打印完整的相关系数矩阵

# Save correlation matrix to CSV（把相关系数矩阵导出为 CSV）
correlation_matrix.to_csv('correlation_matrix.csv', encoding='utf-8-sig')  # 把矩阵保存为 'correlation_matrix.csv'，utf-8-sig 带 BOM 便于 Excel 直接打开
print("\nCorrelation matrix saved to 'correlation_matrix.csv'")  # 打印提示：相关系数矩阵已保存


def plot_target_correlation(correlation_matrix, target_col, output_csv, output_png):
    """提取并可视化某个目标变量与其他变量的相关性。
    参数:
        correlation_matrix: 数值列的 Pearson 相关系数矩阵（DataFrame）
        target_col: 目标变量名，如 THI 或 TD
        output_csv: 相关系数排序结果的 CSV 输出路径
        output_png: 水平柱状图的 PNG 输出路径
    返回: 无显式返回值；结果写入 CSV 与 PNG 文件。
    """
    if target_col in correlation_matrix.columns:  # 判断目标列是否存在于相关系数矩阵中，存在才继续绘图
        # Get correlations with target, excluding itself（取与目标的相关系数，并排除目标自身）
        target_correlations = correlation_matrix[target_col].drop(target_col)  # 取目标列的相关系数，并用 drop 去掉目标与自身的相关系数

        # Sort by absolute value of correlation（按相关系数绝对值从大到小排序）
        target_correlations_sorted = target_correlations.reindex(  # 按指定顺序重排相关系数（这里用绝对值降序，调用开始）
            target_correlations.abs().sort_values(ascending=False).index  # 先取绝对值并降序排序，用其索引作为重排依据
        )  # reindex 调用结束，得到排序后的相关系数序列

        print(f"\nCorrelations with {target_col}:")  # 打印提示：即将输出与目标的相关系数
        print(target_correlations_sorted)  # 打印按绝对值降序排列后的相关系数

        # Save to CSV（把排序后的相关系数结果保存为 CSV）
        target_corr_df = pd.DataFrame({  # 构造包含“变量名”和“相关系数”两列的 DataFrame（调用开始）
            'Variable': target_correlations_sorted.index,  # Variable 列取排序后相关系数的索引（变量名）
            'Correlation': target_correlations_sorted.values  # Correlation 列取排序后相关系数的值
        })  # DataFrame 构造结束
        target_corr_df.to_csv(output_csv, encoding='utf-8-sig', index=False)  # 把结果保存为 CSV，index=False 表示不写入行索引
        print(f"Correlations with {target_col} saved to '{output_csv}'")  # 打印提示：相关系数结果已保存

        # Create horizontal bar chart（创建水平柱状图画布）
        fig, ax = plt.subplots(figsize=(12, max(6, len(target_correlations_sorted) * 0.5)))  # 创建图形与坐标轴，宽 12 英寸、高随变量数自适应且不小于 6 英寸

        # Create color array: positive = red, negative = blue（按相关系数正负生成颜色：正红负蓝）
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in target_correlations_sorted.values]  # 列表推导式：相关系数大于 0 取红色，否则取蓝色

        # Draw bars（绘制水平柱体）
        bars = ax.barh(  # 调用 barh 绘制水平柱状图（调用开始）
            range(len(target_correlations_sorted)),  # 柱体的 y 位置取变量序号
            target_correlations_sorted.values,  # 柱长取该变量与目标的相关系数
            color=colors,  # 设置柱体颜色
            alpha=0.8,  # 设置柱体透明度为 0.8
            edgecolor='black',  # 设置柱体描边颜色为黑色
            linewidth=0.5  # 设置描边线宽为 0.5
        )  # barh 调用结束

        # Y-axis labels（设置 y 轴刻度位置与变量名标签）
        ax.set_yticks(range(len(target_correlations_sorted)))  # 设置 y 轴刻度位置，使每个变量对应一个刻度
        ax.set_yticklabels(target_correlations_sorted.index, fontsize=13)  # 把 y 轴刻度标签设为变量名，字号 13

        # X-axis settings（设置 x 轴标签、范围、零点参考线与网格）
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=13)  # 设置 x 轴标签为 Pearson Correlation Coefficient，字号 13
        ax.set_xlim(-1, 1)  # 把 x 轴范围限定在相关系数理论范围 -1 到 1
        ax.axvline(x=0, color='black', linewidth=1.5)  # 在 x=0 处画黑色实线，作为正负相关的分界参考线
        ax.grid(axis='x', alpha=0.3, linestyle='--')  # 添加 x 轴半透明虚线网格，便于读数

        # Add value labels（在每个柱体末端标注相关系数数值）
        for i, value in enumerate(target_correlations_sorted.values):  # 遍历每个柱体对应的（序号，相关系数）
            x_pos = value + (0.04 if value > 0 else -0.04)  # 计算数值标签的横向位置：正值右移 0.04，负值左移 0.04
            ha = 'left' if value > 0 else 'right'  # 正值标签左对齐、负值标签右对齐，避免遮挡柱体
            ax.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=10)  # 在柱端标注三位小数的相关系数，垂直居中并按 ha 对齐

        # Title（设置图表标题）
        ax.set_title(f'Correlation of Variables with {target_col}', fontsize=12, pad=20)  # 设置标题，说明展示各变量与目标的相关性

        # Layout（调整布局、保存图片并关闭图形）
        plt.subplots_adjust(left=0.2)  # 调整子图布局，把左边距设为 0.2，给变量名标签留出空间
        plt.savefig(output_png, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存柱状图，bbox_inches=tight 去除多余白边
        print(f"Bar chart saved to '{output_png}'")  # 打印提示：柱状图已保存到指定路径
        plt.close()  # 关闭图形释放内存
    else:  # 目标列不存在时的分支
        print(f"\nWarning: {target_col} column not found in correlation matrix")  # 输出警告：相关系数矩阵中未找到目标列


# Output THI correlation results（输出并绘制与 THI 的相关性结果）
plot_target_correlation(  # 调用函数绘制并保存与 THI 的相关性结果（调用开始）
    correlation_matrix=correlation_matrix,  # 传入前面计算得到的相关系数矩阵
    target_col='THI',  # 指定目标列为 THI
    output_csv='thi_correlations.csv',  # 指定相关系数 CSV 输出文件为 thi_correlations.csv
    output_png='thi_correlation_barplot.png'  # 指定柱状图输出文件为 thi_correlation_barplot.png
)  # 函数调用结束

# Output TD correlation results（输出并绘制与 TD 的相关性结果）
plot_target_correlation(  # 调用函数绘制并保存与 TD 的相关性结果（调用开始）
    correlation_matrix=correlation_matrix,  # 传入同一个相关系数矩阵
    target_col='TD',  # 指定目标列为 TD（温度偏差）
    output_csv='td_correlations.csv',  # 指定相关系数 CSV 输出文件为 td_correlations.csv
    output_png='td_correlation_barplot.png'  # 指定柱状图输出文件为 td_correlation_barplot.png
)  # 函数调用结束

print("\nCorrelation analysis completed!")  # 打印相关性分析全部完成的提示