"""温湿度数据相关性分析脚本。

本脚本读取「温湿度数据111.csv」，将中文列名统一重命名为英文列名，
选取关键数值型变量计算 Pearson 相关系数矩阵，并将其导出为 CSV 与多种可视化图片：
热力图、表格图片以及关键变量的散点图矩阵；最后还会筛选出 |r|>0.5 的强相关关系并打印。
"""

import pandas as pd  # 导入 pandas，用于数据读取、列名重命名、相关性计算与结果导出
import numpy as np  # 导入 numpy，用于数值计算（本脚本中主要供相关性分析底层使用）
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图接口，用于绘制热力图、表格图与散点图
import seaborn as sns  # 导入 seaborn，用于绘制带数值标注的相关系数热力图

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体，保证图表中文标题正常显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号，避免出现方块乱码

# ========== 让DataFrame在控制台完整显示（不省略） ==========
pd.set_option('display.max_rows', 200)  # 设置控制台最多显示 200 行，避免数据被折叠
pd.set_option('display.max_columns', 200)  # 设置控制台最多显示 200 列
pd.set_option('display.width', 2000)  # 设置显示宽度为 2000 字符，避免换行
pd.set_option('display.max_colwidth', None)  # 不限制单列显示宽度，完整展示列内容
pd.set_option('display.expand_frame_repr', False)  # 禁止自动扩展 DataFrame 表示宽度

# 读取数据 - 尝试不同的编码
try:  # 首先尝试使用带 BOM 的 UTF-8 编码读取 CSV
    df = pd.read_csv('温湿度数据111.csv', encoding='utf-8-sig')  # 读取温湿度原始数据文件
except UnicodeDecodeError:  # 若 UTF-8 解码失败，则尝试 GBK 编码
    try:  # 使用 GBK 编码再次尝试读取
        df = pd.read_csv('温湿度数据111.csv', encoding='gbk')  # 以 GBK 编码读取 CSV
    except UnicodeDecodeError:  # 若 GBK 仍失败，则使用 GB18030 编码
        df = pd.read_csv('温湿度数据111.csv', encoding='gb18030')  # 以 GB18030 编码读取 CSV
print("数据基本信息:")  # 打印数据基本信息标题
print(df.info())  # 打印 DataFrame 的索引、列类型与缺失情况
print("\n数据前5行:")  # 打印数据前 5 行标题
print(df.head())  # 打印数据集的前 5 行
print("\n数据统计描述:")  # 打印统计描述标题
print(df.describe())  # 打印各数值列的计数、均值、标准差、极值等统计量
# 打印实际的列名
print("\nCSV文件中的实际列名:")  # 打印 CSV 中实际存在的列名标题
print(list(df.columns))  # 以列表形式打印所有列名
# 重命名列：将中文列名改为英文列名，特别是将"当前负压"改为Pin
column_mapping = {  # 定义中文列名到英文列名的映射字典
    '日龄': 'Age',  # 将“日龄”重命名为 Age
    'CO2': 'CO2',  # CO2 列名保持不变
    '当前负压': 'Pin',  # P改为Pin：将“当前负压”重命名为 Pin
    '目标温度': 'Ttarget',  # 将“目标温度”重命名为 Ttarget
    '当前温度': 'Tin',  # 将“当前温度”重命名为 Tin
    '外界温度': 'Tout',  # 将“外界温度”重命名为 Tout
    '相对湿度': 'RHin',  # 将“相对湿度”重命名为 RHin
    '室内风速': 'Vin',  # 将“室内风速”重命名为 Vin
    '温度偏差': 'TD',  # 将“温度偏差”重命名为 TD
    '室外湿度':'RHout',  # 将“室外湿度”重命名为 RHout
    '室外风速':'Vout'  # 将“室外风速”重命名为 Vout
}
df = df.rename(columns=column_mapping)  # 按映射字典重命名 DataFrame 的所有列

# 选择数值列进行相关性分析
numeric_columns = ['Age', 'CO2', 'Pin', 'Ttarget', 'Tin', 'Tout', 'RHin', 'Vin', 'TD','RHout', 'Vout']  # 列出参与相关性分析的 11 个数值型变量
df_numeric = df[numeric_columns]  # 从原数据中截取这些数值列，得到分析用子集

# 计算相关系数矩阵
correlation_matrix = df_numeric.corr()  # 计算各数值变量之间的 Pearson 相关系数矩阵

print("\n相关系数矩阵(完整显示):")  # 打印相关系数矩阵标题
print(correlation_matrix.to_string(float_format=lambda x: f"{x:.3f}"))  # 以保留 3 位小数的格式完整打印矩阵

# 保存相关系数矩阵到CSV
correlation_matrix.to_csv('相关系数矩阵.csv', encoding='utf-8-sig')  # 将相关系数矩阵保存为 CSV 文件
print("\n相关系数矩阵已保存到: 相关系数矩阵.csv")  # 提示矩阵 CSV 已保存

# ========== 热力图：保证每个网格的数字完整显示 ==========
n = correlation_matrix.shape[0]  # 取相关系数矩阵的变量个数（矩阵行数）

# 画布大小随变量数量自适应（每个格子给足空间）
fig_w = max(10, n * 1.2)  # 根据变量数计算画布宽度，最少 10 英寸
fig_h = max(8,  n * 1.1)  # 根据变量数计算画布高度，最少 8 英寸

plt.figure(figsize=(fig_w, fig_h))  # 创建指定大小的图形窗口

ax = sns.heatmap(  # 使用 seaborn 绘制相关系数热力图
    correlation_matrix,  # 传入相关系数矩阵作为绘图数据
    annot=True,  # 在每个格子上显示数值标注
    fmt='.3f',  # 数值标注保留 3 位小数
    cmap='coolwarm',  # 使用 coolwarm 配色（负值为冷色、正值为暖色）
    center=0,  # 将色标中心设为 0，便于区分正负相关
    square=True,  # 每个格子绘制为正方形
    linewidths=0.8,  # 设置格子之间的网格线宽度
    linecolor='white',  # 网格线颜色设为白色
    cbar_kws={"shrink": 0.8},  # 色标条缩放为 0.8 倍
    annot_kws={"size": 12}   # annot_kws 设置数字字体变大，避免看不清
)

# 轴标签完整显示：旋转+对齐
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=12)  # 将横轴标签旋转 45 度并右对齐，防止重叠
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)  # 纵轴标签保持水平，字号设为 12

plt.title('变量相关性热力图', fontsize=18, pad=18)  # 设置热力图标题及字号、与图表的间距

# 自动调整边距，避免标签/数字被截断
plt.tight_layout()  # 自动调整子图边距，防止标签被裁切
plt.savefig('相关性热力图.png', dpi=400, bbox_inches='tight')  # 以 400 DPI 保存热力图为 PNG
print("相关性热力图已保存到: 相关性热力图.png")  # 提示热力图已保存
plt.show()  # 在窗口中显示热力图

# ========== （可选但很有用）把相关矩阵导出为“表格图片”，每格文字100%完整 ==========
fig, ax = plt.subplots(figsize=(fig_w, fig_h))  # 新建图形与坐标轴，用于绘制表格图片
ax.axis('off')  # 关闭坐标轴显示

table_df = correlation_matrix.round(3)  # 将相关系数矩阵四舍五入保留 3 位小数，作为表格数据

tbl = ax.table(  # 在坐标轴上绘制表格
    cellText=table_df.values,  # 表格单元格内容为矩阵的数值
    rowLabels=table_df.index,  # 行标签为矩阵的行名（变量名）
    colLabels=table_df.columns,  # 列标签为矩阵的列名（变量名）
    cellLoc='center',  # 单元格内容居中对齐
    loc='center'  # 表格整体位于坐标轴中心
)

# 表格字体和缩放：确保每个格子内容完整
tbl.auto_set_font_size(False)  # 关闭自动字体大小，改为手动设置
tbl.set_fontsize(12)  # 将表格字体大小设为 12
tbl.scale(1.2, 1.2)  # 将表格在横纵方向各放大 1.2 倍

plt.tight_layout()  # 自动调整边距，避免表格被裁切
plt.savefig('相关系数矩阵_表格.png', dpi=400, bbox_inches='tight')  # 以 400 DPI 保存表格图片
print("相关系数矩阵表格图已保存到: 相关系数矩阵_表格.png")  # 提示表格图片已保存
plt.show()  # 显示表格图片

# 找出强相关关系（相关系数绝对值 > 0.5）
print("\n强相关关系 (|r| > 0.5):")  # 打印强相关关系标题
strong_corr = []  # 初始化空列表，用于存放筛选出的强相关关系
for i in range(len(correlation_matrix.columns)):  # 遍历相关系数矩阵的每一列（变量 i）
    for j in range(i + 1, len(correlation_matrix.columns)):  # 遍历 i 之后的列，避免重复与自相关
        r = correlation_matrix.iloc[i, j]  # 取出变量 i 与变量 j 的相关系数
        if abs(r) > 0.5:  # 若相关系数的绝对值大于 0.5，判定为强相关
            strong_corr.append({  # 将该强相关关系以字典形式加入列表
                '变量1': correlation_matrix.columns[i],  # 记录第一个变量名
                '变量2': correlation_matrix.columns[j],  # 记录第二个变量名
                '相关系数': r  # 记录两者的相关系数
            })

if strong_corr:  # 如果存在强相关关系
    strong_corr_df = pd.DataFrame(strong_corr)  # 将强相关列表转换为 DataFrame
    strong_corr_df = strong_corr_df.sort_values('相关系数', key=lambda s: s.abs(), ascending=False)  # 按相关系数绝对值从大到小排序
    print(strong_corr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))  # 不显示索引、保留 3 位小数打印结果
else:  # 若没有强相关关系
    print("未发现强相关关系")  # 打印提示信息

# 创建散点图矩阵（选择部分关键变量）
key_vars = ['CO2', 'Tin', 'Tout', 'RHin', 'TD']  # 指定绘制散点图矩阵的关键变量
pd.plotting.scatter_matrix(  # 调用 pandas 提供的散点图矩阵绘图方法
    df_numeric[key_vars],  # 传入关键变量的数据子集
    figsize=(15, 12),  # 设置画布大小为 15x12 英寸
    diagonal='kde',  # 对角线绘制核密度估计曲线
    alpha=0.5  # 设置散点透明度为 0.5，便于观察重叠点
)
plt.suptitle('关键变量散点图矩阵', fontsize=16, y=0.995)  # 设置整张图的标题及位置
plt.tight_layout()  # 自动调整子图间距
plt.savefig('散点图矩阵.png', dpi=300, bbox_inches='tight')  # 以 300 DPI 保存散点图矩阵
print("散点图矩阵已保存到: 散点图矩阵.png")  # 提示散点图矩阵已保存
plt.show()  # 显示散点图矩阵

print("\n分析完成！")  # 打印分析结束提示
