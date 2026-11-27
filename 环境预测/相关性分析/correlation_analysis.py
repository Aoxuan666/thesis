"""
温湿度变量相关性分析脚本。

读取温湿度数据后选取关键数值变量计算 Pearson 相关系数矩阵，
以 CSV、热力图、表格图片三种形式导出结果，并进一步筛选强相关关系
（|r| > 0.5）、绘制关键变量的散点图矩阵，直观展示变量间的关联程度。
"""
import pandas as pd  # 导入 pandas 数据分析库并起别名 pd，用于数据读取与相关性计算
import numpy as np  # 导入 numpy 数值计算库并起别名 np，本脚本中作为常用数值依赖保留
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口并起别名 plt，用于绘制热力图与散点图
import seaborn as sns  # 导入 seaborn 并起别名 sns，用于绘制美观的热力图
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，避免图表中中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 关闭负号 Unicode 显示，防止负号变方块
# ========== 让DataFrame在控制台完整显示（不省略） ==========
pd.set_option('display.max_rows', 200)  # 控制台最多显示 200 行
pd.set_option('display.max_columns', 200)  # 控制台最多显示 200 列
pd.set_option('display.width', 2000)  # 设置显示宽度 2000，减少换行
pd.set_option('display.max_colwidth', None)  # 不限制单列内容显示长度
pd.set_option('display.expand_frame_repr', False)  # 禁止表格因过宽而换行展示

# 读取数据 - 尝试不同的编码
try:
    df = pd.read_csv('../温湿度数据111.csv', encoding='utf-8-sig')  # 优先用带 BOM 的 UTF-8 编码读取
except UnicodeDecodeError:  # UTF-8 解码失败时
    try:
        df = pd.read_csv('../温湿度数据111.csv', encoding='gbk')  # 退而用 GBK 编码读取
    except UnicodeDecodeError:  # GBK 也失败时
        df = pd.read_csv('../温湿度数据111.csv', encoding='gb18030')  # 最后用 GB18030 编码读取

print("数据基本信息:")  # 提示接下来输出数据基本信息
print(df.info())  # 打印索引、列名、非空数量与 dtype 等概览
print("\n数据前5行:")  # 提示输出前几行
print(df.head())  # 打印数据前 5 行
print("\n数据统计描述:")  # 提示输出统计描述
print(df.describe())  # 打印数值列的计数、均值、标准差、分位数等统计量

# 选择数值列进行相关性分析
numeric_columns = ['Age', 'CO2', 'P', 'Ttarget', 'Tin', 'Tout', 'RHin', 'Vin', 'TD']  # 列出参与相关性分析的数值列
df_numeric = df[numeric_columns]  # 从原数据中提取这些数值列构成子表

# 计算相关系数矩阵
correlation_matrix = df_numeric.corr()  # 计算各数值列之间的 Pearson 相关系数矩阵

print("\n相关系数矩阵(完整显示):")  # 提示输出相关系数矩阵
print(correlation_matrix.to_string(float_format=lambda x: f"{x:.3f}"))  # 以保留三位小数的格式完整打印矩阵

# 保存相关系数矩阵到CSV
correlation_matrix.to_csv('相关系数矩阵.csv', encoding='utf-8-sig')  # 将矩阵保存为带 BOM 的 CSV，便于 Excel 打开
print("\n相关系数矩阵已保存到: 相关系数矩阵.csv")  # 打印保存提示

# ========== 热力图：保证每个网格的数字完整显示 ==========
n = correlation_matrix.shape[0]  # 取相关系数矩阵的变量个数（方阵边长）

# 画布大小随变量数量自适应（每个格子给足空间）
fig_w = max(10, n * 1.2)  # 根据变量数计算画布宽度，至少 10 英寸
fig_h = max(8,  n * 1.1)  # 根据变量数计算画布高度，至少 8 英寸

plt.figure(figsize=(fig_w, fig_h))  # 按自适应尺寸创建画布

ax = sns.heatmap(  # 使用 seaborn 绘制相关系数热力图
    correlation_matrix,  # 输入相关系数矩阵
    annot=True,  # 在每个格子内标注数值
    fmt='.3f',  # 标注数值保留三位小数
    cmap='coolwarm',  # 使用冷暖配色（负相关偏蓝、正相关偏红）
    center=0,  # 颜色映射以 0 为中心，便于区分正负相关
    square=True,  # 强制每个格子为正方形
    linewidths=0.8,  # 设置格子之间的网格线宽度
    linecolor='white',  # 网格线颜色为白色
    cbar_kws={"shrink": 0.8},  # 颜色条缩放为 0.8
    annot_kws={"size": 12}   # 数字字体变大，避免看不清  # 标注数字的字号为 12
)

# 轴标签完整显示：旋转+对齐
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=12)  # 横轴标签旋转 45 度并右对齐
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)  # 纵轴标签保持水平，字号 12

plt.title('变量相关性热力图', fontsize=18, pad=18)  # 设置热力图标题及字号、内边距

# 自动调整边距，避免标签/数字被截断
plt.tight_layout()  # 自动紧凑布局
plt.savefig('相关性热力图.png', dpi=400, bbox_inches='tight')  # 以 400 DPI 保存热力图
print("相关性热力图已保存到: 相关性热力图.png")  # 打印保存提示
plt.show()  # 显示热力图窗口

# ========== （可选但很有用）把相关矩阵导出为"表格图片"，每格文字100%完整 ==========
fig, ax = plt.subplots(figsize=(fig_w, fig_h))  # 新建画布和坐标轴用于绘制表格图片
ax.axis('off')  # 隐藏坐标轴，只展示表格

table_df = correlation_matrix.round(3)  # 将相关系数矩阵四舍五入保留三位小数

tbl = ax.table(  # 在坐标轴上绘制表格
    cellText=table_df.values,  # 表格主体数据（矩阵值）
    rowLabels=table_df.index,  # 行标签为变量名
    colLabels=table_df.columns,  # 列标签为变量名
    cellLoc='center',  # 单元格内容居中对齐
    loc='center'  # 表格整体居中显示
)

# 表格字体和缩放：确保每个格子内容完整
tbl.auto_set_font_size(False)  # 关闭自动字号，改为手动设置
tbl.set_fontsize(12)  # 设置表格字号为 12
tbl.scale(1.2, 1.2)  # 将表格整体放大 1.2 倍

plt.tight_layout()  # 自动紧凑布局
plt.savefig('相关系数矩阵_表格.png', dpi=400, bbox_inches='tight')  # 以 400 DPI 保存表格图片
print("相关系数矩阵表格图已保存到: 相关系数矩阵_表格.png")  # 打印保存提示
plt.show()  # 显示表格图片窗口

# 找出强相关关系（相关系数绝对值 > 0.5）
print("\n强相关关系 (|r| > 0.5):")  # 提示输出强相关变量对
strong_corr = []  # 初始化列表，收集强相关关系记录
for i in range(len(correlation_matrix.columns)):  # 遍历矩阵列索引 i
    for j in range(i + 1, len(correlation_matrix.columns)):  # 遍历 i 之后的列索引 j，避免重复与自相关
        r = correlation_matrix.iloc[i, j]  # 取出第 i、j 两个变量的相关系数
        if abs(r) > 0.5:  # 绝对值大于 0.5 视为强相关
            strong_corr.append({  # 追加一条强相关记录
                '变量1': correlation_matrix.columns[i],  # 第一个变量名
                '变量2': correlation_matrix.columns[j],  # 第二个变量名
                '相关系数': r  # 两者的相关系数
            })

if strong_corr:  # 若找到强相关关系
    strong_corr_df = pd.DataFrame(strong_corr)  # 将记录列表转为 DataFrame
    strong_corr_df = strong_corr_df.sort_values('相关系数', key=lambda s: s.abs(), ascending=False)  # 按相关系数绝对值降序排序
    print(strong_corr_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))  # 不显示索引、保留三位小数打印
else:  # 未找到强相关关系
    print("未发现强相关关系")  # 打印提示

# 创建散点图矩阵（选择部分关键变量）
key_vars = ['CO2', 'Tin', 'Tout', 'RHin', 'TD']  # 挑选关键变量用于散点图矩阵
pd.plotting.scatter_matrix(  # 绘制散点图矩阵
    df_numeric[key_vars],  # 输入关键变量子表
    figsize=(15, 12),  # 画布大小 15x12 英寸
    diagonal='kde',  # 对角线绘制核密度估计曲线
    alpha=0.5  # 散点半透明，便于观察密度
)
plt.suptitle('关键变量散点图矩阵', fontsize=16, y=0.995)  # 设置总标题及位置
plt.tight_layout()  # 自动紧凑布局
plt.savefig('散点图矩阵.png', dpi=300, bbox_inches='tight')  # 以 300 DPI 保存散点图矩阵
print("散点图矩阵已保存到: 散点图矩阵.png")  # 打印保存提示
plt.show()  # 显示散点图矩阵窗口

print("\n分析完成！")  # 打印全部分析结束标志
