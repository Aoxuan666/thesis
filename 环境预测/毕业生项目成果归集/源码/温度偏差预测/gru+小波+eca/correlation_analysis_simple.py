"""
温湿度数据相关性分析脚本（简化的中文版）。

读取温湿度数据 CSV，计算全部数值型变量的 Pearson 相关系数矩阵并以 GBK 编码保存为 CSV，
绘制带数值标注的相关性热力图；随后遍历矩阵上三角，找出 |r| > 0.7 的强相关变量对，
按相关系数绝对值降序排序后另存为 CSV。使用 Agg 后端与中文字体，适合无界面批量出图。
"""
import pandas as pd  # 导入 pandas 数据处理库，简写为 pd，用于读取 CSV 与相关性计算
import numpy as np  # 导入 numpy 数值计算库，简写为 np，用于判断数值型列
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端（原注释保留），只保存图片不弹窗
import matplotlib.pyplot as plt  # 导入 pyplot 绘图模块，用于生成热力图
import seaborn as sns  # 导入 seaborn 统计绘图库，用于绘制热力图
from matplotlib import rcParams  # 导入 matplotlib 全局配置对象，用于设置中文字体
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体候选：黑体、微软雅黑
rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号显示为方块

print("正在读取数据...")  # 提示开始读取数据
# 读取数据
df = pd.read_csv('温湿度数据.csv', encoding='gbk')  # 以 GBK 编码读取温湿度数据 CSV

print("数据基本信息：")  # 打印数据基本信息标题
print(f"数据形状: {df.shape}")  # 打印数据的行数与列数
print(f"\n列名: {list(df.columns)}")  # 打印所有列的名称列表

# 选择数值型列进行相关性分析
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # 筛选 dtype 为数值的列
print(f"\n数值型列: {numeric_cols}")  # 打印数值型列列表

# 计算相关系数矩阵
print("\n正在计算相关系数矩阵...")  # 提示开始计算相关系数矩阵
correlation_matrix = df[numeric_cols].corr()  # 计算数值列之间的 Pearson 相关系数矩阵

print("\n相关系数矩阵：")  # 打印相关系数矩阵标题
print(correlation_matrix)  # 打印相关系数矩阵

# 保存相关系数矩阵到CSV
correlation_matrix.to_csv('相关系数矩阵.csv', encoding='gbk')  # 以 GBK 编码保存矩阵为 CSV
print("\n相关系数矩阵已保存到 '相关系数矩阵.csv'")  # 打印保存提示

# 创建热力图
print("\n正在生成热力图...")  # 提示正在生成热力图
plt.figure(figsize=(14, 12))  # 创建 14x12 英寸画布
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})  # 绘制带数值标注的蓝红发散热力图，单元格为正方形
plt.title('温湿度数据相关性热力图', fontsize=16, pad=20)  # 设置热力图标题
plt.tight_layout()  # 自动调整布局防止裁切
plt.savefig('相关性热力图.png', dpi=600, bbox_inches='tight')  # 保存热力图为图片
plt.close()  # 关闭图形，释放内存
print("相关性热力图已保存到 '相关性热力图.png'")  # 打印保存提示

# 找出强相关关系（相关系数绝对值 > 0.7，排除自相关）
print("\n强相关关系（|r| > 0.7）：")  # 打印强相关关系标题
strong_corr = []  # 初始化强相关变量对列表
for i in range(len(correlation_matrix.columns)):  # 遍历矩阵列索引 i
    for j in range(i+1, len(correlation_matrix.columns)):  # 遍历 j>i，只看上三角以避免重复和自相关
        corr_value = correlation_matrix.iloc[i, j]  # 取第 i 行第 j 列的相关系数
        if abs(corr_value) > 0.7:  # 若相关系数绝对值超过 0.7
            strong_corr.append({  # 记录这对强相关变量
                '变量1': correlation_matrix.columns[i],  # 第一个变量名
                '变量2': correlation_matrix.columns[j],  # 第二个变量名
                '相关系数': corr_value  # 对应的相关系数
            })

if strong_corr:  # 若找到了强相关关系
    strong_corr_df = pd.DataFrame(strong_corr)  # 把列表转换为 DataFrame
    strong_corr_df = strong_corr_df.sort_values('相关系数', key=abs, ascending=False)  # 按相关系数绝对值降序排序
    print(strong_corr_df.to_string(index=False))  # 打印强相关关系表（不显示行索引）
    strong_corr_df.to_csv('强相关关系.csv', encoding='gbk', index=False)  # 以 GBK 编码保存为 CSV
    print("\n强相关关系已保存到 '强相关关系.csv'")  # 打印保存提示
else:  # 若没有强相关关系
    print("未发现强相关关系")  # 打印提示信息

print("\n相关性分析完成！")  # 打印相关性分析全部完成标志
