"""
CSV 数据读取快速测试脚本

这是一个轻量级探测脚本：用多种候选编码依次尝试读取温湿度数据 CSV，
验证文件能否被正确解析；读取成功后打印数据的行数、列名以及前 3 行内容，
常用于排查数据文件的编码问题并快速确认数据结构。
"""
import pandas as pd  # 导入 pandas 库，用于读取和处理 CSV 表格数据
import numpy as np  # 导入 numpy 库，供数值计算备用（本脚本未直接使用）

# 读取数据
csv_path = '温湿度数据.csv'  # 指定要读取的温湿度 CSV 文件路径
encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']  # 定义依次尝试的字符编码列表
df = None  # 初始化数据框变量为空，用于保存成功读取的结果

for encoding in encodings:  # 依次尝试每一种候选编码读取文件
    try:  # 捕获读取或解码过程中出现的异常
        df = pd.read_csv(csv_path, encoding=encoding)  # 用当前编码把 CSV 文件读成 DataFrame
        print(f"✓ 读取成功（编码: {encoding}）")  # 打印读取成功提示及本次使用的编码
        break  # 读取成功后跳出循环，不再尝试其它编码
    except:  # 若当前编码读取失败则忽略该异常
        pass  # 不做其它处理，继续尝试下一种编码

if df is not None:  # 若已经成功读取到数据
    print(f"行数: {len(df)}")  # 打印数据框的行数
    print(f"列: {list(df.columns)}")  # 打印数据框的所有列名
    print(df.head(3))  # 打印前 3 行数据，便于快速查看数据结构















































































































































