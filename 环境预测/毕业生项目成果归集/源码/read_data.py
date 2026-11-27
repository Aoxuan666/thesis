# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8
"""
本脚本用于探测并读取“机器人测试.csv”环境数据文件。

由于 CSV 可能采用 GBK、UTF-8 等不同编码，脚本按候选编码顺序逐一尝试读取；
一旦成功即打印列名、数据规模、各列数据类型以及前 5 行内容，并终止后续尝试。
若所有编码均失败，每次尝试后都会输出失败原因，便于定位编码或路径问题。
"""
import pandas as pd  # 导入 pandas 数据分析库并简写为 pd
import os  # 导入操作系统接口模块，用于判断文件路径是否存在

path = r'd:\resources\pythonTest\Total - 副本 - 副本\机器人测试.csv'  # 定义待读取 CSV 文件的原始字符串路径（r 前缀避免反斜杠被当作转义）
print('文件存在:', os.path.exists(path))  # 打印目标文件是否存在的检查结果

encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'utf-8-sig']  # 准备按优先级依次尝试的候选字符编码列表
for enc in encodings:  # 遍历每种候选编码逐一尝试读取
    try:  # 捕获读取过程中可能抛出的异常
        df = pd.read_csv(path, encoding=enc)  # 用当前编码尝试把 CSV 文件读成 DataFrame
        print(f'编码: {enc}')  # 打印本次成功使用的编码名称
        print('列名:', list(df.columns))  # 打印 DataFrame 的全部列名
        print('数据形状:', df.shape)  # 打印数据的行列规模（行数, 列数）
        print('数据类型:')  # 提示即将输出各列的数据类型
        print(df.dtypes)  # 打印每一列对应的数据类型
        print('前5行:')  # 提示即将输出前 5 行数据
        print(df.head().to_string())  # 以字符串形式打印前 5 行的完整内容
        break  # 读取成功后跳出循环，不再尝试其他编码
    except Exception as e:  # 捕获读取失败产生的异常并绑定到变量 e
        print(f'{enc} 失败: {e}')  # 打印当前编码失败的具体原因
