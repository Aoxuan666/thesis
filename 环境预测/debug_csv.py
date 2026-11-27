# -*- coding: utf-8 -*-
"""CSV 文件编码与列结构调试脚本。

本脚本将工作目录切换到自身所在路径，遍历当前目录下除预警日志与结果文件之外的所有 CSV 文件，
逐一尝试 gbk、utf-8、utf-8-sig 三种编码读取前 3 行；一旦读取成功，就打印文件采用的编码、
列数以及每一列的列名和首行取值，便于排查中文 CSV 的编码与列结构问题。
"""

import os  # 导入 os 模块，用于路径处理、切换目录与列举文件
import pandas as pd  # 导入 pandas，用于尝试读取 CSV 并查看列信息

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 将当前工作目录切换到脚本所在目录，保证相对路径可用

for fname in sorted(os.listdir('.')):  # 排序遍历当前目录下的所有文件与子目录名
    if fname.endswith('.csv') and fname not in ['warning_simulation_log.csv', 'warning_metrics_result.csv']:  # 仅处理 CSV 文件，且排除预警日志与指标结果这两个已知文件
        for enc in ['gbk', 'utf-8', 'utf-8-sig']:  # 依次尝试三种常见编码
            try:  # 尝试用当前编码读取文件
                df = pd.read_csv(fname, encoding=enc, nrows=3)  # 只读取前 3 行以快速探测结构
                print(f'文件: {fname}  编码: {enc}  列数: {len(df.columns)}')  # 打印文件名、成功编码与列数
                for i, c in enumerate(df.columns):  # 遍历每一列，输出列索引与列名
                    print(f'  [{i}] {c}: {df.iloc[0, i]}')  # 打印列索引、列名及第一行对应的取值
                print()  # 打印空行分隔不同文件
                break  # 读取成功后跳出编码尝试循环，处理下一个文件
            except Exception as e:  # 若当前编码读取失败（捕获所有异常）
                continue  # 忽略该异常，继续尝试下一种编码
