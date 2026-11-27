#!/usr/bin/env python
# -*- coding: utf-8 -*-  # 声明源文件采用 UTF-8 编码（首行 shebang 不加注释，以免破坏直接执行）
"""
温度偏差预警评价 - 简化版本
直接在IDE中运行此文件

本模块是预警评价逻辑的「精简、自包含」实现，不依赖外部数据清理模块，
自带从阈值查询、预警标签构造、二分类指标计算到结果导出与可视化的完整流程。
典型运行顺序：读取 CSV（多编码尝试）→ 列名映射 → 构造未来时刻的真实值/阈值/标签
→ 计算精确率、召回率、F1 与误报率 → 输出明细 CSV 并绘制典型时段预警对比图。
适合直接在 IDE 中运行，以快速查看温度偏差预警的评价效果。
"""

import pandas as pd  # 导入 pandas 并简写为 pd，用于表格数据的读取、变换与导出
import numpy as np  # 导入 numpy 并简写为 np，用于数值计算与随机噪声生成
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 并简写为 plt，用于绘制预警图
from typing import Tuple, Dict, Optional  # 导入 typing 中的类型注解，用于标注函数参数与返回值类型


def get_thresholds(age: int) -> Tuple[float, float]:  # 定义阈值查询函数：按日龄返回（偏冷阈值, 偏热阈值）
    """
    根据日龄返回对应的偏冷阈值和偏热阈值

    Args:
        age: 日龄（天数）
    Returns:
        (cold_th, hot_th): 偏冷阈值与偏热阈值组成的二元组
    """
    if 1 <= age <= 3:  # 日龄 1~3 天：偏冷 -0.5℃、偏热 +1.0℃
        return -0.5, 1.0  # 返回该日龄段的偏冷/偏热阈值
    elif 4 <= age <= 7:  # 日龄 4~7 天：偏冷 -0.5℃、偏热 +1.5℃
        return -0.5, 1.5  # 返回该日龄段的偏冷/偏热阈值
    elif 8 <= age <= 11:  # 日龄 8~11 天：偏冷 -1.0℃、偏热 +2.0℃
        return -1.0, 2.0  # 返回该日龄段的偏冷/偏热阈值
    elif 12 <= age <= 14:  # 日龄 12~14 天：偏冷 -1.5℃、偏热 +2.5℃
        return -1.5, 2.5  # 返回该日龄段的偏冷/偏热阈值
    elif 15 <= age <= 21:  # 日龄 15~21 天：偏冷 -2.0℃、偏热 +3.0℃
        return -2.0, 3.0  # 返回该日龄段的偏冷/偏热阈值
    elif 22 <= age <= 28:  # 日龄 22~28 天：偏冷 -2.5℃、偏热 +3.5℃
        return -2.5, 3.5  # 返回该日龄段的偏冷/偏热阈值
    else:  # 其余日龄（29 天及以上）：采用最宽松阈值
        return -3.0, 4.0  # 返回兜底阈值：偏冷 -3.0℃、偏热 +4.0℃


def prepare_warning_dataframe(df: pd.DataFrame, horizon: int = 6, use_future_threshold: bool = True) -> pd.DataFrame:  # 定义数据预处理函数：生成预警标签、阈值与特征，返回处理后的 DataFrame
    """
    预处理数据框，生成预警标签和相关特征

    Args:
        df: 输入 DataFrame，至少包含 age、td_true、td_pred 等列
        horizon: 预测步长（默认 6，对应未来 30 分钟）
        use_future_threshold: 为 True 时用未来阈值生成预测标签（更严格，推荐）；为 False 时用当前阈值
    Returns:
        处理后的 DataFrame，追加了阈值、未来值、预警标签、severity、warning_level 等列
    """
    df = df.copy()  # 复制输入 DataFrame，避免修改调用方原始数据
    
    # 1. 获取当前时刻的阈值
    df[['cold_th', 'hot_th']] = df['age'].apply(  # 对每行日龄调用阈值函数，拆成两列写入当前偏冷/偏热阈值
        lambda x: pd.Series(get_thresholds(int(x)))  # 日龄转为整数后查阈值，以 Series 返回便于展开为两列
    )  # 当前阈值列赋值结束
    
    # 2. 获取未来真实值和未来日龄
    df['future_td_true'] = df['td_true'].shift(-horizon)  # 向上平移 horizon 行，得到 t+H 时刻的真实温度偏差
    df['future_age'] = df['age'].shift(-horizon)  # 同样向上平移，得到 t+H 时刻对应的日龄
    
    # 3. 获取未来阈值
    df[['future_cold_th', 'future_hot_th']] = df['future_age'].apply(  # 按未来日龄查询未来阈值并拆成两列
        lambda x: pd.Series(get_thresholds(int(x))) if pd.notna(x) else pd.Series([np.nan, np.nan])  # 未来日龄非空则查阈值，为空则填 NaN
    )  # 未来阈值列赋值结束
    
    # 4. 生成预测预警标签
    if use_future_threshold:  # 若启用未来阈值（推荐），预测标签以未来阈值为判据
        df['pred_hot'] = (df['td_pred'] > df['future_hot_th']).astype(int)  # 预测偏差超未来偏热阈值记为偏热预警 1，否则 0
        df['pred_cold'] = (df['td_pred'] < df['future_cold_th']).astype(int)  # 预测偏差低于未来偏冷阈值记为偏冷预警 1，否则 0
    else:  # 否则使用当前时刻阈值作为判据（原始版本）
        df['pred_hot'] = (df['td_pred'] > df['hot_th']).astype(int)  # 预测偏差超当前偏热阈值记为 1
        df['pred_cold'] = (df['td_pred'] < df['cold_th']).astype(int)  # 预测偏差低于当前偏冷阈值记为 1
    
    # 5. 生成真实预警标签
    df['true_hot_future'] = (df['future_td_true'] > df['future_hot_th']).astype(int)  # 未来真实偏差超未来偏热阈值记为真实偏热 1
    df['true_cold_future'] = (df['future_td_true'] < df['future_cold_th']).astype(int)  # 未来真实偏差低于未来偏冷阈值记为真实偏冷 1
    
    # 6. 计算综合超限幅度
    df['severity'] = 0.0  # 初始化超限幅度列为 0，表示默认未超限
    hot_mask = df['td_pred'] > df['future_hot_th']  # 布尔掩码：预测值超过未来偏热阈值的行
    cold_mask = df['td_pred'] < df['future_cold_th']  # 布尔掩码：预测值低于未来偏冷阈值的行
    
    df.loc[hot_mask, 'severity'] = df.loc[hot_mask, 'td_pred'] - df.loc[hot_mask, 'future_hot_th']  # 偏热行：超限幅度 = 预测值 - 偏热阈值
    df.loc[cold_mask, 'severity'] = df.loc[cold_mask, 'future_cold_th'] - df.loc[cold_mask, 'td_pred']  # 偏冷行：超限幅度 = 偏冷阈值 - 预测值
    
    # 7. 生成告警等级
    def get_warning_level(severity):  # 定义内部函数：把连续的超限幅度映射为离散告警等级
        """将超限幅度 severity 映射为 none/general/important/severe 四档告警等级。"""  # 内部函数文档字符串：说明等级划分规则
        if severity == 0:  # 超限幅度为 0，表示未触发预警
            return 'none'  # 返回「无」
        elif 0 < severity < 1.5:  # 超限幅度在 (0, 1.5) 之间
            return 'general'  # 返回「一般」
        elif 1.5 <= severity < 2.0:  # 超限幅度在 [1.5, 2.0) 之间
            return 'important'  # 返回「重要」
        else:  # 超限幅度 ≥ 2.0
            return 'severe'  # 返回「严重」
    
    df['warning_level'] = df['severity'].apply(get_warning_level)  # 对每行超限幅度应用等级映射，生成告警等级列
    
    # 8. 删除最后 horizon 个样本
    df = df.iloc[:-horizon].reset_index(drop=True)  # 截掉末尾 horizon 行（无未来真实值），并重置索引
    
    return df  # 返回预处理完成的 DataFrame


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:  # 定义二分类指标计算函数，返回精确率/召回率/F1/误报率字典
    """
    计算二分类评价指标

    Args:
        y_true: 真实标签数组（0/1）
        y_pred: 预测标签数组（0/1）
    Returns:
        包含 precision（精确率）、recall（召回率）、f1（F1 值）、far（误报率）的字典
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))  # 构建有效掩码：剔除真实值或预测值为 NaN 的样本
    y_true = y_true[mask]  # 仅保留有效样本的真实标签
    y_pred = y_pred[mask]  # 仅保留有效样本的预测标签
    
    tp = np.sum((y_pred == 1) & (y_true == 1))  # 真正例 TP：预测为 1 且真实为 1 的数量
    fp = np.sum((y_pred == 1) & (y_true == 0))  # 假正例 FP：预测为 1 但真实为 0 的数量
    fn = np.sum((y_pred == 0) & (y_true == 1))  # 假反例 FN：预测为 0 但真实为 1 的数量
    tn = np.sum((y_pred == 0) & (y_true == 0))  # 真反例 TN：预测为 0 且真实为 0 的数量
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 精确率 = TP/(TP+FP)，无预测正例时取 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 召回率 = TP/(TP+FN)，无真实正例时取 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # F1 = 精确率与召回率的调和平均
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0  # 误报率 FAR = FP/(TP+FP)，无预测正例时取 0
    
    return {  # 将四项指标打包成字典返回
        'precision': precision,  # 精确率：预测为正的样本中有多少真的为正
        'recall': recall,  # 召回率：真实为正的样本中有多少被预测为正
        'f1': f1,  # F1 值：精确率与召回率的调和平均
        'far': far  # 误报率：预测为正的样本中有多少属于误报
    }  # 字典构造结束


def evaluate_warning(df_prepared: pd.DataFrame) -> pd.DataFrame:  # 定义评价汇总函数：分别计算偏热/偏冷指标并整理成表
    """
    计算预警评价指标

    Args:
        df_prepared: 预处理后的 DataFrame，需含真实/预测的偏热、偏冷标签列
    Returns:
        指标汇总 DataFrame，每行对应一种预警类型（hot/cold）
    """
    hot_metrics = compute_binary_metrics(  # 计算偏热预警的二分类指标
        df_prepared['true_hot_future'].values,  # 传入未来真实偏热标签数组
        df_prepared['pred_hot'].values  # 传入预测偏热标签数组
    )  # 偏热指标计算结束
    
    cold_metrics = compute_binary_metrics(  # 计算偏冷预警的二分类指标
        df_prepared['true_cold_future'].values,  # 传入未来真实偏冷标签数组
        df_prepared['pred_cold'].values  # 传入预测偏冷标签数组
    )  # 偏冷指标计算结束
    
    results = pd.DataFrame({  # 将偏热/偏冷指标汇总成结果表
        'warning_type': ['hot', 'cold'],  # 预警类型列：偏热、偏冷
        'precision': [hot_metrics['precision'], cold_metrics['precision']],  # 精确率：偏热、偏冷各一行
        'recall': [hot_metrics['recall'], cold_metrics['recall']],  # 召回率：偏热、偏冷各一行
        'f1': [hot_metrics['f1'], cold_metrics['f1']],  # F1 值：偏热、偏冷各一行
        'far': [hot_metrics['far'], cold_metrics['far']]  # 误报率：偏热、偏冷各一行
    })  # 结果表构造结束
    
    return results  # 返回指标汇总表


def plot_warning_segment(  # 定义绘图函数：绘制测试集典型时段的预警对比图
    df_prepared: pd.DataFrame,  # 预处理后的 DataFrame
    start: Optional[int] = None,  # 起始索引，None 表示从头开始
    end: Optional[int] = None,  # 结束索引，None 表示到末尾
    n_points: int = 200,  # 未指定 start/end 时绘制的点数
    figsize: Tuple[int, int] = (14, 6)  # 画布尺寸（宽, 高），单位英寸
) -> None:  # 无返回值，直接出图
    """
    绘制测试集典型时段的预警图（时间对齐版本）

    注意：td_pred 是对未来 t+H 的预测，因此图中真实值和阈值都用 future 版本以保证时间对齐。

    Args:
        df_prepared: 预处理后的 DataFrame
        start: 起始索引（None 表示从头开始）
        end: 结束索引（None 表示到末尾）
        n_points: 未指定 start/end 时绘制前 n_points 个点
        figsize: 图形尺寸
    Returns:
        None
    """
    if start is None and end is None:  # 若起止索引都未指定，则绘制前 n_points 个点
        end = min(n_points, len(df_prepared))  # 结束索引取 n_points 与数据总长中较小者
        start = 0  # 起始索引置 0
    elif start is None:  # 若仅未指定起始索引
        start = 0  # 起始索引置 0，使用传入的结束索引
    elif end is None:  # 若仅未指定结束索引
        end = len(df_prepared)  # 结束索引取数据总长
    
    df_plot = df_prepared.iloc[start:end].reset_index(drop=True)  # 截取 [start, end) 区间作为绘图数据并重置索引
    
    fig, ax = plt.subplots(figsize=figsize)  # 创建画布与坐标轴对象
    
    x = np.arange(len(df_plot))  # 生成横坐标 0..n-1，代表时间步长
    
    # 绘制预测TD和未来真实TD
    ax.plot(x, df_plot['td_pred'], label='预测 TD (t+H)', marker='s', markersize=4, linewidth=1.5)  # 绘制预测偏差曲线（方点标记）
    ax.plot(x, df_plot['future_td_true'], label='未来真实 TD (t+H)', marker='o', markersize=4, linewidth=1.5)  # 绘制未来真实偏差曲线（圆点标记）
    
    # 绘制未来阈值曲线
    ax.plot(x, df_plot['future_hot_th'], color='red', linestyle='--',   # 绘制未来偏热阈值（红色虚线），参数换行
            label='未来偏热阈值', linewidth=1.5, alpha=0.7)  # 续行：设置图例、线宽与透明度
    ax.plot(x, df_plot['future_cold_th'], color='blue', linestyle='--',   # 绘制未来偏冷阈值（蓝色虚线），参数换行
            label='未来偏冷阈值', linewidth=1.5, alpha=0.7)  # 续行：设置图例、线宽与透明度
    
    # 标注预测预警
    hot_indices = df_plot[df_plot['pred_hot'] == 1].index  # 取出预测偏热样本的行索引
    cold_indices = df_plot[df_plot['pred_cold'] == 1].index  # 取出预测偏冷样本的行索引
    
    if len(hot_indices) > 0:  # 若存在预测偏热样本
        ax.scatter(hot_indices, df_plot.loc[hot_indices, 'td_pred'],   # 在预测值位置绘制偏热散点，参数换行
                  color='red', s=100, marker='^', label='预测偏热', zorder=5, edgecolors='darkred', linewidths=1.5)  # 续行：红色三角、置顶显示
    
    if len(cold_indices) > 0:  # 若存在预测偏冷样本
        ax.scatter(cold_indices, df_plot.loc[cold_indices, 'td_pred'],   # 在预测值位置绘制偏冷散点，参数换行
                  color='blue', s=100, marker='v', label='预测偏冷', zorder=5, edgecolors='darkblue', linewidths=1.5)  # 续行：蓝色倒三角、置顶显示
    
    # 标注真实预警
    true_hot_indices = df_plot[df_plot['true_hot_future'] == 1].index  # 取出真实偏热样本的行索引
    true_cold_indices = df_plot[df_plot['true_cold_future'] == 1].index  # 取出真实偏冷样本的行索引
    
    if len(true_hot_indices) > 0:  # 若存在真实偏热样本
        ax.scatter(true_hot_indices, df_plot.loc[true_hot_indices, 'future_td_true'],   # 在未来真实值位置绘制真实偏热散点，参数换行
                  color='darkred', s=120, marker='*', label='真实偏热 (未来)', zorder=4)  # 续行：深红色五角星
    
    if len(true_cold_indices) > 0:  # 若存在真实偏冷样本
        ax.scatter(true_cold_indices, df_plot.loc[true_cold_indices, 'future_td_true'],   # 在未来真实值位置绘制真实偏冷散点，参数换行
                  color='darkblue', s=120, marker='*', label='真实偏冷 (未来)', zorder=4)  # 续行：深蓝色五角星
    
    ax.set_xlabel('时间步长', fontsize=11)  # 设置横轴标签
    ax.set_ylabel('温度偏差 (°C)', fontsize=11)  # 设置纵轴标签
    ax.set_title('温度偏差预警评价 - 典型时段（时间对齐）', fontsize=13, fontweight='bold')  # 设置图表标题（加粗）
    ax.legend(loc='best', fontsize=9)  # 自动选择最佳位置显示图例
    ax.grid(True, alpha=0.3)  # 开启网格线并设置透明度
    
    plt.tight_layout()  # 自动调整子图间距，避免标签重叠
    plt.show()  # 显示图形窗口


# ============ 主程序 ============
if __name__ == '__main__':  # 当本文件作为主程序直接运行时，执行以下完整评价流程
    print("=" * 60)  # 打印分隔线
    print("温度偏差预警评价系统")  # 打印系统标题
    print("=" * 60)  # 再次打印分隔线
    
    # 1. 读取数据
    print("\n[1] 读取数据...")  # 打印步骤 1：读取数据
    csv_path = '温湿度数据.csv'  # 指定待读取的 CSV 文件路径
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']  # 列出候选编码，按常见程度排列
    df = None  # 初始化数据框为空，用于记录是否读取成功
    for encoding in encodings:  # 依次尝试每种编码
        try:  # 对单次读取单独捕获解码异常
            df = pd.read_csv(csv_path, encoding=encoding)  # 用当前编码尝试读取 CSV
            print(f"✓ 成功读取数据（编码: {encoding}），共 {len(df)} 行")  # 打印成功使用的编码与数据行数
            break  # 读取成功即跳出循环
        except (UnicodeDecodeError, UnicodeError):  # 若编码不匹配导致解码错误
            continue  # 尝试下一种编码
    
    if df is None:  # 若所有编码均读取失败
        print("✗ 无法读取文件")  # 打印读取失败提示
        exit(1)  # 以错误码 1 退出程序
    
    print(f"✓ 原始列名: {list(df.columns)}")  # 打印原始数据列名，便于核对
    
    # 2. 列名映射
    print("\n[2] 处理列名...")  # 打印步骤 2：处理列名
    col_mapping = {}  # 初始化列名映射字典（统一名 → 原始名）
    for col in df.columns:  # 遍历原始列名
        col_str = str(col).lower()  # 转小写便于匹配英文列名
        if 'time' in col_str or '时间' in col:  # 若列名含 time（小写）或「时间」
            col_mapping['timestamp'] = col  # 把原始时间列名映射到统一名 timestamp
        elif 'age' in col_str or '日龄' in col:  # 若列名含 age 或「日龄」
            col_mapping['age'] = col  # 映射到统一名 age
        elif 'td' in col_str or '温度偏差' in col:  # 若列名含 td 或「温度偏差」
            col_mapping['td_true'] = col  # 映射到统一名 td_true
    
    df = df.rename(columns=col_mapping)  # 按映射关系重命名列，统一列名
    
    # 检查必需列
    required_cols = ['timestamp', 'age', 'td_true']  # 定义后续步骤必需的列名
    missing_cols = [col for col in required_cols if col not in df.columns]  # 计算缺失的必需列
    if missing_cols:  # 若存在缺失列
        print(f"✗ 缺少必需列: {missing_cols}")  # 打印缺失列清单
        print(f"✓ 可用列: {list(df.columns)}")  # 打印当前可用列供排查
        exit(1)  # 以错误码 1 退出程序
    
    # 处理 td_pred
    if 'td_pred' not in df.columns:  # 若数据中没有预测列
        print("⚠ 未找到预测值列 (td_pred)，使用 td_true + 小噪声作为演示")  # 打印警告说明将用噪声模拟预测值
        df['td_pred'] = df['td_true'] + np.random.normal(0, 0.1, len(df))  # 真实偏差叠加均值 0、标准差 0.1 的噪声
    
    print(f"✓ 处理后列名: {list(df.columns)}")  # 打印重命名后的列名
    
    # 3. 数据预处理
    print("\n[3] 预处理数据...")  # 打印步骤 3：预处理数据
    df_prepared = prepare_warning_dataframe(df, horizon=6)  # 构造未来标签与预警特征（预测步长 6）
    print(f"✓ 预处理完成，有效样本数: {len(df_prepared)}")  # 打印预处理后的有效样本数
    
    # 4. 计算评价指标
    print("\n[4] 计算评价指标...")  # 打印步骤 4：计算指标
    metrics_df = evaluate_warning(df_prepared)  # 计算偏热/偏冷评价指标
    print("\n预警评价指标汇总:")  # 打印指标表标题
    print(metrics_df.to_string(index=False))  # 以不显示索引的方式打印指标表
    
    # 5. 输出明细表
    print("\n[5] 输出明细表...")  # 打印步骤 5：输出明细结果
    output_columns = [  # 定义需要导出的明细列清单
        'timestamp', 'age', 'td_true', 'td_pred',  # 基础列：时间、日龄、真实与预测偏差
        'cold_th', 'hot_th',  # 当前时刻偏冷/偏热阈值
        'future_td_true', 'future_age', 'future_cold_th', 'future_hot_th',  # 未来时刻真实值、日龄与阈值
        'pred_hot', 'pred_cold', 'true_hot_future', 'true_cold_future',  # 预测与真实的偏热/偏冷标签
        'severity', 'warning_level'  # 超限幅度与告警等级
    ]  # 列清单定义结束
    
    output_columns = [col for col in output_columns if col in df_prepared.columns]  # 过滤掉不存在的列，避免导出报错
    df_output = df_prepared[output_columns]  # 按筛选后的列取出明细数据
    output_csv = 'warning_evaluation_results.csv'  # 定义输出 CSV 文件名
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 导出为带 BOM 的 UTF-8 CSV，便于 Excel 打开
    print(f"✓ 明细表已保存: {output_csv}")  # 打印明细表保存路径
    
    # 6. 绘制图表
    print("\n[6] 绘制预警图...")  # 打印步骤 6：绘图
    plot_warning_segment(df_prepared, n_points=200)  # 绘制前 200 个点的预警对比图
    
    print("\n" + "=" * 60)  # 打印结束分隔线
    print("✓ 评价完成！")  # 打印完成提示
    print("=" * 60)  # 再次打印结束分隔线















































































































































