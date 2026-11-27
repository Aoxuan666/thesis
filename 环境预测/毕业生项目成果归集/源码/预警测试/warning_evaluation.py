"""
温度偏差（TD）预测驱动预警评价模块
用于对测试集中的预警效果进行点级评价

本模块是预警评价体系的核心实现，围绕「温度偏差预测值（td_pred）是否会在未来
某个时点突破偏冷/偏热阈值」这一判断展开。主要能力包括：按规则清理缺失值、
按日龄查询动态阈值、构造未来时刻的真实值与标签、计算偏热/偏冷二分类指标
（精确率、召回率、F1、误报率）、导出评价明细 CSV 并绘制典型时段预警图。
对外提供 main_example 作为可配置的主流程入口（支持指定测试集比例）。
"""

import pandas as pd  # 导入 pandas 并简写为 pd，用于表格数据的读取、变换与导出
import numpy as np  # 导入 numpy 并简写为 np，用于数值计算与布尔掩码运算
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 并简写为 plt，用于绘制预警图
import matplotlib  # 导入 matplotlib 根包，用于设置全局 rcParams 参数
from typing import Tuple, Dict, Optional  # 导入 typing 中的类型注解，用于标注函数参数与返回值类型

# 设置中文字体和禁用警告
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体优先顺序，保证图中中文正常显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，避免负号呈现为方块
import warnings  # 导入 warnings 模块，用于过滤运行期警告
warnings.filterwarnings('ignore')  # 忽略所有警告，保持运行输出简洁


def clean_data(df: pd.DataFrame) -> pd.DataFrame:  # 定义数据清理函数：按四条规则处理缺失值，返回清理后的数据
    """
    数据清理函数，按照规则处理缺失值
    
    规则1: 连续变量用线性插值补充，最多补2个连续缺口
    规则2: 离散变量用前向填充+后向填充
    规则3: 关键列(age, td_true, td_pred)的空值删除对应行
    规则4: future_td_true、future_age的空值不补，最后删除对应行
    
    Args:
        df: 输入DataFrame
        
    Returns:
        清理后的DataFrame
    """
    df = df.copy()  # 复制输入数据，避免修改原始 DataFrame
    
    # 识别连续变量和离散变量
    continuous_cols = ['td_true', 'td_pred', 'CO2', '当前温度', '外界温度', '目标温度',   # 定义需线性插值的连续列清单（续行）
                       '相对湿度', '室外湿度', '风速']  # 连续列清单续行：环境相关列
    discrete_cols = ['age', '日龄']  # 定义需前后向填充的离散列清单
    
    # 规则1: 连续变量线性插值（最多补2个连续缺口）
    for col in continuous_cols:  # 遍历每个连续列
        if col in df.columns:  # 仅当该列真实存在时才处理
            df[col] = df[col].interpolate(method='linear', limit=2)  # 线性插值补缺，最多补 2 个连续空值
    
    # 规则2: 离散变量前向填充+后向填充
    for col in discrete_cols:  # 遍历每个离散列
        if col in df.columns:  # 仅当该列真实存在时才处理
            df[col] = df[col].ffill().bfill()  # 先前向填充，再用后向填充补齐剩余空值
    
    # 规则3: 关键列有空值则删除对应行
    key_cols = ['age', 'td_true', 'td_pred']  # 定义关键列清单（这些列的空值不能填充）
    # 处理列名映射
    key_cols_mapped = []  # 初始化实际存在的关键列名列表
    for col in key_cols:  # 遍历关键列的目标名称
        if col in df.columns:  # 若该列标准名直接存在
            key_cols_mapped.append(col)  # 直接加入待检查列表
        elif col == 'age' and '日龄' in df.columns:  # 若 age 不存在但中文列「日龄」存在
            key_cols_mapped.append('日龄')  # 用中文列名替代
        elif col == 'td_true' and '温度偏差' in df.columns:  # 若 td_true 不存在但中文列「温度偏差」存在
            key_cols_mapped.append('温度偏差')  # 用中文列名替代
    
    if key_cols_mapped:  # 若至少有一个关键列可用于判空
        df = df.dropna(subset=key_cols_mapped)  # 删除关键列中任一为空的行
    
    print("[INFO] After cleaning: {} rows".format(len(df)))  # 打印清理后剩余的行数
    
    return df  # 返回清理后的数据


def get_thresholds(age: int) -> Tuple[float, float]:  # 定义阈值查询函数：按日龄返回（偏冷阈值, 偏热阈值）
    """
    根据日龄返回对应的偏冷阈值和偏热阈值
    
    Args:
        age: 日龄（天数）
        
    Returns:
        (cold_th, hot_th): 偏冷阈值和偏热阈值
    """
    if 1 <= age <= 3:  # 日龄 1~3 天
        return -0.5, 1.0  # 偏冷 -0.5℃，偏热 +1.0℃
    elif 4 <= age <= 7:  # 日龄 4~7 天
        return -0.5, 1.5  # 偏冷 -0.5℃，偏热 +1.5℃
    elif 8 <= age <= 11:  # 日龄 8~11 天
        return -1.0, 2.0  # 偏冷 -1.0℃，偏热 +2.0℃
    elif 12 <= age <= 14:  # 日龄 12~14 天
        return -1.5, 2.5  # 偏冷 -1.5℃，偏热 +2.5℃
    elif 15 <= age <= 21:  # 日龄 15~21 天
        return -2.0, 3.0  # 偏冷 -2.0℃，偏热 +3.0℃
    elif 22 <= age <= 28:  # 日龄 22~28 天
        return -2.5, 3.5  # 偏冷 -2.5℃，偏热 +3.5℃
    else:  # 29及以后
        return -3.0, 4.0  # 兜底阈值：偏冷 -3.0℃，偏热 +4.0℃
def prepare_warning_dataframe(df: pd.DataFrame, horizon: int = 6, use_future_threshold: bool = True) -> pd.DataFrame:  # 定义预处理函数：生成预警标签、阈值与特征，返回处理后的数据
    """
    预处理数据框，生成预警标签和相关特征
    
    Args:
        df: 输入DataFrame，必须包含 timestamp, age, td_true, td_pred 列
        horizon: 预测步长（默认6步，对应30分钟）
        use_future_threshold: 是否使用未来阈值生成预测标签（默认True，推荐）
                             True: 预测标签基于未来阈值（更严格）
                             False: 预测标签基于当前阈值（原始版本）
        
    Returns:
        处理后的DataFrame，包含所有预警相关列
    """
    df = df.copy()  # 复制输入数据，避免修改原始 DataFrame
    
    # 1. 获取当前时刻的阈值
    df[['cold_th', 'hot_th']] = df['age'].apply(  # 对每行日龄查询阈值，拆成两列写入当前偏冷/偏热阈值
        lambda x: pd.Series(get_thresholds(x))  # 调用阈值函数并以 Series 返回，便于展开为两列
    )  # 当前阈值列赋值结束
    
    # 2. 获取未来真实值和未来日龄
    df['future_td_true'] = df['td_true'].shift(-horizon)  # 上移 horizon 行，得到 t+H 时刻的真实温度偏差
    df['future_age'] = df['age'].shift(-horizon)  # 上移 horizon 行，得到 t+H 时刻对应的日龄
    
    # 3. 获取未来阈值
    df[['future_cold_th', 'future_hot_th']] = df['future_age'].apply(  # 按未来日龄查询未来阈值并拆成两列
        lambda x: pd.Series(get_thresholds(int(x))) if pd.notna(x) else pd.Series([np.nan, np.nan])  # 未来日龄非空则查阈值，否则填 NaN
    )  # 未来阈值列赋值结束
    
    # 4. 生成预测预警标签
    if use_future_threshold:  # 若启用未来阈值（推荐），预测标签以未来阈值为判据
        # 默认版本（推荐）：基于未来阈值生成预测标签
        df['pred_hot'] = (df['td_pred'] > df['future_hot_th']).astype(int)  # 预测偏差超未来偏热阈值记为 1，否则 0
        df['pred_cold'] = (df['td_pred'] < df['future_cold_th']).astype(int)  # 预测偏差低于未来偏冷阈值记为 1，否则 0
    else:  # 否则使用当前阈值作为判据（原始版本）
        # 备选版本：基于当前阈值生成预测标签
        df['pred_hot'] = (df['td_pred'] > df['hot_th']).astype(int)  # 预测偏差超当前偏热阈值记为 1
        df['pred_cold'] = (df['td_pred'] < df['cold_th']).astype(int)  # 预测偏差低于当前偏冷阈值记为 1
    
    # 5. 生成真实预警标签（基于未来时刻的阈值）
    df['true_hot_future'] = (df['future_td_true'] > df['future_hot_th']).astype(int)  # 未来真实偏差超未来偏热阈值记为真实偏热 1
    df['true_cold_future'] = (df['future_td_true'] < df['future_cold_th']).astype(int)  # 未来真实偏差低于未来偏冷阈值记为真实偏冷 1
    
    # 6. 计算综合超限幅度（severity）- 基于未来阈值
    df['severity'] = 0.0  # 初始化超限幅度为 0（未超限）
    hot_mask = df['td_pred'] > df['future_hot_th']  # 布尔掩码：预测超未来偏热阈值的行
    cold_mask = df['td_pred'] < df['future_cold_th']  # 布尔掩码：预测低于未来偏冷阈值的行
    
    df.loc[hot_mask, 'severity'] = df.loc[hot_mask, 'td_pred'] - df.loc[hot_mask, 'future_hot_th']  # 偏热行超限幅度 = 预测值 - 偏热阈值
    df.loc[cold_mask, 'severity'] = df.loc[cold_mask, 'future_cold_th'] - df.loc[cold_mask, 'td_pred']  # 偏冷行超限幅度 = 偏冷阈值 - 预测值
    
    # 7. 生成告警等级
    def get_warning_level(severity):  # 定义内部函数：将连续超限幅度映射为离散告警等级
        """把 severity 映射为 none/general/important/severe 四档告警等级。"""  # 内部函数文档字符串说明等级划分
        if severity == 0:  # 超限幅度为 0，未触发预警
            return 'none'  # 返回「无」
        elif 0 < severity < 1.5:  # 幅度在 (0, 1.5) 之间
            return 'general'  # 返回「一般」
        elif 1.5 <= severity < 2.0:  # 幅度在 [1.5, 2.0) 之间
            return 'important'  # 返回「重要」
        else:  # severity >= 2.0
            return 'severe'  # 返回「严重」
    
    df['warning_level'] = df['severity'].apply(get_warning_level)  # 对每行超限幅度应用等级映射，生成告警等级列
    
    # 8. 删除最后 horizon 个样本（没有未来真实值）
    df = df.iloc[:-horizon].reset_index(drop=True)  # 截掉末尾 horizon 行（无未来真值），重置索引
    
    # 规则4: 删除 future_td_true、future_age 的空值行
    df = df.dropna(subset=['future_td_true', 'future_age'])  # 删除未来真值或未来日龄为空的行
    df = df.reset_index(drop=True)  # 再次重置索引，保证索引连续
    
    return df  # 返回预处理完成的数据


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:  # 定义二分类指标计算函数，返回精确率/召回率/F1/误报率字典
    """
    计算二分类评价指标
    
    Args:
        y_true: 真实标签（0/1）
        y_pred: 预测标签（0/1）
        
    Returns:
        包含 precision, recall, f1, far 的字典
    """
    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))  # 构建有效掩码：剔除真实值或预测值为 NaN 的样本
    y_true = y_true[mask]  # 仅保留有效样本的真实标签
    y_pred = y_pred[mask]  # 仅保留有效样本的预测标签
    
    # 计算混淆矩阵
    tp = np.sum((y_pred == 1) & (y_true == 1))  # 真正例 TP：预测为 1 且真实为 1 的数量
    fp = np.sum((y_pred == 1) & (y_true == 0))  # 假正例 FP：预测为 1 但真实为 0 的数量
    fn = np.sum((y_pred == 0) & (y_true == 1))  # 假反例 FN：预测为 0 但真实为 1 的数量
    tn = np.sum((y_pred == 0) & (y_true == 0))  # 真反例 TN：预测为 0 且真实为 0 的数量
    
    # 计算指标
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
        df_prepared: 预处理后的DataFrame
        
    Returns:
        指标汇总表
    """
    # 偏热评价
    hot_metrics = compute_binary_metrics(  # 计算偏热预警的二分类指标
        df_prepared['true_hot_future'].values,  # 传入未来真实偏热标签数组
        df_prepared['pred_hot'].values  # 传入预测偏热标签数组
    )  # 偏热指标计算结束
    
    # 偏冷评价
    cold_metrics = compute_binary_metrics(  # 计算偏冷预警的二分类指标
        df_prepared['true_cold_future'].values,  # 传入未来真实偏冷标签数组
        df_prepared['pred_cold'].values  # 传入预测偏冷标签数组
    )  # 偏冷指标计算结束
    
    # 构建结果表
    results = pd.DataFrame({  # 将偏热/偏冷指标汇总成结果表
        'warning_type': ['hot', 'cold'],  # 预警类型列：偏热、偏冷
        'precision': [hot_metrics['precision'], cold_metrics['precision']],  # 精确率：偏热、偏冷各一行
        'recall': [hot_metrics['recall'], cold_metrics['recall']],  # 召回率：偏热、偏冷各一行
        'f1': [hot_metrics['f1'], cold_metrics['f1']],  # F1 值：偏热、偏冷各一行
        'far': [hot_metrics['far'], cold_metrics['far']]  # 误报率：偏热、偏冷各一行
    })  # 结果表构造结束
    
    return results  # 返回指标汇总表


def plot_warning_segment(  # 定义绘图函数：绘制测试集典型时段的预警对比图并保存
    df_prepared: pd.DataFrame,  # 预处理后的 DataFrame
    start: Optional[int] = None,  # 起始索引，None 表示从头开始
    end: Optional[int] = None,  # 结束索引，None 表示到尾结束
    n_points: int = 200,  # 不指定 start/end 时绘制前 n_points 个点
    figsize: Tuple[int, int] = (14, 6)  # 画布尺寸（宽, 高），单位英寸
) -> None:  # 无返回值，绘图并保存文件
    """
    绘制测试集典型时段的预警图（时间对齐版本）
    
    注意：td_pred 是对未来 t+H 的预测，因此图中所有真实值和阈值都使用 future 版本
    
    Args:
        df_prepared: 预处理后的DataFrame
        start: 起始索引（默认为None，从头开始）
        end: 结束索引（默认为None，到尾结束）
        n_points: 如果不指定start/end，则绘制前n_points个点
        figsize: 图形大小
    """
    # 确定绘制范围
    if start is None and end is None:  # 若起止索引都未指定，绘制前 n_points 个点
        end = min(n_points, len(df_prepared))  # 结束索引取 n_points 与数据总长中较小者
        start = 0  # 起始索引置 0
    elif start is None:  # 若仅未指定起始索引
        start = 0  # 起始索引置 0
    elif end is None:  # 若仅未指定结束索引
        end = len(df_prepared)  # 结束索引取数据总长
    
    df_plot = df_prepared.iloc[start:end].reset_index(drop=True)  # 截取 [start, end) 区间作为绘图数据并重置索引
    
    fig, ax = plt.subplots(figsize=figsize)  # 创建画布与坐标轴对象
    
    x = np.arange(len(df_plot))  # 生成横坐标 0..n-1，代表时间步长
    
    # 绘制预测TD和未来真实TD（时间对齐）
    ax.plot(x, df_plot['td_pred'], label='预测 TD (t+H)', marker='s', markersize=4, linewidth=1.5)  # 绘制预测偏差曲线（方点标记）
    ax.plot(x, df_plot['future_td_true'], label='未来真实 TD (t+H)', marker='o', markersize=4, linewidth=1.5)  # 绘制未来真实偏差曲线（圆点标记）
    
    # 绘制未来阈值曲线（随时间变化）
    ax.plot(x, df_plot['future_hot_th'], color='red', linestyle='--',   # 绘制未来偏热阈值（红色虚线），参数换行
            label='未来偏热阈值', linewidth=1.5, alpha=0.7)  # 续行：图例、线宽与透明度
    ax.plot(x, df_plot['future_cold_th'], color='blue', linestyle='--',   # 绘制未来偏冷阈值（蓝色虚线），参数换行
            label='未来偏冷阈值', linewidth=1.5, alpha=0.7)  # 续行：图例、线宽与透明度
    
    # 标注预测预警（标在预测值上）
    hot_indices = df_plot[df_plot['pred_hot'] == 1].index  # 取出预测偏热样本的行索引
    cold_indices = df_plot[df_plot['pred_cold'] == 1].index  # 取出预测偏冷样本的行索引
    
    if len(hot_indices) > 0:  # 若存在预测偏热样本
        ax.scatter(hot_indices, df_plot.loc[hot_indices, 'td_pred'],   # 在预测值位置绘制偏热散点，参数换行
                  color='red', s=100, marker='^', label='预测偏热', zorder=5, edgecolors='darkred', linewidths=1.5)  # 续行：红色三角、置顶显示
    
    if len(cold_indices) > 0:  # 若存在预测偏冷样本
        ax.scatter(cold_indices, df_plot.loc[cold_indices, 'td_pred'],   # 在预测值位置绘制偏冷散点，参数换行
                  color='blue', s=100, marker='v', label='预测偏冷', zorder=5, edgecolors='darkblue', linewidths=1.5)  # 续行：蓝色倒三角、置顶显示
    
    # 标注真实预警（标在未来真实值上）
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
    
    # 保存图片
    output_file = 'warning_evaluation_plot.png'  # 定义输出图片文件名
    plt.savefig(output_file, dpi=150, bbox_inches='tight')  # 以 150 DPI 保存图片并紧凑裁剪边距
    print("[OK] Plot saved to: {}".format(output_file))  # 打印图片保存路径
    
    # 尝试显示图片（如果环境支持）
    try:  # 尝试显示图像，无界面环境可能失败
        plt.show()  # 显示图形窗口
    except:  # 若显示失败（缺少图形界面等）
        pass  # 静默忽略，不影响已保存的图片
    
    plt.close()  # 关闭画布，释放资源


def main_example(csv_path='部分温湿度数据.csv', td_pred_col=None, test_ratio=None):  # 定义主流程示例函数：串联读取、清理、预处理、评价、导出与绘图
    """
    主流程示例：演示如何使用各个函数
    
    Args:
        csv_path: CSV文件路径
        td_pred_col: 预测值列名。如果为None，会尝试自动检测或使用 td_true 作为预测值
        test_ratio: 测试集比例（默认None，使用全部数据；设置为0.2则使用后20%的数据）
    Returns:
        成功时返回 (预处理后的DataFrame, 指标汇总表)，读取失败时返回 None
    """
    try:  # 捕获文件读取等可能出现的异常
        # 尝试多种编码方式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']  # 列出候选编码，按常见程度排列
        df = None  # 初始化数据框为空，用于记录是否读取成功
        for encoding in encodings:  # 依次尝试每种编码
            try:  # 对单次读取单独捕获解码异常
                df = pd.read_csv(csv_path, encoding=encoding)  # 用当前编码尝试读取 CSV
                print("[OK] Read data successfully (encoding: {}), {} rows".format(encoding, len(df)))  # 打印成功编码与行数
                break  # 读取成功即跳出循环
            except (UnicodeDecodeError, UnicodeError):  # 若编码不匹配导致解码错误
                continue  # 尝试下一种编码
        
        if df is None:  # 若所有编码均读取失败
            print("[ERROR] Failed to read file with all encoding methods")  # 打印读取失败提示
            return  # 提前退出函数
        
        # 显示原始列名
        print("[INFO] Original columns: {}".format(list(df.columns)))  # 打印原始数据列名，便于核对
        
        # 列名映射（处理中文列名）
        col_mapping = {}  # 初始化列名映射字典（原始名 → 统一名）
        for col in df.columns:  # 遍历原始列名
            col_str = str(col)  # 将列名转为字符串便于匹配
            if '监测时间' in col_str or 'time' in col_str.lower():  # 若列名为「监测时间」或含 time
                col_mapping[col] = 'timestamp'  # 映射为统一名 timestamp
            elif '日龄' in col_str or 'age' in col_str.lower():  # 若列名为「日龄」或含 age
                col_mapping[col] = 'age'  # 映射为统一名 age
            elif '温度偏差' in col_str or 'td' in col_str.lower():  # 若列名为「温度偏差」或含 td
                col_mapping[col] = 'td_true'  # 映射为统一名 td_true
        
        # 重命名列
        df = df.rename(columns=col_mapping)  # 按映射关系重命名列，统一列名
        
        # 检查必需列
        required_cols = ['timestamp', 'age', 'td_true']  # 定义后续步骤必需的列名
        missing_cols = [col for col in required_cols if col not in df.columns]  # 计算缺失的必需列
        if missing_cols:  # 若存在缺失列
            print("[ERROR] Missing required columns: {}".format(missing_cols))  # 打印缺失列清单
            print("[INFO] Available columns: {}".format(list(df.columns)))  # 打印当前可用列供排查
            return  # 提前退出函数
        
        # 处理 td_pred 列
        if 'td_pred' not in df.columns:  # 若数据中没有预测列
            if td_pred_col and td_pred_col in df.columns:  # 若调用方指定了预测列名且该列存在
                df['td_pred'] = df[td_pred_col]  # 用指定列作为预测值
            else:  # 否则
                print("[WARNING] td_pred column not found, using td_true + noise as demo")  # 打印警告：用真实值加噪声演示
                df['td_pred'] = df['td_true'] + np.random.normal(0, 0.1, len(df))  # 叠加均值 0、标准差 0.1 的噪声
        
        print("[OK] Processed columns: {}".format(list(df.columns)))  # 打印重命名后的列名
        
        # 提取测试数据
        if test_ratio is not None and test_ratio > 0 and test_ratio < 1:  # 若指定了合法的测试比例 (0,1)
            test_start_idx = int(len(df) * (1 - test_ratio))  # 计算末尾 test_ratio 比例数据的起始下标
            df = df.iloc[test_start_idx:].reset_index(drop=True)  # 截取末尾测试数据并重置索引
            print("[INFO] Using last {:.0f}% data for testing, {} rows".format(test_ratio * 100, len(df)))  # 打印测试集占比与行数
        else:  # 未指定有效比例
            print("[INFO] Using all data for testing, {} rows".format(len(df)))  # 打印使用全部数据的提示
            
    except FileNotFoundError:  # 捕获文件不存在的异常
        print("[ERROR] File not found: {}".format(csv_path))  # 打印文件未找到提示
        print("[INFO] Please ensure CSV contains: timestamp, age, td_true, td_pred")  # 提示 CSV 应包含的列
        return  # 提前退出函数
    # 1.5 数据清理
    print("\n[STEP 0] Data cleaning...")  # 打印步骤 0：数据清理
    df = clean_data(df)  # 按规则清理缺失值
    # 2. 数据预处理
    print("\n[STEP 1] Data preprocessing...")  # 打印步骤 1：数据预处理
    df_prepared = prepare_warning_dataframe(df, horizon=6)  # 构造预警标签与特征（预测步长 6）
    print("[OK] Preprocessing complete, valid samples: {}".format(len(df_prepared)))  # 打印有效样本数
    # 3. 计算评价指标
    print("\n[STEP 2] Computing metrics...")  # 打印步骤 2：计算指标
    metrics_df = evaluate_warning(df_prepared)  # 计算偏热/偏冷评价指标
    print("\nWarning evaluation metrics:")  # 打印指标表标题
    print(metrics_df.to_string(index=False))  # 以无索引形式打印指标表
    # 4. 输出明细表
    print("\n[STEP 3] Saving detailed results...")  # 打印步骤 3：保存明细结果
    output_columns = [  # 定义需要导出的明细列清单
        'timestamp', 'age', 'td_true', 'td_pred',  # 基础列：时间、日龄、真实与预测偏差
        'cold_th', 'hot_th',  # 当前时刻偏冷/偏热阈值
        'future_td_true', 'future_age', 'future_cold_th', 'future_hot_th',  # 未来时刻真实值、日龄与阈值
        'pred_hot', 'pred_cold', 'true_hot_future', 'true_cold_future',  # 预测与真实的偏热/偏冷标签
        'severity', 'warning_level'  # 超限幅度与告警等级
    ]  # 列清单定义结束
    # 只输出存在的列
    output_columns = [col for col in output_columns if col in df_prepared.columns]  # 过滤掉不存在的列，避免导出报错
    df_output = df_prepared[output_columns]  # 按筛选后的列取出明细数据
    output_csv = 'warning_evaluation_results.csv'  # 定义输出 CSV 文件名
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 导出为带 BOM 的 UTF-8 CSV，便于 Excel 打开
    print("[OK] Results saved to: {}".format(output_csv))  # 打印结果保存路径
    # 5. 绘制典型时段图
    print("\n[STEP 4] Plotting warning chart...")  # 打印步骤 4：绘制预警图
    plot_warning_segment(df_prepared, n_points=200)  # 绘制前 200 个点的预警对比图
    return df_prepared, metrics_df  # 返回预处理数据与指标表，供调用方继续使用
if __name__ == '__main__':  # 当本文件作为主程序直接运行时
    main_example()  # 调用主流程示例，使用全部数据执行完整预警评价

