# -*- coding: utf-8 -*-
"""
预警系统评估模块 v3（与预测脚本口径一致）
功能：
1. 加载 warning_simulation_log.csv
2. 按与预测阶段一致的规则重新计算 TD / THI 等级
3. 计算回归指标、多分类指标、综合预警二分类指标
4. 保存核心指标结果到 warning_metrics_result.csv

补充说明：本模块在评估阶段严格复用预测脚本的温度区间与预警裕度口径（WARN_MARGIN=2.5），
不再对 CSV 中已保存的预测值做二次四舍五入；同时额外导出带等级的明细文件便于逐条核查。
"""

import os  # 导入 os 模块，用于获取结果文件的绝对路径
import warnings  # 导入 warnings 模块，用于屏蔽第三方库的告警
warnings.filterwarnings('ignore')  # 忽略运行过程中的警告信息，保持输出整洁

import numpy as np  # 导入 numpy，用于数组转换与均方误差开方等数值计算
import pandas as pd  # 导入 pandas，用于 CSV 读取、数值清洗与结果写出
from sklearn.metrics import (  # 从 sklearn.metrics 导入各类评估指标函数
    r2_score,  # 决定系数 R2
    mean_squared_error,  # 均方误差 MSE
    mean_absolute_error,  # 平均绝对误差 MAE
    confusion_matrix,  # 混淆矩阵
    accuracy_score,  # 准确率
    precision_recall_fscore_support  # 精确率、召回率、F1 与支持度
)

# ============================================================
# 温度规则（必须与 warning_simulation_v2.py 保持一致）
# ============================================================
TEMP_RANGES = [  # 定义各日龄段对应的适宜温度上下限列表
    (1, 3, 34, 35),  # 1~3 日龄：适宜温度 34~35℃
    (4, 7, 32, 33),  # 4~7 日龄：适宜温度 32~33℃
    (8, 15, 30, 31),  # 8~15 日龄：适宜温度 30~31℃
    (16, 23, 28, 29),  # 16~23 日龄：适宜温度 28~29℃
    (24, 31, 26, 27),  # 24~31 日龄：适宜温度 26~27℃
    (32, 39, 24, 25),  # 32~39 日龄：适宜温度 24~25℃
    (40, 999, 22, 23),  # 40 日龄以上：适宜温度 22~23℃
]

# 注意：这里改成 2.5，与预测脚本一致
WARN_MARGIN = 2.5  # 预警温度裕度（℃），与预测脚本保持一致的 2.5


# ============================================================
# 工具函数
# ============================================================
def safe_read_csv(csv_path):
    """稳健读取 CSV：依次尝试多种编码，直至成功。

    参数:
        csv_path: 待读取的 CSV 文件路径。
    返回:
        成功读取得到的 pandas DataFrame。
    异常:
        当所有候选编码都失败时抛出 ValueError，并附带各编码的失败原因。
    """
    tried = []  # 记录各编码尝试的失败信息
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'gb2312']:  # 按常见程度依次尝试五种编码
        try:  # 尝试用当前编码读取
            df = pd.read_csv(csv_path, encoding=enc)  # 以当前编码读取 CSV 到 DataFrame
            print(f'  [OK] 成功读取: {csv_path}  encoding={enc}')  # 打印成功信息及实际使用的编码
            return df  # 返回读取成功的数据
        except Exception as e:  # 捕获当前编码读取失败的任何异常
            tried.append(f'{enc}: {repr(e)}')  # 将编码及异常信息追加到失败记录
    raise ValueError(f'无法读取文件: {csv_path}\n尝试情况:\n' + '\n'.join(tried))  # 全部失败则抛出异常并汇总尝试情况


def get_temp_range(age):
    """根据肉鸡日龄获取对应的适宜温度区间。

    参数:
        age: 肉鸡日龄，可转换为整数。
    返回:
        (下限温度, 上限温度, 区间中值) 三元组；未匹配时返回默认的 (22, 23, 22.5)。
    """
    age = int(age)  # 将日龄统一转换为整数，便于区间比较
    for lo, hi, tlo, thi in TEMP_RANGES:  # 遍历每个日龄区间配置
        if lo <= age <= hi:  # 若当前日龄落在该区间内
            return tlo, thi, (tlo + thi) / 2.0  # 返回温度下限、上限及中值
    return 22, 23, 22.5  # 未找到匹配区间时返回默认温度范围


def classify_td(td_val, target_temp, age):
    """
    TD 分类（TD保留1位小数）
    返回:
        -1 -> 偏冷
         0 -> 正常
         1 -> 偏热

    参数:
        td_val: 温度偏差 TD（℃）。
        target_temp: 当前目标温度（℃），用于换算实际舍内温度。
        age: 肉鸡日龄，用于查询适宜温度区间。
    """
    td_val = round(float(td_val), 1)  # 将 TD 转为浮点数并保留 1 位小数
    tlo, thi, _ = get_temp_range(age)  # 根据日龄获取适宜温度的下限与上限
    cur_temp = td_val + target_temp  # 由目标温度加偏差还原出当前实际温度
    hot_thresh = thi + WARN_MARGIN  # 偏热阈值 = 适宜温度上限 + 预警裕度
    cold_thresh = tlo - WARN_MARGIN  # 偏冷阈值 = 适宜温度下限 - 预警裕度

    if cur_temp > hot_thresh:  # 实际温度高于偏热阈值
        return 1  # 判定为偏热，返回 1
    elif cur_temp < cold_thresh:  # 实际温度低于偏冷阈值
        return -1  # 判定为偏冷，返回 -1
    else:  # 温度处于正常区间
        return 0  # 判定为正常，返回 0


def classify_thi(thi_val):
    """
    THI 分类（THI保留3位小数）
    0 -> 舒适 (<70)
    1 -> 警告 [70,76)
    2 -> 危险 [76,81)
    3 -> 紧急 (>=81)
    """
    thi_val = round(float(thi_val), 3)  # 将 THI 转为浮点数并保留 3 位小数
    if thi_val < 70:  # THI 小于 70
        return 0  # 舒适等级
    elif thi_val < 76:  # THI 在 [70,76) 区间
        return 1  # 警告等级
    elif thi_val < 81:  # THI 在 [76,81) 区间
        return 2  # 危险等级
    else:  # THI 大于等于 81
        return 3  # 紧急等级


def load_results(csv_path):
    """加载预测结果 CSV，不再二次四舍五入，直接按文件中的值评估。

    参数:
        csv_path: 预测结果 CSV 路径。
    返回:
        完成编码读取、列校验、数值转换与缺失值清洗后的 DataFrame。
    异常:
        缺少必要列或清洗后无有效样本时抛出 ValueError。
    """
    df = safe_read_csv(csv_path)  # 使用稳健读取函数加载 CSV

    # 必要列
    required_cols = ['TD预测', 'TD实际', 'THI预测', 'THI实际', '日龄', '目标温度']  # 列出评估必需的列名
    missing_cols = [c for c in required_cols if c not in df.columns]  # 筛选出数据中缺失的必要列
    if missing_cols:  # 若存在缺失的必要列
        raise ValueError(f'缺少必要列: {missing_cols}')  # 抛出异常终止程序

    # 转为数值
    for col in required_cols:  # 遍历每个必要列
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 强制转为数值，无法转换的置为 NaN

    # 可选时间列
    if '当前时间' in df.columns:  # 若数据中包含时间列
        df['当前时间'] = pd.to_datetime(df['当前时间'], errors='coerce')  # 将时间列解析为日期时间类型

    # 删除缺失
    before_n = len(df)  # 记录清洗前的样本数
    df = df.dropna(subset=required_cols).reset_index(drop=True)  # 删除必要列含缺失的行并重置索引
    after_n = len(df)  # 记录清洗后的样本数

    print(f'  原始样本: {before_n}')  # 打印清洗前样本数
    print(f'  有效样本: {after_n}')  # 打印清洗后有效样本数
    print(f'  删除样本: {before_n - after_n}')  # 打印被删除的样本数

    if after_n == 0:  # 若清洗后没有任何有效样本
        raise ValueError('清洗后有效样本为 0，请检查 CSV 文件内容')  # 抛出异常提示数据异常

    return df  # 返回清洗后的有效数据


# ============================================================
# 评估函数
# ============================================================
def eval_regression(y_true, y_pred, name):
    """回归指标评估：计算并打印 R2、MAE、RMSE 与 MAPE。

    参数:
        y_true: 真实值序列。
        y_pred: 预测值序列。
        name: 评估对象名称，用于打印标题。
    返回:
        包含 R2/MAE/RMSE/MAPE 的字典。
    """
    y_true = np.asarray(y_true, dtype=float)  # 将真实值转换为浮点型 numpy 数组
    y_pred = np.asarray(y_pred, dtype=float)  # 将预测值转换为浮点型 numpy 数组

    r2 = r2_score(y_true, y_pred)  # 计算决定系数 R2
    mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差 MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算均方误差并开方得到 RMSE
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100  # 计算平均绝对百分比误差，加 1e-8 防止除零

    print(f'\n[{name}]')  # 打印评估对象标题
    print(f'  R2   = {r2:.4f}')  # 打印决定系数
    print(f'  MAE  = {mae:.4f}')  # 打印平均绝对误差
    print(f'  RMSE = {rmse:.4f}')  # 打印均方根误差
    print(f'  MAPE = {mape:.2f}%')  # 打印平均绝对百分比误差

    return {  # 返回各项回归指标组成的字典
        'R2': r2,  # 决定系数
        'MAE': mae,  # 平均绝对误差
        'RMSE': rmse,  # 均方根误差
        'MAPE': mape  # 平均绝对百分比误差
    }


def eval_multiclass(y_true, y_pred, class_names, name):
    """多分类评估：计算宏平均指标、各类别指标与混淆矩阵。

    参数:
        y_true: 真实类别标签序列。
        y_pred: 预测类别标签序列。
        class_names: 类别数值到中文名称的映射字典。
        name: 评估对象名称，用于打印标题。
    返回:
        包含 Accuracy、宏平均 P/R/F1、混淆矩阵与类别顺序的字典。
    """
    y_true = np.asarray(y_true)  # 将真实标签转换为 numpy 数组
    y_pred = np.asarray(y_pred)  # 将预测标签转换为 numpy 数组

    class_order = sorted(set(y_true) | set(y_pred))  # 合并真实与预测中出现的所有类别并排序

    acc = accuracy_score(y_true, y_pred)  # 计算总体准确率
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(  # 计算宏平均精确率、召回率与 F1
        y_true, y_pred, average='macro', zero_division=0  # 使用 macro 平均，零除时置 0
    )
    prec_each, rec_each, f1_each, support_each = precision_recall_fscore_support(  # 计算每个类别各自的指标
        y_true, y_pred, labels=class_order, average=None, zero_division=0  # 按类别顺序逐类计算
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_order)  # 按类别顺序计算混淆矩阵

    print(f'\n[{name}]')  # 打印评估对象标题
    print(f'  Accuracy        = {acc:.4f}')  # 打印准确率
    print(f'  Precision_macro = {prec_macro:.4f}')  # 打印宏平均精确率
    print(f'  Recall_macro    = {rec_macro:.4f}')  # 打印宏平均召回率
    print(f'  F1_macro        = {f1_macro:.4f}')  # 打印宏平均 F1 值

    print('\n  混淆矩阵（行=实际, 列=预测）')  # 打印混淆矩阵说明
    header = '       ' + ''.join([f'{class_names.get(c, str(c)):>10}' for c in class_order])  # 构造混淆矩阵表头
    print(header)  # 打印表头
    for i, true_c in enumerate(class_order):  # 遍历每个真实类别，构造矩阵的一行
        row = f'{class_names.get(true_c, str(true_c)):>6} '  # 行首为真实类别名称
        for j in range(len(class_order)):  # 遍历每个预测类别
            row += f'{cm[i, j]:>10}'  # 拼接混淆矩阵对应位置的数值
        print(row)  # 打印该真实类别对应的一行

    print('\n  各类别指标')  # 打印各类别指标标题
    for i, c in enumerate(class_order):  # 遍历每个类别
        print(  # 打印该类别的详细指标
            f'  {class_names.get(c, str(c))}: '  # 类别名称
            f'P={prec_each[i]:.4f} '  # 精确率
            f'R={rec_each[i]:.4f} '  # 召回率
            f'F1={f1_each[i]:.4f} '  # F1 值
            f'Support={int(support_each[i])}'  # 该类别的样本支持数
        )

    return {  # 返回多分类评估结果字典
        'Accuracy': acc,  # 准确率
        'Precision_macro': prec_macro,  # 宏平均精确率
        'Recall_macro': rec_macro,  # 宏平均召回率
        'F1_macro': f1_macro,  # 宏平均 F1 值
        'ConfusionMatrix': cm,  # 混淆矩阵
        'ClassOrder': class_order  # 类别顺序
    }


def eval_binary(y_true, y_pred, name):
    """二分类评估：计算 TP/TN/FP/FN 以及精确率、召回率、F1 和准确率。

    参数:
        y_true: 真实布尔标签序列。
        y_pred: 预测布尔标签序列。
        name: 评估对象名称，用于打印标题。
    返回:
        包含混淆矩阵四项计数及各项比率的字典。
    """
    y_true = pd.Series(y_true).astype(bool)  # 将真实标签转为布尔型 Series
    y_pred = pd.Series(y_pred).astype(bool)  # 将预测标签转为布尔型 Series

    TP = int((y_pred & y_true).sum())  # 真正例：预测预警且实际预警
    TN = int((~y_pred & ~y_true).sum())  # 真反例：预测正常且实际正常
    FP = int((y_pred & ~y_true).sum())  # 假正例：误报
    FN = int((~y_pred & y_true).sum())  # 假反例：漏报

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # 精确率，分母为 0 时取 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 召回率，分母为 0 时取 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0  # F1 值，分母为 0 时取 0
    acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0  # 准确率，样本为空时取 0

    print(f'\n[{name}]')  # 打印评估对象标题
    print(f'  TP (正确预警) = {TP}')  # 打印正确预警数
    print(f'  TN (正确正常) = {TN}')  # 打印正确正常数
    print(f'  FP (误报)     = {FP}')  # 打印误报数
    print(f'  FN (漏报)     = {FN}')  # 打印漏报数
    print(f'  Accuracy      = {acc:.4f}')  # 打印准确率
    print(f'  Precision     = {prec:.4f}')  # 打印精确率
    print(f'  Recall        = {rec:.4f}')  # 打印召回率
    print(f'  F1            = {f1:.4f}')  # 打印 F1 值

    return {  # 返回二分类评估结果字典
        'TP': TP,  # 真正例数
        'TN': TN,  # 真反例数
        'FP': FP,  # 假正例数
        'FN': FN,  # 假反例数
        'Accuracy': acc,  # 准确率
        'Precision': prec,  # 精确率
        'Recall': rec,  # 召回率
        'F1': f1  # F1 值
    }


# ============================================================
# 主程序
# ============================================================
def main():
    """评估主流程：加载数据、重算等级、执行回归/分类评估并保存结果文件。"""
    print('=' * 70)  # 打印分隔线
    print('预警系统评估模块 v3（口径一致版）')  # 打印程序标题
    print('=' * 70)  # 打印分隔线

    # --------------------------------------------------------
    # [1] 加载结果
    # --------------------------------------------------------
    print('\n[1] 加载预测结果...')  # 打印当前步骤提示
    csv_path = 'warning_simulation_log.csv'  # 指定预测结果 CSV 文件路径
    res = load_results(csv_path)  # 调用加载函数读取并清洗数据

    # --------------------------------------------------------
    # [2] 重新计算等级（按 CSV 中保存的值直接计算，不再二次取整）
    # --------------------------------------------------------
    print('\n[2] 重新计算预警等级...')  # 打印当前步骤提示
    res['TD预测等级'] = res.apply(  # 按行计算 TD 预测等级
        lambda row: classify_td(row['TD预测'], row['目标温度'], row['日龄']),  # 使用预测 TD、目标温度与日龄分类
        axis=1  # axis=1 表示按行应用函数
    )
    res['TD实际等级'] = res.apply(  # 按行计算 TD 实际等级
        lambda row: classify_td(row['TD实际'], row['目标温度'], row['日龄']),  # 使用实际 TD、目标温度与日龄分类
        axis=1  # 按行应用函数
    )

    res['THI预测等级'] = res['THI预测'].apply(classify_thi)  # 对预测 THI 应用分级函数得到预测等级
    res['THI实际等级'] = res['THI实际'].apply(classify_thi)  # 对实际 THI 应用分级函数得到实际等级

    # 综合预警：TD异常 或 THI达到警告及以上
    res['预测触发预警'] = (res['TD预测等级'] != 0) | (res['THI预测等级'] >= 1)  # 预测侧：TD 偏热/偏冷或 THI 达到警告级即触发
    res['实际触发预警'] = (res['TD实际等级'] != 0) | (res['THI实际等级'] >= 1)  # 实际侧：按同样规则判定是否触发预警

    print('  [OK] 等级计算完成')  # 提示等级计算完成

    # --------------------------------------------------------
    # [3] 回归评估
    # --------------------------------------------------------
    print('\n' + '=' * 70)  # 打印分隔线
    print('回归指标评估')  # 打印回归评估标题
    print('=' * 70)  # 打印分隔线

    td_reg = eval_regression(res['TD实际'], res['TD预测'], 'TD温度偏差')  # 计算 TD 的回归指标
    thi_reg = eval_regression(res['THI实际'], res['THI预测'], 'THI温湿度指数')  # 计算 THI 的回归指标

    # --------------------------------------------------------
    # [4] 统计信息
    # --------------------------------------------------------
    print('\n' + '=' * 70)  # 打印分隔线
    print('预警统计')  # 打印预警统计标题
    print('=' * 70)  # 打印分隔线

    total = len(res)  # 统计有效样本总数
    pred_warn = int(res['预测触发预警'].sum())  # 统计预测触发预警的样本数
    real_warn = int(res['实际触发预警'].sum())  # 统计实际触发预警的样本数

    print(f'\n总样本数: {total}')  # 打印总样本数
    print(f'预测触发预警: {pred_warn} ({pred_warn / total * 100:.2f}%)')  # 打印预测预警数量与占比
    print(f'实际触发预警: {real_warn} ({real_warn / total * 100:.2f}%)')  # 打印实际预警数量与占比

    print('\n[TD预测状态统计]')  # 打印 TD 预测状态统计标题
    for level in [-1, 0, 1]:  # 遍历偏冷、正常、偏热三种状态
        count = int((res['TD预测等级'] == level).sum())  # 统计该等级的样本数
        state = {-1: '偏冷', 0: '正常', 1: '偏热'}[level]  # 将等级数值映射为中文状态名
        print(f'  {state}: {count}')  # 打印状态名与数量

    print('\n[TD实际状态统计]')  # 打印 TD 实际状态统计标题
    for level in [-1, 0, 1]:  # 遍历偏冷、正常、偏热三种状态
        count = int((res['TD实际等级'] == level).sum())  # 统计该等级的实际样本数
        state = {-1: '偏冷', 0: '正常', 1: '偏热'}[level]  # 映射为中文状态名
        print(f'  {state}: {count}')  # 打印状态名与数量

    print('\n[THI预测分级统计]')  # 打印 THI 预测分级统计标题
    for level in [0, 1, 2, 3]:  # 遍历四个 THI 等级
        count = int((res['THI预测等级'] == level).sum())  # 统计该预测等级的样本数
        state = {0: '舒适', 1: '警告', 2: '危险', 3: '紧急'}[level]  # 映射为中文等级名
        print(f'  {state}: {count}')  # 打印等级名与数量

    print('\n[THI实际分级统计]')  # 打印 THI 实际分级统计标题
    for level in [0, 1, 2, 3]:  # 遍历四个 THI 等级
        count = int((res['THI实际等级'] == level).sum())  # 统计该实际等级的样本数
        state = {0: '舒适', 1: '警告', 2: '危险', 3: '紧急'}[level]  # 映射为中文等级名
        print(f'  {state}: {count}')  # 打印等级名与数量

    # --------------------------------------------------------
    # [5] 分类评估
    # --------------------------------------------------------
    print('\n' + '=' * 70)  # 打印分隔线
    print('分类评估')  # 打印分类评估标题
    print('=' * 70)  # 打印分隔线

    td_cls = eval_multiclass(  # 执行 TD 三分类评估
        res['TD实际等级'],  # 传入实际等级
        res['TD预测等级'],  # 传入预测等级
        {-1: '偏冷', 0: '正常', 1: '偏热'},  # 类别名称映射
        'TD三分类评估'  # 评估名称
    )

    thi_cls = eval_multiclass(  # 执行 THI 四分类评估
        res['THI实际等级'],  # 传入实际等级
        res['THI预测等级'],  # 传入预测等级
        {0: '舒适', 1: '警告', 2: '危险', 3: '紧急'},  # 类别名称映射
        'THI四分类评估'  # 评估名称
    )

    binary_eval = eval_binary(  # 执行综合预警二分类评估
        res['实际触发预警'],  # 传入实际是否触发预警
        res['预测触发预警'],  # 传入预测是否触发预警
        '综合预警二分类评估'  # 评估名称
    )

    # --------------------------------------------------------
    # [6] 保存核心指标
    # --------------------------------------------------------
    print('\n' + '=' * 70)  # 打印分隔线
    print('保存结果')  # 打印保存结果标题
    print('=' * 70)  # 打印分隔线

    metric_rows = [  # 构造待保存的核心指标行列表
        {'模块': 'TD回归', '指标': 'R2', '值': td_reg['R2']},  # TD 回归 R2
        {'模块': 'TD回归', '指标': 'MAE', '值': td_reg['MAE']},  # TD 回归 MAE
        {'模块': 'TD回归', '指标': 'RMSE', '值': td_reg['RMSE']},  # TD 回归 RMSE
        {'模块': 'TD回归', '指标': 'MAPE', '值': td_reg['MAPE']},  # TD 回归 MAPE

        {'模块': 'THI回归', '指标': 'R2', '值': thi_reg['R2']},  # THI 回归 R2
        {'模块': 'THI回归', '指标': 'MAE', '值': thi_reg['MAE']},  # THI 回归 MAE
        {'模块': 'THI回归', '指标': 'RMSE', '值': thi_reg['RMSE']},  # THI 回归 RMSE
        {'模块': 'THI回归', '指标': 'MAPE', '值': thi_reg['MAPE']},  # THI 回归 MAPE

        {'模块': 'TD三分类', '指标': 'Accuracy', '值': td_cls['Accuracy']},  # TD 三分类准确率
        {'模块': 'TD三分类', '指标': 'Precision_macro', '值': td_cls['Precision_macro']},  # TD 三分类宏平均精确率
        {'模块': 'TD三分类', '指标': 'Recall_macro', '值': td_cls['Recall_macro']},  # TD 三分类宏平均召回率
        {'模块': 'TD三分类', '指标': 'F1_macro', '值': td_cls['F1_macro']},  # TD 三分类宏平均 F1

        {'模块': 'THI四分类', '指标': 'Accuracy', '值': thi_cls['Accuracy']},  # THI 四分类准确率
        {'模块': 'THI四分类', '指标': 'Precision_macro', '值': thi_cls['Precision_macro']},  # THI 四分类宏平均精确率
        {'模块': 'THI四分类', '指标': 'Recall_macro', '值': thi_cls['Recall_macro']},  # THI 四分类宏平均召回率
        {'模块': 'THI四分类', '指标': 'F1_macro', '值': thi_cls['F1_macro']},  # THI 四分类宏平均 F1

        {'模块': '综合预警二分类', '指标': 'TP', '值': binary_eval['TP']},  # 二分类真正例
        {'模块': '综合预警二分类', '指标': 'TN', '值': binary_eval['TN']},  # 二分类真反例
        {'模块': '综合预警二分类', '指标': 'FP', '值': binary_eval['FP']},  # 二分类假正例
        {'模块': '综合预警二分类', '指标': 'FN', '值': binary_eval['FN']},  # 二分类假反例
        {'模块': '综合预警二分类', '指标': 'Accuracy', '值': binary_eval['Accuracy']},  # 二分类准确率
        {'模块': '综合预警二分类', '指标': 'Precision', '值': binary_eval['Precision']},  # 二分类精确率
        {'模块': '综合预警二分类', '指标': 'Recall', '值': binary_eval['Recall']},  # 二分类召回率
        {'模块': '综合预警二分类', '指标': 'F1', '值': binary_eval['F1']},  # 二分类 F1
    ]

    metrics_df = pd.DataFrame(metric_rows)  # 将指标行列表转换为 DataFrame
    out_csv = 'warning_metrics_result.csv'  # 指定核心指标输出文件名
    metrics_df.to_csv(out_csv, index=False, encoding='utf-8-sig')  # 保存指标文件，不带行索引且带 BOM
    print(f'  [OK] 核心指标已保存: {os.path.abspath(out_csv)}')  # 打印核心指标文件的绝对路径

    # 也把带等级的明细保存一份，方便你核查
    detail_csv = 'warning_simulation_log_with_levels.csv'  # 指定带等级明细的输出文件名
    res.to_csv(detail_csv, index=False, encoding='utf-8-sig')  # 保存带等级的明细数据
    print(f'  [OK] 明细结果已保存: {os.path.abspath(detail_csv)}')  # 打印明细文件的绝对路径

    print('\n评估完成。')  # 打印评估结束提示


if __name__ == '__main__':  # 仅当脚本被直接运行时执行主流程
    main()  # 调用主函数启动评估
