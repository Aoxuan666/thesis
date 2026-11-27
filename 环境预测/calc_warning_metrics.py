"""预警仿真日志评估脚本：基于 warning_simulation_log.csv 计算 TD 与 THI 的预警评估指标。

本脚本完成以下工作：
1. 读取预警仿真日志，并打印样本量与列名等基本信息；
2. 依据温度偏差（TD）与温湿度指数（THI）的阈值规则，分别构造真实标签与预测标签；
3. 计算二分类评估指标（Precision / Recall / F1 / Accuracy）并打印混淆矩阵；
4. 统计 THI 分级分布与总体预警分布；
5. 将核心指标保存为 warning_metrics_result.csv（UTF-8 BOM 编码，便于 Excel 打开）。
"""

import pandas as pd  # 导入 pandas 库并重命名为 pd，用于 CSV 读取、表格计算与结果保存
import numpy as np  # 导入 numpy 库并重命名为 np，供数值计算使用（本脚本主要用于兼容与扩展）

# ============================================================
# 读取数据
# ============================================================
df = pd.read_csv('warning_simulation_log.csv')  # 从当前目录读取预警仿真日志 CSV 到 DataFrame

print(f"总样本数: {len(df)}")  # 打印数据集的总行数（样本数量）
print(f"列名: {list(df.columns)}")  # 打印数据集包含的所有列名
print()  # 打印一个空行，使输出更清晰

# ============================================================
# THI 分级函数
# THI <  70            => 舒适 Comfort   => 0 (无预警)
# 70 <= THI <  76      => 警告 Warning   => 1
# 76 <= THI <  81      => 危险 Danger    => 2
# THI >= 81            => 紧急 Emergency => 3
# 预警判定：等级 >= 1 即视为需要预警
# ============================================================
def thi_grade(v):
    """将单个 THI 数值按阈值映射为 0~3 的四级风险等级。

    参数:
        v: 待分级的 THI（温湿度指数）数值。
    返回:
        整数等级：0=舒适(<70)、1=警告([70,76))、2=危险([76,81))、3=紧急(>=81)。
    """
    if v < 70:   return 0  # THI 小于 70 时为舒适等级，返回 0
    elif v < 76: return 1  # THI 在 [70,76) 区间为警告等级，返回 1
    elif v < 81: return 2  # THI 在 [76,81) 区间为危险等级，返回 2
    else:        return 3  # THI 大于等于 81 时为紧急等级，返回 3

# ============================================================
# 真实标签（ground truth）
# ============================================================

# --- TD 真实标签（二分类）---
# 偏热: TD实际 > TD偏热阈值  => 1
# 偏冷: TD实际 < TD偏冷阈值  => 1
# 正常:                      => 0
df['TD_actual_label'] = (  # 构造 TD 的真实二分类标签列：偏热或偏冷记为 1，正常记为 0
    (df['TD实际'] > df['TD偏热阈值']) |  # 判断实际 TD 是否超过偏热阈值
    (df['TD实际'] < df['TD偏冷阈值'])  # 判断实际 TD 是否低于偏冷阈值
).astype(int)  # 将布尔结果转换为整数 0/1

# --- THI 真实标签（基于 THI实际 与阈值规则）---
df['THI_actual_grade'] = df['THI实际'].apply(thi_grade)  # 对实际 THI 逐行应用分级函数，得到 0~3 的真实等级
df['THI_actual_label'] = (df['THI_actual_grade'] >= 1).astype(int)  # 等级 >=1 视为需预警，转为二分类真实标签

# ============================================================
# 预测标签（基于模型预测值与阈值规则）
# ============================================================

# --- TD 预测标签 ---
# 直接使用日志中的 TD等级 字段（0=正常, 1=偏热/偏冷预警）
df['TD_pred_label'] = df['TD等级']  # 直接将日志中的 TD等级 字段作为预测标签

# --- THI 预测标签（基于 THI预测 与同样的分级规则）---
df['THI_pred_grade'] = df['THI预测'].apply(thi_grade)  # 对预测 THI 逐行应用同一分级函数，得到预测等级
df['THI_pred_label'] = (df['THI_pred_grade'] >= 1).astype(int)  # 预测等级 >=1 视为需预警，转为二分类预测标签

# ============================================================
# 计算 Precision / Recall / F1
# ============================================================
def calc_metrics(y_true, y_pred, name):
    """计算二分类预警评估指标并打印混淆矩阵。

    参数:
        y_true: 真实二分类标签序列（1=预警，0=正常）。
        y_pred: 预测二分类标签序列（1=预警，0=正常）。
        name: 用于打印标题的评估对象名称。
    返回:
        包含 TP/FP/FN/TN 及 Precision/Recall/F1/Accuracy 的字典（比率指标保留 4 位小数）。
    """
    TP = int(((y_pred == 1) & (y_true == 1)).sum())  # 真正例：预测预警且实际预警
    FP = int(((y_pred == 1) & (y_true == 0)).sum())  # 假正例：预测预警但实际正常（误报）
    FN = int(((y_pred == 0) & (y_true == 1)).sum())  # 假反例：预测正常但实际预警（漏报）
    TN = int(((y_pred == 0) & (y_true == 0)).sum())  # 真反例：预测正常且实际正常

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # 精确率：预测为预警中实际为预警的比例
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 召回率：实际为预警中被正确预测的比例
    f1        = (2 * precision * recall / (precision + recall)  # F1 值：精确率与召回率的调和平均
                 if (precision + recall) > 0 else 0.0)  # 当分母为 0 时 F1 取 0
    accuracy  = (TP + TN) / (TP + FP + FN + TN)  # 准确率：所有样本中预测正确的比例

    print(f"{'='*60}")  # 打印分隔线
    print(f"  {name} 预警评估指标")  # 打印当前评估对象的标题
    print(f"{'='*60}")  # 打印分隔线
    print(f"  混淆矩阵:")  # 打印混淆矩阵标题
    print(f"              预测=预警   预测=正常")  # 打印混淆矩阵表头
    print(f"  实际=预警     TP={TP:5d}    FN={FN:5d}")  # 打印实际为预警时 TP/FN 的数量
    print(f"  实际=正常     FP={FP:5d}    TN={TN:5d}")  # 打印实际为正常时 FP/TN 的数量
    print()  # 打印空行
    print(f"  Precision : {precision:.4f}  (预测为预警中实际为预警的比例)")  # 打印精确率及其含义
    print(f"  Recall    : {recall:.4f}  (实际为预警中被正确预测的比例)")  # 打印召回率及其含义
    print(f"  F1 Score  : {f1:.4f}")  # 打印 F1 值
    print(f"  Accuracy  : {accuracy:.4f}")  # 打印准确率
    print()  # 打印空行
    return dict(name=name, TP=TP, FP=FP, FN=FN, TN=TN,  # 返回汇总字典：名称与混淆矩阵四项计数
                Precision=round(precision, 4),  # 返回保留 4 位小数的精确率
                Recall=round(recall, 4),  # 返回保留 4 位小数的召回率
                F1=round(f1, 4),  # 返回保留 4 位小数的 F1 值
                Accuracy=round(accuracy, 4))  # 返回保留 4 位小数的准确率

# --- TD 指标 ---
td_metrics  = calc_metrics(df['TD_actual_label'], df['TD_pred_label'],  # 计算 TD 二分类预警指标
                           'TD  (温差偏热/偏冷)')  # 传入 TD 评估对象的显示名称

# --- THI 指标 ---
thi_metrics = calc_metrics(df['THI_actual_label'], df['THI_pred_label'],  # 计算 THI 二分类预警指标
                           'THI (热应激 | Warning/Danger/Emergency)')  # 传入 THI 评估对象的显示名称

# ============================================================
# THI 分级分布统计
# ============================================================
grade_map = {0: '舒适(Comfort)', 1: '警告(Warning)',  # 定义 THI 等级数值到中文名称的映射
             2: '危险(Danger)',  3: '紧急(Emergency)'}  # 继续定义等级 2、3 的中文名称
print(f"{'='*60}")  # 打印分隔线
print("  THI 实际等级分布")  # 打印实际等级分布标题
print(f"{'='*60}")  # 打印分隔线
for g, label in grade_map.items():  # 遍历每个 THI 等级及其显示名称
    cnt = (df['THI_actual_grade'] == g).sum()  # 统计实际等级等于当前等级的样本数
    print(f"  {label:20s}: {cnt:5d} 条")  # 打印该等级的名称与样本条数
print()  # 打印空行
print(f"{'='*60}")  # 打印分隔线
print("  THI 预测等级分布")  # 打印预测等级分布标题
print(f"{'='*60}")  # 打印分隔线
for g, label in grade_map.items():  # 遍历每个 THI 等级及其显示名称
    cnt = (df['THI_pred_grade'] == g).sum()  # 统计预测等级等于当前等级的样本数
    print(f"  {label:20s}: {cnt:5d} 条")  # 打印该等级的名称与样本条数
print()  # 打印空行

# ============================================================
# 总体分布统计
# ============================================================
print(f"{'='*60}")  # 打印分隔线
print("  样本分布统计")  # 打印总体分布统计标题
print(f"{'='*60}")  # 打印分隔线
print(f"  TD  预测为预警: {(df['TD_pred_label']==1).sum():5d} / {len(df)}")  # 打印 TD 预测为预警的数量与总数
print(f"  TD  实际为预警: {(df['TD_actual_label']==1).sum():5d} / {len(df)}")  # 打印 TD 实际为预警的数量与总数
print(f"  THI 预测为预警: {(df['THI_pred_label']==1).sum():5d} / {len(df)}")  # 打印 THI 预测为预警的数量与总数
print(f"  THI 实际为预警: {(df['THI_actual_label']==1).sum():5d} / {len(df)}")  # 打印 THI 实际为预警的数量与总数
print()  # 打印空行

# ============================================================
# 保存结果
# ============================================================
result_df = pd.DataFrame([td_metrics, thi_metrics])  # 将 TD 与 THI 指标字典组成一个 DataFrame
result_df.to_csv('warning_metrics_result.csv', index=False, encoding='utf-8-sig')  # 保存指标结果，不写行索引且带 BOM
print("结果已保存到: warning_metrics_result.csv")  # 提示结果文件已保存
