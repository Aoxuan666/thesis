"""
GRU + 小波变换 + ECA 通道注意力模型训练与评估脚本（温度偏差预测）。

本脚本在 GRU+ECA 基础上引入 db4 小波的多分辨率分析：仅对目标变量"温度偏差"按滑动窗口做
三层小波分解（得到 1 个近似系数与 3 个细节系数），其余特征保持原样，从而既提取趋势/波动信息
又避免未来信息泄漏。脚本完成温湿度数据的读取与缺失值处理、按时间顺序划分数据集、
小波分解与可视化、仅在训练集上拟合的归一化、时间窗口序列构造、GRU+小波+ECA 模型构建与
带早停的训练、测试集评估与绘图、模型保存，以及用全量数据归一化器对新数据集进行预测评估的完整流程。
"""
import pandas as pd  # 导入 pandas 数据处理库，简写为 pd，用于读取 CSV 与表格操作
import numpy as np  # 导入 numpy 数值计算库，简写为 np，用于数组与矩阵运算
import random  # 导入 random 模块（供下方已注释的 set_seed 使用，当前未启用）
import os  # 导入 os 模块（供已注释的 set_seed 设置环境变量，当前未启用）
import time  # 导入 time 模块，用于统计训练耗时

# matplotlib后端配置：必须在导入pyplot之前设置（否则 Agg 后端可能不生效）
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片（原注释保留）
import matplotlib.pyplot as plt  # 导入 pyplot 绘图模块，简写为 plt

from sklearn.preprocessing import MinMaxScaler  # 导入最小-最大归一化器，把特征缩放到 [0,1]
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R²、MSE、MAE

import torch  # 导入 PyTorch 主库，用于张量计算、GPU 加速与模型训练
import torch.nn as nn  # 导入神经网络模块，用于构建 GRU、Linear 等层
import torch.optim as optim  # 导入优化器模块，这里使用 Adam
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载器与张量数据集封装工具

import pywt  # 小波变换库（原注释保留），用于 db4 小波分解
import copy  # 导入 copy 模块，用 deepcopy 保存早停过程中的最佳模型权重
# 以下为已整体注释掉的随机种子设置函数，保留原文不改（当前未启用，故不再逐行加注释）
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(99)
# ===================== 辅助函数 =====================
def get_model_size(model):
    """计算模型大小（MB）。

    Args:
        model: 一个 PyTorch 模型，统计其参数与缓冲区的字节占用。
    Returns:
        float: 参数加缓冲区占用字节数换算得到的 MB 数值。
    """
    param_size = 0  # 初始化参数字节数累加器
    for param in model.parameters():  # 遍历模型的所有参数
        param_size += param.nelement() * param.element_size()  # 累加：元素个数 × 单元素字节数
    buffer_size = 0  # 初始化缓冲区字节数累加器
    for buffer in model.buffers():  # 遍历模型的所有缓冲区（如 BatchNorm 统计量）
        buffer_size += buffer.nelement() * buffer.element_size()  # 累加缓冲区字节数
    size_all_mb = (param_size + buffer_size) / 1024**2  # 把字节总数换算成 MB（除以 1024 的平方）
    return size_all_mb  # 返回模型大小（MB）


def format_time(seconds):
    """格式化时间显示：按秒/分/小时自适应输出。

    Args:
        seconds: 以秒为单位的时长。
    Returns:
        str: 保留两位小数并带中文单位的时长字符串。
    """
    if seconds < 60:  # 若时长小于 60 秒
        return f"{seconds:.2f}秒"  # 直接以秒显示
    elif seconds < 3600:  # 若时长小于 3600 秒（1 小时）
        minutes = seconds / 60  # 换算为分钟
        return f"{minutes:.2f}分钟"  # 以分钟显示
    else:  # 时长达到 1 小时及以上
        hours = seconds / 3600  # 换算为小时
        return f"{hours:.2f}小时"  # 以小时显示


# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若 CUDA 可用则选 GPU，否则回退到 CPU
print(f"使用设备: {device}")  # 打印当前选定的计算设备
if torch.cuda.is_available():  # 判断是否存在可用的 NVIDIA GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 数量

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图默认中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号显示为方块


# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，按顺序尝试以兼容不同来源的 CSV
data = None  # 初始化数据占位变量为空
for enc in encodings:  # 遍历每种候选编码
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 读取成功则打印所用编码
        break  # 成功后跳出尝试循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获编码解码错误
        continue  # 尝试下一种编码

if data is None:  # 若所有编码均失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔生成与数据等长的时间戳列

features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 定义候选输入特征列表
target = '温度偏差'  # 指定预测目标为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中真实存在的特征列

# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印 60 个等号组成的分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计标题
for col in features + [target]:  # 遍历所有特征列与目标列
    if col in data.columns:  # 仅处理数据中存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比（百分比）
        if nan_count > 0:  # 仅输出存在缺失的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印列名、缺失数与占比

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', 'CO2', '目标温度', '外界温度']  # 连续变量：用线性插值+前后填充
discrete_vars = ['日龄']  # 离散变量：只用前后填充

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失值数量

        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 双向线性插值填充连续缺失
        data[col] = data[col].fillna(method='ffill')  # 前向填充开头残留缺失
        data[col] = data[col].fillna(method='bfill')  # 后向填充结尾残留缺失

        after_nan = data[col].isna().sum()  # 统计填充后剩余缺失值
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失值数量

        data[col] = data[col].ffill()  # 前向填充离散变量缺失值
        data[col] = data[col].bfill()  # 后向填充离散变量缺失值

        after_nan = data[col].isna().sum()  # 统计填充后剩余缺失值
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征与目标列中剩余缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除仍含缺失值的样本行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后数据规模
print(f"{'=' * 60}\n")  # 打印分隔线并换行结束本阶段

# 保存原始数据副本（用于后续测试集预测）
data_original = data.copy()  # 深拷贝一份处理后的数据备用

print(f"\n【GRU + 小波变换模型】")  # 打印模型标题
print(f"使用特征: {features}")  # 打印实际使用的特征列表
print(f"特征数量: {len(features)}")  # 打印特征个数


# ===================== 小波变换特征提取（仅对温度偏差） =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（温度偏差）进行小波分解，其他列保持原样
    每个时间点只使用历史数据

    Args:
        data: 输入特征 DataFrame，每行一个时间步。
        target_col: 需要进行小波分解的目标列名，默认"温度偏差"。
        wavelet: 小波基函数名称，默认 db4。
        level: 小波分解层数，默认 3。
        window_size: 滑动窗口大小；为 None 时按 2^(level+6) 与数据长度较小者自动确定。
    Returns:
        pd.DataFrame: 含近似/细节系数列（目标列）与其他原始列的特征表。
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印分解开始提示
    print(f"小波基函数: {wavelet}")  # 打印所用小波基函数
    print(f"分解层数: {level}")  # 打印分解层数

    if window_size is None:  # 若未指定窗口大小
        window_size = min(2 ** (level + 6), len(data))  # 按层数估算窗口并夹到数据长度以内

    print(f"滑动窗口大小: {window_size}")  # 打印实际使用的窗口大小

    wavelet_features = pd.DataFrame(index=data.index)  # 创建与输入同索引的空结果表

    for col in data.columns:  # 逐列处理
        if col == 'timestamp':  # 时间戳列不参与分解
            continue  # 跳过该列

        signal = data[col].values  # 取出当前列的数值序列

        if col == target_col:  # 仅对目标列做小波分解
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 存放各时间点的近似系数
            detail_lists = [[] for _ in range(level)]  # 为每层细节系数各建一个空列表

            for i in range(len(signal)):  # 对每个时间点用其历史窗口做分解
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证不越界
                window_signal = signal[start_idx:i + 1]  # 截取截至当前时刻的历史窗口

                if len(window_signal) < window_size:  # 若历史长度不足一个窗口
                    pad_width = window_size - len(window_signal)  # 计算需要在开头补齐的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值向前补齐

                try:
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层小波分解

                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为趋势特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取细节系数末值，空则为 0
                except Exception:
                    approx_list.append(window_signal[-1])  # 分解失败时退化为窗口最后一个原始值
                    for j in range(level):  # 遍历每层
                        detail_lists[j].append(0.0)  # 细节系数填 0 兜底

            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数特征列
            for j in range(level):  # 遍历每一层细节系数
                wavelet_features[f'{col}_detail{j + 1}'] = detail_lists[j]  # 写入第 j+1 层细节特征列
        else:
            print(f"  '{col}' 保持原样（不分解）")  # 打印非目标列不分解
            wavelet_features[col] = signal  # 非目标列原样保留

    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印生成特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印来自目标列的小波特征数
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印保留的原始列数
    return wavelet_features  # 返回构造好的小波特征表


# ===================== 数据集划分（在小波分解之前！） =====================
time_step = 36   # 36步 × 5分钟 = 3小时（原注释保留，输入历史窗口长度）
pred_offset = 6  # 6步 × 5分钟 = 30分钟（原注释保留，预测偏移）

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印输入窗口步数与分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移步数与分钟数

split_index = int(len(data) * 0.8)  # 取前 80% 位置作为按时间顺序的划分点
train_data_raw = data.iloc[:split_index].copy()  # 前 80% 为训练集
test_data_raw = data.iloc[split_index:].copy()  # 后 20% 为测试集

print(f"\n数据集划分:")  # 打印划分结果标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw) / len(data) * 100:.1f}%)")  # 打印训练集样本数与占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw) / len(data) * 100:.1f}%)")  # 打印测试集样本数与占比

print("\n对训练集进行小波分解（仅分解温度偏差）...")  # 提示开始分解训练集
wavelet_window = time_step  # 令小波窗口等于时间窗口，保证只用历史信息
print(f"小波窗口大小设置为: {wavelet_window} 步 (与GRU时间窗口一致)")  # 打印小波窗口设置
train_wavelet_features = wavelet_decompose_sliding(  # 对训练集特征做滑动窗口小波分解
    train_data_raw[features],  # 传入训练集的特征列
    target_col='温度偏差',  # 指定只分解温度偏差
    wavelet='db4',  # 使用 db4 小波基
    level=3,  # 分解 3 层
    window_size=wavelet_window  # 窗口大小与 GRU 时间窗口一致
)

print("\n对测试集进行小波分解（仅分解温度偏差）...")  # 提示开始分解测试集
test_wavelet_features = wavelet_decompose_sliding(  # 对测试集特征做滑动窗口小波分解
    test_data_raw[features],  # 传入测试集的特征列
    target_col='温度偏差',  # 指定只分解温度偏差
    wavelet='db4',  # 使用 db4 小波基
    level=3,  # 分解 3 层
    window_size=wavelet_window  # 窗口大小与 GRU 时间窗口一致
)

def visualize_wavelet_decomposition(original_signal, wavelet_features, feature_name,
                                    save_path='wavelet_decomposition.png'):
    """可视化小波分解结果：原始信号、近似系数与三层细节系数共 5 个子图。

    Args:
        original_signal: 原始信号（一维数组），绘制于首个子图。
        wavelet_features: 含近似/细节系数列的小波特征 DataFrame。
        feature_name: 被分解特征名，用于拼装系数列名与标题。
        save_path: 图片保存路径。
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))  # 创建 5 行 1 列的画布
    axes[0].plot(original_signal, linewidth=1, color='black')  # 在首个子图绘制原始信号
    axes[0].set_title(f'原始信号: {feature_name}', fontsize=12, fontweight='bold')  # 设置原始信号子图标题
    axes[0].set_ylabel('幅值')  # 设置纵轴标签
    axes[0].grid(True, alpha=0.3)  # 显示半透明网格

    axes[1].plot(wavelet_features[f'{feature_name}_approx'].values, linewidth=1, color='blue')  # 绘制近似系数（趋势）
    axes[1].set_title('近似系数 (低频成分 - 趋势)', fontsize=12)  # 设置近似系数子图标题
    axes[1].set_ylabel('幅值')  # 设置纵轴标签
    axes[1].grid(True, alpha=0.3)  # 显示网格

    colors = ['green', 'orange', 'red']  # 三层细节系数分别用绿、橙、红
    for i in range(1, 4):  # 遍历三层细节系数
        axes[i + 1].plot(  # 在第 i+1 个子图绘制第 i 层细节系数
            wavelet_features[f'{feature_name}_detail{i}'].values,  # 取第 i 层细节系数序列
            linewidth=1,  # 线宽 1
            color=colors[i - 1]  # 按层选择颜色
        )
        axes[i + 1].set_title(f'细节系数 Level {i} (高频成分 - 波动)', fontsize=12)  # 设置细节系数子图标题
        axes[i + 1].set_ylabel('幅值')  # 设置纵轴标签
        axes[i + 1].grid(True, alpha=0.3)  # 显示网格

    axes[-1].set_xlabel('时间步', fontsize=11)  # 为最后一个子图设置横轴标签
    plt.tight_layout()  # 自动调整布局防止裁切
    plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 保存小波分解可视化图
    print(f"小波分解可视化已保存: {save_path}")  # 打印保存提示
    plt.close()  # 关闭图形释放内存
first_feature = '温度偏差'  # 指定要可视化的特征为温度偏差
visualize_wavelet_decomposition(  # 调用可视化函数
    train_data_raw[first_feature].values,  # 传入训练集原始温度偏差信号
    train_wavelet_features,  # 传入训练集小波特征
    first_feature,  # 传入特征名
    'wavelet_decomposition_temperature_deviation.png'  # 指定图片保存路径
)

print(f"\n原始特征数: {len(features)}")  # 打印原始特征数量
print(f"小波分解后特征数: {len(train_wavelet_features.columns)}")  # 打印分解后特征数量
print(f"  - 温度偏差的小波特征: 4 个（1个近似系数 + 3个细节系数）")  # 说明小波特征组成
print(f"  - 其他原始特征: {len(features) - 1} 个")  # 说明保留的原始特征数

train_combined_features = train_wavelet_features  # 训练集采用小波特征作为联合特征
test_combined_features = test_wavelet_features  # 测试集采用小波特征作为联合特征
feature_names = list(train_wavelet_features.columns)  # 记录最终特征名列表

print(f"\n使用模式: 温度偏差小波分解 + 其他原始特征")  # 打印建模模式说明
print(f"总特征数: {len(feature_names)}")  # 打印总特征数
print(f"特征列表: {feature_names}")  # 打印特征列表

# ===================== 归一化（只在训练集上fit！） =====================
print("\n归一化处理（避免数据泄漏）...")  # 提示开始归一化，强调避免数据泄漏
feature_scaler = MinMaxScaler()  # 创建特征归一化器
target_scaler = MinMaxScaler()  # 创建目标归一化器

feature_scaler.fit(train_combined_features)  # 仅在训练集特征上拟合
target_scaler.fit(train_data_raw[[target]])  # 仅在训练集目标上拟合

train_combined_features = pd.DataFrame(  # 归一化训练特征并重建为 DataFrame
    feature_scaler.transform(train_combined_features),  # 用训练集归一化器变换训练特征
    columns=feature_names  # 指定列名
)
test_combined_features = pd.DataFrame(  # 归一化测试特征并重建为 DataFrame
    feature_scaler.transform(test_combined_features),  # 用同一归一化器变换测试特征
    columns=feature_names  # 指定列名
)

train_data_raw[target] = target_scaler.transform(train_data_raw[[target]])  # 归一化训练集目标值
test_data_raw[target] = target_scaler.transform(test_data_raw[[target]])  # 归一化测试集目标值

# 用全量数据fit新数据专用scaler（避免新数据超出训练集范围导致预测偏差）
print("\n为新数据预测准备全量归一化器...")  # 提示准备全量归一化器
# 对全量数据做小波分解
all_wavelet_features = wavelet_decompose_sliding(  # 对全量数据做滑动窗口小波分解
    data[features],  # 传入全量数据的特征列
    target_col=target,  # 分解目标列
    wavelet='db4',  # 使用 db4 小波基
    level=3,  # 分解 3 层
    window_size=time_step  # 窗口大小与时间窗口一致
)
new_feature_scaler = MinMaxScaler()  # 创建新数据专用特征归一化器
new_target_scaler = MinMaxScaler()  # 创建新数据专用目标归一化器
new_feature_scaler.fit(all_wavelet_features)  # 在全量小波特征上拟合（供新数据使用）
new_target_scaler.fit(data[[target]])  # 在全量目标值上拟合
print(f"  全量 target_scaler 范围: [{data[target].min():.4f}, {data[target].max():.4f}]")  # 打印全量目标归一化范围


# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):
    """将二维特征表转换为三维时间窗口序列样本。

    Args:
        data: 归一化后的特征 DataFrame，每行一个时间步。
        time_step: 输入序列长度。
    Returns:
        np.ndarray: 形状为 (样本数, time_step, 特征数) 的序列数组。
    """
    sequences = []  # 收集各时间窗口样本
    for i in range(len(data) - time_step - pred_offset + 1):  # 滑动窗口起点，预留偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 取连续 time_step 行作为一个样本
    return np.array(sequences)  # 转换为 numpy 数组返回


print("\n创建训练集序列...")  # 提示创建训练集序列
X_train = create_sequences(train_combined_features, time_step)  # 构造训练集输入序列
y_train = train_data_raw[target].values[time_step + pred_offset - 1:]  # 按偏移取训练集标签
y_train = y_train[:len(X_train)]  # 截断标签与样本数对齐

print("\n创建测试集序列...")  # 提示创建测试集序列
X_test = create_sequences(test_combined_features, time_step)  # 构造测试集输入序列
y_test = test_data_raw[target].values[time_step + pred_offset - 1:]  # 按偏移取测试集标签
y_test = y_test[:len(X_test)]  # 截断标签与样本数对齐

print(f"\n序列数据形状:")  # 打印序列形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集输入与标签形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集输入与标签形状


# ===================== PyTorch模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention（高效通道注意力）
    通过 1D 卷积实现通道注意力，避免降维造成的信息损失。
    Args:
        channels: 输入通道数（即特征维度）。
        gamma: 自适应卷积核尺度公式中的缩放参数。
        b: 自适应卷积核尺度公式中的偏置参数。
    """
    def __init__(self, channels, gamma=2, b=1):
        """按通道数自适应确定卷积核大小并构建池化、卷积与激活模块。"""
        super(ECALayer, self).__init__()  # 调用父类初始化
        t = int(abs((np.log2(channels) + b) / gamma))  # 由通道数计算卷积核尺度 t
        k_size = t if t % 2 else t + 1  # 保证卷积核大小为奇数

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道交互
        self.sigmoid = nn.Sigmoid()  # Sigmoid 把权重映射到 (0,1)

    def forward(self, x):
        """通道注意力前向：置换维度→全局池化→1D卷积→Sigmoid→按通道加权→还原维度。"""
        x_permuted = x.permute(0, 2, 1)  # 由 (batch,time,ch) 置换为 (batch,ch,time)
        y = self.avg_pool(x_permuted)  # 全局平均池化得到通道描述
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 换维做 1D 卷积后再换回
        y = self.sigmoid(y)  # 得到各通道注意力权重
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 权重广播乘回各通道
        return x_permuted.permute(0, 2, 1)  # 还原维度顺序后返回


class GRUWaveletModel(nn.Module):
    """
    GRU + 小波变换 + ECA注意力机制。
    Args:
        input_dim: 输入特征维度（含小波特征）。
        time_step: 时间窗口长度（仅记录）。
        gru_hidden: GRU 隐藏维度，可为整数或每层隐藏维度组成的列表。
        gru_layers: GRU 层数（gru_hidden 为整数时使用）。
        dropout: 每层 GRU 之后的 dropout 比例。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):
        """构建 ECA 注意力、逐层 GRU 栈、Dropout 与全连接输出层。"""
        super(GRUWaveletModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入维度
        self.gru_hidden = gru_hidden  # 保存隐藏维度配置

        self.eca = ECALayer(channels=input_dim)  # ECA 注意力增强输入特征

        # 若 gru_hidden 为列表，则逐层构建 GRU，每层hidden_size可不同
        if isinstance(gru_hidden, list):  # 若隐藏维度以列表形式给出
            hidden_sizes = gru_hidden  # 直接采用该列表作为各层隐藏维度
        else:  # 若给出的是单个整数
            hidden_sizes = [gru_hidden] * gru_layers  # 复制 gru_layers 次得到各层相同隐藏维度

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多层 GRU
        in_size = input_dim  # 当前层的输入维度，初始为输入特征数
        for i, h_size in enumerate(hidden_sizes):  # 逐层构建 GRU
            self.gru_layers_list.append(  # 把构建好的 GRU 加入列表
                nn.GRU(  # 单层 GRU
                    input_size=in_size,  # 该层输入维度
                    hidden_size=h_size,  # 该层隐藏维度
                    num_layers=1,  # 每次只堆一层，多层由循环实现
                    dropout=0.0,  # 层内 dropout 关闭，改用外层独立 Dropout
                    batch_first=True,  # 以 batch 为第一维
                    bidirectional=False  # 单向 GRU
                )
            )
            in_size = h_size  # 下一层输入维度等于当前层隐藏维度

        self.dropout = nn.Dropout(dropout)  # 定义每层 GRU 之后使用的 Dropout
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 输出层：最后一层隐藏维度映射到单值

    def forward(self, x):
        """前向传播：ECA 增强→逐层 GRU+Dropout→取末时刻输出→全连接预测。"""
        x = self.eca(x)  # ECA 通道注意力加权输入
        out = x  # 初始化循环输入为注意力输出
        for gru in self.gru_layers_list:  # 依次通过每层 GRU
            out, _ = gru(out)  # GRU 前向，忽略隐藏状态
            out = self.dropout(out)  # 施加 dropout
        out_last = out[:, -1, :]  # 取最后时间步的输出
        return self.fc(out_last)  # 全连接层输出预测值
import copy  # 重复导入 copy 模块（原文件如此，保留不改）
import numpy as np  # 重复导入 numpy（原文件如此，保留不改）
import torch  # 重复导入 torch（原文件如此，保留不改）
class EarlyStopping:
    """通用早停策略，默认用于最小化验证损失。"""
    def __init__(self, patience=50, min_delta=1e-4, mode='min', restore_best_weights=True, verbose=True):
        """
        Args:
            patience: 连续多少轮没有提升就停止。
            min_delta: 认为"有提升"所需的最小变化量。
            mode: 'min' 表示越小越好（如 loss），'max' 表示越大越好（如 R2）。
            restore_best_weights: 早停后是否恢复到最佳权重。
            verbose: 是否打印日志。
        """
        self.patience = patience  # 保存容忍轮数
        self.min_delta = min_delta  # 保存最小提升阈值
        self.mode = mode  # 保存优化方向
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重
        self.verbose = verbose  # 保存是否打印日志

        self.counter = 0  # 未提升轮数计数器
        self.best_score = None  # 记录最佳分数
        self.best_epoch = 0  # 记录最佳分数所在轮次
        self.early_stop = False  # 早停标志
        self.best_model = None  # 存放最佳模型状态字典

    def _is_improvement(self, score):
        """按 mode 判断当前分数相对最佳是否有足够提升，首个分数视为提升。"""
        if self.best_score is None:  # 尚无基准分数
            return True  # 视为提升

        if self.mode == 'min':  # 最小化模式
            return score < self.best_score - self.min_delta  # 需小于最佳减阈值
        elif self.mode == 'max':  # 最大化模式
            return score > self.best_score + self.min_delta  # 需大于最佳加阈值
        else:  # 非法模式
            raise ValueError("mode must be 'min' or 'max'")  # 抛出异常

    def __call__(self, score, model, epoch=None):
        """每轮用验证分数更新早停状态；有提升则深拷贝保存最佳权重，无效分数按未提升处理。"""
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 分数为空或非有限值
            if self.verbose:  # 若开启日志
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印无效分数警告
            self.counter += 1  # 计数器加一
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 提前返回

        if self._is_improvement(score):  # 若判定为提升
            self.best_score = score  # 更新最佳分数
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 更新最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝保存最佳权重
            self.counter = 0  # 重置计数器
            if self.verbose:  # 若开启日志
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印提升信息
        else:  # 若无提升
            self.counter += 1  # 计数器加一
            if self.verbose:  # 若开启日志
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印进度
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停

    def restore(self, model):
        """若允许且已保存最佳权重，则把最佳权重加载回模型。"""
        if self.restore_best_weights and self.best_model is not None:  # 满足恢复条件
            model.load_state_dict(self.best_model)  # 载入最佳权重
            if self.verbose:  # 若开启日志
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息

def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=20):
    """训练模型：MSE 损失 + Adam 优化，按验证损失早停，返回两阶段损失曲线。"""
    criterion = nn.MSELoss()  # 定义均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 定义 Adam 优化器
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器

    train_losses = []  # 记录每轮训练平均损失
    val_losses = []  # 记录每轮验证平均损失

    for epoch in range(epochs):  # 遍历训练轮次
        model.train()  # 切换为训练模式
        train_loss = 0.0  # 初始化本轮训练损失累计
        for batch_x, batch_y in train_loader:  # 遍历训练批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 批次数据搬到设备

            optimizer.zero_grad()  # 清空梯度
            outputs = model(batch_x)  # 前向预测
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()  # 累计批次损失

        train_loss /= len(train_loader)  # 求本轮训练平均损失
        train_losses.append(train_loss)  # 记录训练损失

        model.eval()  # 切换为评估模式
        val_loss = 0.0  # 初始化本轮验证损失累计
        with torch.no_grad():  # 关闭梯度计算
            for batch_x, batch_y in val_loader:  # 遍历验证批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 批次数据搬到设备
                outputs = model(batch_x)  # 前向预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累计验证损失

        val_loss /= len(val_loader)  # 求本轮验证平均损失
        val_losses.append(val_loss)  # 记录验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印轮次与损失

        early_stopping(val_loss, model)  # 用验证损失更新早停状态
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停轮次
            model.load_state_dict(early_stopping.best_model)  # 恢复最佳权重
            break  # 退出训练循环

    if early_stopping.best_model is not None:  # 若保存过最佳权重
        model.load_state_dict(early_stopping.best_model)  # 训练结束确保加载最佳权重

    return train_losses, val_losses  # 返回训练与验证损失曲线
# ===================== K折交叉验证训练（仅在训练集上） =====================
# 以下为已整体注释掉的 K 折交叉验证实验代码，保留原文不改（当前未启用，不再逐行加注释）
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(feature_names)  # 输入维度等于最终特征数量
#
# print("\n========== 开始5折交叉验证（仅训练集） ==========")
# print(f"交叉验证数据量: {len(X_train)} 样本")
#
# for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
#     print(f"\n{'=' * 50}")
#     print(f"第 {fold}/5 折训练")
#     print(f"{'=' * 50}")
#
#     X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
#     y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
#
#     print(f"训练集样本数: {len(X_train_fold)}")
#     print(f"验证集样本数: {len(X_val_fold)}")
#
#     X_train_fold_tensor = torch.FloatTensor(X_train_fold)
#     y_train_fold_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)
#     X_val_fold_tensor = torch.FloatTensor(X_val_fold)
#     y_val_fold_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)
#
#     fold_train_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
#     fold_val_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)
#     fold_train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True)
#     fold_val_loader = DataLoader(fold_val_dataset, batch_size=32, shuffle=False)
#
#     fold_model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3
#                                  ).to(device)
#
#     if fold == 1:
#         print(f"模型参数量: {sum(p.numel() for p in fold_model.parameters()):,}")
#
#     fold_train_losses, fold_val_losses = train_model(fold_model, fold_train_loader, fold_val_loader,
#                                                      epochs=200, lr=0.001, patience=20)
#     history_list.append({
#         'train_losses': fold_train_losses,
#         'val_losses': fold_val_losses
#     })
#
#     best_val_loss = min(fold_val_losses)
#     val_losses_all.append(best_val_loss)
#     print(f"\n第 {fold} 折最佳验证损失: {best_val_loss:.6f}")
#
# print(f"\n{'=' * 50}")
# print("5折交叉验证结果汇总")
# print(f"{'=' * 50}")
# for i, loss in enumerate(val_losses_all, 1):
#     print(f"第 {i} 折验证损失: {loss:.6f}")
# print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")
# print(f"验证损失标准差: {np.std(val_losses_all):.6f}")
#
# # ===================== 绘制K折交叉验证训练过程曲线 =====================
# plt.figure(figsize=(15, 10))
#
# plt.subplot(2, 2, 1)
# for i, history in enumerate(history_list, 1):
#     plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)
# plt.title('5-Fold Cross Validation - Training Loss', fontsize=14)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 2)
# for i, history in enumerate(history_list, 1):
#     plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)
# plt.title('5-Fold Cross Validation - Validation Loss', fontsize=14)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 3)
# folds = [f'Fold {i}' for i in range(1, 6)]
# plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)
# plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',
#             label=f'Mean: {np.mean(val_losses_all):.6f}')
# plt.title('Best Validation Loss per Fold', fontsize=14)
# plt.xlabel('Fold')
# plt.ylabel('Validation Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3, axis='y')
#
# plt.subplot(2, 2, 4)
# max_epochs = max(len(h['train_losses']) for h in history_list)
# avg_train_losses = []
# avg_val_losses = []
# for epoch in range(max_epochs):
#     train_vals = [h['train_losses'][epoch] for h in history_list if epoch < len(h['train_losses'])]
#     val_vals = [h['val_losses'][epoch] for h in history_list if epoch < len(h['val_losses'])]
#     if train_vals:
#         avg_train_losses.append(np.mean(train_vals))
#     if val_vals:
#         avg_val_losses.append(np.mean(val_vals))
# plt.plot(avg_train_losses, label='Avg Train Loss', linewidth=2)
# plt.plot(avg_val_losses, label='Avg Val Loss', linewidth=2)
# plt.title('5-Fold Average Loss Curves', fontsize=14)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('kfold_training_GRUWaveletECA_pytorch.png', dpi=600, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRUWaveletECA_pytorch.png")

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印训练标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试样本数

X_train_tensor = torch.FloatTensor(X_train)  # 训练集序列转 float 张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练集标签转张量并增加列维
X_test_tensor = torch.FloatTensor(X_test)  # 测试集序列转 float 张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试集标签转张量并增加列维

# 从训练集中分割验证集（80-20），避免使用测试集进行早停
train_val_split = int(len(X_train) * 0.8)  # 取训练集前 80% 作为划分点
X_train_split = X_train_tensor[:train_val_split]  # 前 80% 为实际训练输入
y_train_split = y_train_tensor[:train_val_split]  # 前 80% 为实际训练标签
X_val_split = X_train_tensor[train_val_split:]  # 后 20% 为验证输入
y_val_split = y_train_tensor[train_val_split:]  # 后 20% 为验证标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 封装训练数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 封装验证数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 封装测试数据集

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练加载器，打乱顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证加载器，不打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试加载器，不打乱

input_dim = len(feature_names)  # 重新确认输入维度
final_model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)  # 实例化最终模型并搬到设备

print(final_model)  # 打印模型结构

print("\n开始训练最终模型...")  # 提示开始训练
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印参数总量
print(f"模型大小: {get_model_size(final_model):.2f} MB")  # 打印模型占用大小
print(f"训练集样本数: {len(X_train_split)}")  # 打印训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

training_start_time = time.time()  # 记录训练开始时间

train_losses, val_losses = train_model(  # 调用训练函数训练最终模型
    final_model,  # 待训练模型
    train_loader,  # 训练数据加载器
    val_loader,  # 验证数据加载器
    epochs=200,  # 最大训练轮数 200
    lr=0.001,  # 学习率 0.001
   patience=20  # 早停耐心值 20（原有缩进保留）
)

training_end_time = time.time()  # 记录训练结束时间
total_training_time = training_end_time - training_start_time  # 计算训练总耗时（秒）

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印最小验证损失
print(f"训练总时长: {format_time(total_training_time)}")  # 以自适应格式打印训练时长
print(f"训练总时长（秒）: {total_training_time:.2f}秒")  # 以秒为单位打印训练时长

# ===================== 绘制训练过程曲线 =====================
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  # 导入刻度定位器与格式化器
import matplotlib.pyplot as plt  # 再次导入 pyplot（原文件如此，保留不改）

plt.figure(figsize=(12, 8))   # 高度稍微压低一点（原注释保留），创建画布

plt.plot(train_losses, label='Train Loss', linewidth=4)  # 绘制训练损失曲线
plt.plot(val_losses, label='Validation Loss', linewidth=4)  # 绘制验证损失曲线

plt.title('Model Training - Loss Curves')  # 设置图表标题
plt.xlabel('Epoch')  # 设置横轴标签
plt.ylabel('Loss (MSE)')  # 设置纵轴标签

ax = plt.gca()  # 获取当前坐标轴对象以便细粒度设置

# x轴刻度：每10个epoch显示一个
ax.xaxis.set_major_locator(MultipleLocator(5))  # 设置横轴主刻度间隔为 5

# 关键：把y轴范围拉大，波动就不明显
ax.set_ylim(0, 0.05)  # 固定纵轴范围 0 到 0.05，弱化波动视觉

# y轴刻度间隔
ax.yaxis.set_major_locator(MultipleLocator(0.005))  # 设置纵轴主刻度间隔为 0.005

# y轴显示格式
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 纵轴标签保留三位小数

# 刻度字体
ax.tick_params(axis='both', labelsize=12)  # 设置两轴刻度字号为 12

plt.legend()  # 显示图例
plt.grid(True, alpha=0.2)  # 显示半透明网格
plt.tight_layout()  # 自动调整布局
plt.savefig('training_loss_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')  # 保存训练损失图
plt.close()  # 关闭图形释放内存

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 重建测试加载器（原文件如此，保留）

y_test_normalized = y_test  # 测试标签已是归一化值，直接引用
y_test_rescaled = target_scaler.inverse_transform(y_test_normalized.reshape(-1, 1))  # 反归一化标签为真实值

print(f"\n测试集数据检查:")  # 打印测试集数据检查标题
print(f"X_test shape: {X_test.shape}")  # 打印测试输入形状
print(f"y_test shape: {y_test.shape}")  # 打印测试标签形状
print(f"y_test_rescaled shape: {y_test_rescaled.shape}")  # 打印反归一化后标签形状

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试序列转张量并搬到设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 预测并搬回 CPU 转 numpy

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 反归一化预测值为真实量纲

print(f"y_pred_rescaled shape: {y_pred_rescaled.shape}")  # 打印反归一化后预测形状

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算 MSE
rmse_test = np.sqrt(mse_test)  # 计算 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果（GRU+小波+ECA）")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印测试集 R²
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印测试集 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15x5 画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 展平真实值
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 展平预测值

test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset - 1:]  # 按偏移截取测试集时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳与绘图数据等长

print(f"\n绘图数据检查:")  # 打印绘图数据检查标题
print(f"test_timestamps length: {len(test_timestamps)}")  # 打印时间戳长度
print(f"y_test_rescaled_flat length: {len(y_test_rescaled_flat)}")  # 打印真实值长度
print(f"y_pred_rescaled_flat length: {len(y_pred_rescaled_flat)}")  # 打印预测值长度

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实值曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+小波+ECA预测', linewidth=2)  # 绘制预测值曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置含指标标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格
plt.savefig('test_prediction_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')  # 保存测试预测图
print("测试集预测结果已保存: test_prediction_GRUWavelet_pytorch.png")  # 打印保存提示
plt.close()  # 关闭图形释放内存


# ===================== 模型保存 =====================
torch.save({  # 将模型与配置打包保存为检查点
    'model_state_dict': final_model.state_dict(),  # 保存模型权重
    'input_dim': input_dim,  # 保存输入维度
    'time_step': time_step,  # 保存时间窗口
    'feature_names': feature_names,  # 保存特征名列表
    'wavelet': 'db4',  # 保存小波基名称
    'level': 3,  # 保存分解层数
    'wavelet_window': time_step,  # 保存小波窗口大小
    'wavelet_target_col': '温度偏差',  # 保存被分解的目标列名
    'feature_scaler': feature_scaler,  # 保存特征归一化器
    'target_scaler': target_scaler,  # 保存目标归一化器
    'no_data_leakage': True,  # 标记无数据泄漏
    'original_features': features,  # 保存原始特征列表
}, 'Model_GRUWavelet_no_leakage_pytorch.pth')  # 指定检查点文件路径
print("\nGRU模型已保存: Model_GRUWavelet_no_leakage_pytorch.pth")  # 打印保存提示


# ===================== 新数据预测 =====================
print("\n开始处理新数据")  # 提示开始处理新数据

new_data = None  # 初始化新数据占位变量
for enc in encodings:  # 遍历候选编码
    try:
        new_data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取新数据
        print(f"成功使用 {enc} 编码加载新数据")  # 打印成功编码
        break  # 成功后跳出
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码错误
        continue  # 尝试下一种编码

if new_data is None:  # 若所有编码均失败
    raise ValueError("无法使用任何已知编码读取新数据文件")  # 抛出异常

new_data['timestamp'] = pd.to_datetime(  # 由索引按 5 分钟间隔构造新数据时间戳列
    new_data.index * 5,  # 索引乘以 5 得到分钟偏移
    unit='m',  # 单位为分钟
    origin=pd.Timestamp('2025/6/21 0:04')  # 起始时间基准
)

print("\n对新数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 提示开始分解新数据
new_wavelet_features = wavelet_decompose_sliding(  # 对新数据做滑动窗口小波分解
    new_data[features],  # 传入新数据特征列
    target_col='温度偏差',  # 只分解温度偏差
    wavelet='db4',  # 使用 db4 小波基
    level=3,  # 分解 3 层
    window_size=time_step  # 窗口大小与时间窗口一致
)

new_combined_features = new_wavelet_features  # 采用新数据的小波特征作为联合特征
new_combined_features = pd.DataFrame(  # 用全量归一化器变换并重建为 DataFrame
    new_feature_scaler.transform(new_combined_features),  # 变换新数据小波特征
    columns=feature_names  # 指定列名
)

y_new_original = new_data[target].values  # 保存新数据真实目标值（未归一化）

X_new = create_sequences(new_combined_features, time_step)  # 构造新数据输入序列
y_new = y_new_original[time_step + pred_offset - 1:]  # 按偏移截取新数据标签
X_new = X_new[:len(y_new)]  # 截断序列与标签等长

valid_mask = ~np.isnan(y_new)  # 构造非 NaN 标签的有效掩码
if valid_mask.sum() < len(y_new):  # 若存在 NaN 标签
    print(f"警告: 新数据目标列中有 {(~valid_mask).sum()} 个 NaN，已过滤")  # 打印过滤数量提示

X_new = X_new[valid_mask]  # 用有效掩码过滤输入序列
y_new = y_new[valid_mask]  # 用有效掩码过滤标签

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 新数据序列转张量并搬到设备

print("\nGRU+小波+ECA模型预测新数据...")  # 提示开始预测新数据
final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 预测并搬回 CPU 转 numpy

y_new_pred_rescaled = new_target_scaler.inverse_transform(y_new_pred)  # 用全量目标归一化器反归一化预测
y_new_rescaled = y_new.reshape(-1, 1)  # 真实值重塑为列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MSE
rmse_new = np.sqrt(mse_new)  # 计算新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果（GRU+小波+ECA）")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R²:   {r2_new:.4f}")  # 打印新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset - 1:]  # 按偏移截取新数据时间戳

plt.figure(figsize=(15, 5))  # 创建 15x5 画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 展平新数据真实值
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 展平新数据预测值

plt.plot(  # 绘制新数据真实值曲线
    timestamps[:len(y_new_rescaled_flat)],  # 横轴：截取后的时间戳
    y_new_rescaled_flat,  # 纵轴：真实值
    label='真实温度偏差',  # 图例标签
    linewidth=2,  # 线宽 2
    alpha=0.8  # 透明度 0.8
)
plt.plot(  # 绘制新数据预测值曲线
    timestamps[:len(y_new_pred_rescaled_flat)],  # 横轴：截取后的时间戳
    y_new_pred_rescaled_flat,  # 纵轴：预测值
    label='GRU+小波+ECA预测',  # 图例标签
    linewidth=2,  # 线宽 2
    alpha=0.8  # 透明度 0.8
)
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签
plt.title(f'新数据集温度偏差结果 (R²: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置含指标标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格
plt.savefig('new_data_prediction_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')  # 保存新数据预测图
print("\n新数据预测结果已保存: new_data_prediction_GRUWavelet_pytorch.png")  # 打印保存提示
plt.close()  # 关闭图形释放内存

print("\n========== 所有任务完成（GRU+小波+ECA模型） ==========")  # 打印全部任务完成标志
