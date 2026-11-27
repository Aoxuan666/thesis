"""
GRU + 小波变换 + ECA 通道注意力的 THI（温湿指数）预测训练脚本。

本脚本完成端到端建模流程：读取温湿度 CSV、按时间顺序划分训练/测试集、
在各自子集内部做缺失值填补以避免泄漏、用滑动窗口对 THI 做小波分解提取趋势与波动特征、
只用训练集拟合归一化、构造时间序列样本、构建 GRU+小波+ECA 模型并带早停训练，
最后在原始尺度上评估并绘制预测曲线、损失曲线，可选对外部数据做同样评估。
"""
import pandas as pd  # 导入 pandas 库，用于表格数据读取、清洗与处理（别名 pd）
import numpy as np  # 导入 NumPy 库，用于数组与数值计算（别名 np）
import random  # 导入标准库 random，配合随机种子设置保证可复现（当前种子代码被注释）
import os  # 导入标准库 os，用于设置环境变量 PYTHONHASHSEED
import time  # 导入标准库 time，用于统计训练耗时
import copy  # 导入标准库 copy，用于深拷贝模型权重实现早停

# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用 Agg 非交互式后端，允许在无界面环境保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口（别名 plt）
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  # 导入刻度定位器与格式化器，用于美化损失曲线坐标轴

from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于归一化特征与目标值
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R²、均方误差、平均绝对误差

import torch  # 导入 PyTorch 主库，用于张量计算与模型训练
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU、全连接层等
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载工具，按批次组织训练/验证/测试数据

import pywt  # 小波变换库，用于 pywt.wavedec 对 THI 做多尺度分解


# ===================== 可选：随机种子 =====================
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


# set_seed(10)


# ===================== 辅助函数 =====================
def get_model_size(model):  # 计算模型参数与缓冲区占用的内存大小（MB）
    """
    计算模型大小（MB）。

    参数:
        model: PyTorch 模型，遍历其 parameters() 与 buffers() 统计字节数。
    返回:
        float: 参数与缓冲区所占内存换算为 MB 后的大小。
    """
    param_size = 0  # 初始化参数字节数累计变量
    for param in model.parameters():  # 遍历模型所有可学习参数
        param_size += param.nelement() * param.element_size()  # 累加该参数元素个数 × 单元素字节数

    buffer_size = 0  # 初始化缓冲区（如 BatchNorm 统计量）字节数累计变量
    for buffer in model.buffers():  # 遍历模型所有缓冲区
        buffer_size += buffer.nelement() * buffer.element_size()  # 累加该缓冲区元素个数 × 单元素字节数

    size_all_mb = (param_size + buffer_size) / 1024 ** 2  # 总字节数换算为 MB（1 MB = 1024² 字节）
    return size_all_mb  # 返回模型大小（MB）


def format_time(seconds):  # 将秒数格式化为易读的秒/分/小时字符串
    """
    将秒数格式化为易读的时间字符串。

    参数:
        seconds: 以秒为单位的时长。
    返回:
        str: 小于 60 秒显示“秒”，小于 3600 秒显示“分钟”，否则显示“小时”，均保留两位小数。
    """
    if seconds < 60:  # 不足 1 分钟，直接以秒显示
        return f"{seconds:.2f}秒"  # 返回秒数字符串（保留两位小数）
    elif seconds < 3600:  # 不足 1 小时，以分钟显示
        minutes = seconds / 60  # 秒换算为分钟
        return f"{minutes:.2f}分钟"  # 返回分钟字符串
    else:  # 1 小时及以上，以小时显示
        hours = seconds / 3600  # 秒换算为小时
        return f"{hours:.2f}小时"  # 返回小时字符串


def load_csv_with_multiple_encodings(file_path, encodings=None):  # 依次尝试多种编码读取 CSV，直到成功
    """
    尝试多种编码读取 CSV，避免因中文编码不一致导致读取失败。

    参数:
        file_path: CSV 文件路径。
        encodings: 备选编码列表，为 None 时使用默认的 utf-8/gbk/gb2312/gb18030/latin1。
    返回:
        DataFrame: 成功读取的数据表。
    异常:
        ValueError: 所有编码均解码失败时抛出。
    """
    if encodings is None:  # 调用方未指定编码列表时使用默认列表
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 默认依次尝试的常见编码

    data = None  # 初始化读取结果变量
    for enc in encodings:  # 逐个编码尝试
        try:  # 捕获解码异常
            data = pd.read_csv(file_path, encoding=enc)  # 用当前编码读取 CSV
            print(f"成功使用 {enc} 编码加载文件: {file_path}")  # 打印成功使用的编码
            return data  # 读取成功即返回数据
        except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
            continue  # 换下一个编码继续尝试

    raise ValueError(f"无法使用任何已知编码读取文件: {file_path}")  # 所有编码均失败则抛出异常


def fill_missing_values(df, features, target, dataset_name="数据集"):  # 对当前数据子集单独做缺失值处理，避免全量处理后切分造成泄漏
    """
    分别对当前子集做缺失值处理，避免在全量数据上先处理再切分。

    参数:
        df: 当前子集的 DataFrame。
        features: 特征列名列表。
        target: 目标列名。
        dataset_name: 数据集名称，仅用于打印提示。
    返回:
        DataFrame: 缺失值处理完成的数据副本。
    """
    df = df.copy()  # 复制数据，避免修改原始 DataFrame
    print(f"\n{'=' * 60}")  # 打印分隔线
    print(f"{dataset_name}缺失值处理")  # 打印当前数据集名称与处理标题
    print(f"{'=' * 60}")  # 再次打印分隔线
    print(f"原始数据形状: {df.shape}")  # 打印处理前数据行列数
    print(f"\n各列缺失值统计:")  # 打印缺失值统计表头
    for col in features + [target]:  # 遍历特征列与目标列
        if col in df.columns:  # 仅处理数据中实际存在的列
            nan_count = df[col].isna().sum()  # 统计该列缺失值个数
            nan_pct = nan_count / len(df) * 100  # 计算缺失值占比（百分比）
            if nan_count > 0:  # 仅打印存在缺失值的列
                print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与占比

    # 定义连续变量和离散变量（两类采用不同填充策略）
    continuous_vars = ['THI', '当前温度', '目标温度']  # 连续变量：线性插值 + 前后填充
    discrete_vars = ['日龄']  # 离散变量：仅前后填充

    # 连续变量：仅在当前子集内部插值 + 前后填充
    for col in continuous_vars:  # 遍历连续变量
        if col in df.columns and df[col].isna().sum() > 0:  # 该列存在且有缺失时才处理
            print(f"\n处理 {col}:")  # 打印正在处理的列名
            before_nan = df[col].isna().sum()  # 记录处理前缺失值个数

            df[col] = df[col].interpolate(method='linear', limit_direction='both')  # 线性插值，双向填充内部缺失
            df[col] = df[col].ffill()  # 前向填充开头残留缺失
            df[col] = df[col].bfill()  # 后向填充结尾残留缺失

            after_nan = df[col].isna().sum()  # 记录处理后缺失值个数
            print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前缺失数
            print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后缺失数

    # 离散变量：仅在当前子集内部前后填充
    for col in discrete_vars:  # 遍历离散变量
        if col in df.columns and df[col].isna().sum() > 0:  # 该列存在且有缺失时才处理
            print(f"\n处理 {col}:")  # 打印正在处理的列名
            before_nan = df[col].isna().sum()  # 记录填充前缺失值个数

            df[col] = df[col].ffill()  # 前向填充
            df[col] = df[col].bfill()  # 后向填充

            after_nan = df[col].isna().sum()  # 记录填充后缺失值个数
            print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
            print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

    remaining_nan = df[features + [target]].isna().sum().sum()  # 统计特征列与目标列剩余缺失总数
    if remaining_nan > 0:  # 若仍有缺失值
        print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
        df = df.dropna(subset=features + [target])  # 删除特征/目标中含缺失值的行

    print(f"\n处理后数据形状: {df.shape}")  # 打印处理后数据行列数
    print(f"{'=' * 60}\n")  # 打印结束分隔线
    return df  # 返回缺失值处理完成的数据


def wavelet_decompose_sliding(data, target_col='THI', wavelet='db4', level=3, window_size=None):  # 滑动窗口小波分解：只分解目标列，避免未来信息泄漏
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏。
    仅对指定列（THI）进行小波分解，其他列保持原样，每个时间点只使用历史数据。

    参数:
        data: 待分解的 DataFrame。
        target_col: 需要分解的目标列名（默认 'THI'）。
        wavelet: 小波基函数名称（默认 'db4'）。
        level: 分解层数（默认 3）。
        window_size: 滑动窗口长度，为 None 时按 2^(level+6) 与数据长度取较小值。
    返回:
        DataFrame: 包含近似系数、各层细节系数及未分解原始列的特征表。
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印分解开始提示与目标列
    print(f"小波基函数: {wavelet}")  # 打印小波基函数
    print(f"分解层数: {level}")  # 打印分解层数

    if window_size is None:  # 未指定窗口大小时自动计算
        window_size = min(2 ** (level + 6), len(data))  # 默认窗口 = 2^(层数+6) 与数据长度取较小值

    print(f"滑动窗口大小: {window_size}")  # 打印实际使用的窗口大小

    wavelet_features = pd.DataFrame(index=data.index)  # 创建空结果表，索引与原数据一致

    for col in data.columns:  # 遍历每一列
        signal = data[col].values  # 取该列的一维数值数组

        if col == target_col:  # 仅对目标列做小波分解
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列
            approx_list = []  # 存放近似系数序列
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数建立列表

            for i in range(len(signal)):  # 逐时间点滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，不越过数组开头
                window_signal = signal[start_idx:i + 1]  # 截取截至当前点、长度不超过窗口的历史信号

                if len(window_signal) < window_size:  # 序列开头的窗口不足指定长度
                    pad_width = window_size - len(window_signal)  # 计算需要补齐的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 在开头用边缘值补齐

                try:  # 尝试小波分解，异常时回退
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层小波分解
                    approx_list.append(coeffs[0][-1])  # 取最后一层近似系数作为趋势特征

                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层最后一个细节系数（为空则补 0）
                except Exception:  # 分解异常时回退
                    approx_list.append(window_signal[-1])  # 近似系数退化为窗口最后一个值
                    for j in range(level):  # 遍历各层
                        detail_lists[j].append(0.0)  # 细节系数补 0

            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数特征列
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j + 1}'] = detail_lists[j]  # 写入每层细节系数特征列
        else:  # 非目标列不做分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印保持原样提示
            wavelet_features[col] = signal  # 原样保留该列

    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印生成的特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印小波特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印保留的原始特征数量
    return wavelet_features  # 返回小波特征表


def visualize_wavelet_decomposition(original_signal, wavelet_features, feature_name,  # 可视化小波分解结果（原始信号 + 近似 + 各层细节）
                                    save_path='wavelet_decomposition.png'):  # 参数 save_path 指定图片保存路径
    """
    可视化小波分解结果：绘制原始信号、近似系数与三层细节系数。

    参数:
        original_signal: 原始信号的一维数组。
        wavelet_features: 小波分解得到的特征表。
        feature_name: 特征名，用于拼装列名与标题。
        save_path: 图片保存路径。
    返回:
        None，图片直接保存到文件。
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))  # 创建 5 行 1 列子图画布

    axes[0].plot(original_signal, linewidth=1, color='black')  # 第 1 个子图：绘制原始信号（黑色）
    axes[0].set_title(f'原始信号: {feature_name}', fontsize=12, fontweight='bold')  # 设置原始信号子图标题
    axes[0].set_ylabel('幅值')  # 设置纵轴标签
    axes[0].grid(True, alpha=0.3)  # 开启网格线

    axes[1].plot(wavelet_features[f'{feature_name}_approx'].values, linewidth=1, color='blue')  # 第 2 个子图：绘制近似系数（蓝色）
    axes[1].set_title('近似系数 (低频成分 - 趋势)', fontsize=12)  # 设置近似系数子图标题
    axes[1].set_ylabel('幅值')  # 设置纵轴标签
    axes[1].grid(True, alpha=0.3)  # 开启网格线

    colors = ['green', 'orange', 'red']  # 三层细节系数的绘图颜色
    for i in range(1, 4):  # 遍历 3 个细节层
        axes[i + 1].plot(  # 在对应子图绘制该层细节系数
            wavelet_features[f'{feature_name}_detail{i}'].values,  # 取该层细节系数数值
            linewidth=1,  # 设置线宽
            color=colors[i - 1]  # 设置颜色
        )
        axes[i + 1].set_title(f'细节系数 Level {i} (高频成分 - 波动)', fontsize=12)  # 设置细节系数子图标题
        axes[i + 1].set_ylabel('幅值')  # 设置纵轴标签
        axes[i + 1].grid(True, alpha=0.3)  # 开启网格线

    axes[-1].set_xlabel('时间步', fontsize=11)  # 最底部子图设置横轴标签
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存图片，150 dpi
    print(f"小波分解可视化已保存: {save_path}")  # 打印保存路径
    plt.close()  # 关闭画布释放内存


def create_sequences(data, time_step, pred_offset):  # 构造输入序列：用 [i, i+time_step-1] 预测 i+time_step+pred_offset-1
    """
    生成输入序列。

    输入窗口为 [i, i+time_step-1]，预测目标位置为 i+time_step+pred_offset-1。
    参数:
        data: 特征表（行代表时间步）。
        time_step: 输入窗口长度。
        pred_offset: 预测目标相对窗口末尾的偏移。
    返回:
        ndarray: 形状为 (样本数, time_step, 特征数) 的三维数组。
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset + 1):  # 滑动起点，保证目标位置不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行作为一个样本
    return np.array(sequences)  # 转换为三维数组返回


def create_targets(target_array, time_step, pred_offset):  # 生成与 create_sequences 严格对齐的目标值
    """
    与 create_sequences 严格对齐的目标值。

    参数:
        target_array: 一维目标数组。
        time_step: 输入窗口长度。
        pred_offset: 预测偏移。
    返回:
        ndarray: 从第 time_step+pred_offset-1 个位置开始的目标数组。
    """
    return target_array[time_step + pred_offset - 1:]  # 从预测目标位置开始截取


def evaluate_and_plot(model, X_data, y_true_raw, target_scaler, timestamps,  # 在原始尺度上评估模型并绘制预测对比图
                      dataset_name='测试集', plot_path='prediction.png'):  # 参数：数据集名称与图片保存路径
    """
    评估并绘图。

    y_true_raw: 这里传入“真实原始尺度”的目标值（未归一化），与反归一化后的预测值在同一尺度比较。
    参数:
        model: 待评估模型。
        X_data: 输入序列数组。
        y_true_raw: 原始尺度的真实目标值。
        target_scaler: 目标归一化器，用于反归一化预测值。
        timestamps: 与样本对齐的时间戳，用于绘图横轴。
        dataset_name: 数据集名称，用于日志与标题。
        plot_path: 预测图保存路径。
    返回:
        dict: 包含 r2/rmse/mae 及真实值、预测值数组的字典。
    """
    model.eval()  # 切换模型到评估模式（关闭 dropout）
    X_tensor = torch.FloatTensor(X_data).to(device)  # 输入数据转张量并送到设备

    with torch.no_grad():  # 关闭梯度计算
        y_pred_norm = model(X_tensor).cpu().numpy()  # 前向预测并转回 NumPy（归一化尺度）

    y_pred_rescaled = target_scaler.inverse_transform(y_pred_norm)  # 预测值反归一化到原始尺度
    y_true_rescaled = y_true_raw.reshape(-1, 1)  # 真实值调整为列向量

    r2 = r2_score(y_true_rescaled, y_pred_rescaled)  # 计算决定系数 R²
    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)  # 计算均方误差
    rmse = np.sqrt(mse)  # 均方误差开方得 RMSE
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)  # 计算平均绝对误差

    print(f"\n{'=' * 60}")  # 打印分隔线
    print(f"{dataset_name}评估结果（GRU+小波+ECA）")  # 打印评估标题
    print(f"{'=' * 60}")  # 打印分隔线
    print(f"{dataset_name} R2:   {r2:.4f}")  # 打印 R²
    print(f"{dataset_name} RMSE: {rmse:.4f}")  # 打印 RMSE
    print(f"{dataset_name} MAE:  {mae:.4f}")  # 打印 MAE
    print(f"{'=' * 60}")  # 打印分隔线

    y_true_flat = y_true_rescaled.flatten()  # 真实值拉平为一维
    y_pred_flat = y_pred_rescaled.flatten()  # 预测值拉平为一维

    plt.figure(figsize=(15, 5))  # 创建绘图画布
    plt.plot(timestamps[:len(y_true_flat)], y_true_flat, label=f'真实THI', linewidth=2)  # 绘制真实 THI 曲线
    plt.plot(timestamps[:len(y_pred_flat)], y_pred_flat, label=f'GRU+小波+ECA预测', linewidth=2)  # 绘制预测 THI 曲线
    plt.xlabel('时间')  # 设置横轴标签
    plt.ylabel('THI')  # 设置纵轴标签
    plt.title(f'{dataset_name}THI结果 (R2: {r2:.4f}, RMSE: {rmse:.4f})')  # 设置标题并附带指标
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 开启网格
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')  # 保存图片
    plt.close()  # 关闭画布

    print(f"{dataset_name}预测结果已保存: {plot_path}")  # 打印保存提示

    return {  # 返回评估指标与结果数组组成的字典
        'r2': r2,  # 决定系数
        'rmse': rmse,  # 均方根误差
        'mae': mae,  # 平均绝对误差
        'y_true': y_true_rescaled,  # 真实值（原始尺度，列向量）
        'y_pred': y_pred_rescaled  # 预测值（原始尺度，列向量）
    }


# ===================== PyTorch模型定义 =====================
class ECALayer(nn.Module):  # 高效通道注意力模块（ECA-Net）
    """
    ECA-Net: Efficient Channel Attention
    通过一维卷积实现无需降维的高效通道注意力。

    参数:
        channels: 输入通道数。
        gamma: 用于自适应计算卷积核大小的缩放系数（默认 2）。
        b: 卷积核大小计算的偏置项（默认 1）。
    前向返回:
        与输入同形状、各通道按注意力权重加权后的张量。
    """
    def __init__(self, channels, gamma=2, b=1):  # 初始化：channels 为输入通道数
        super(ECALayer, self).__init__()  # 调用父类构造函数
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应计算卷积核参数 t
        k_size = t if t % 2 else t + 1  # 保证卷积核大小为奇数
        k_size = max(k_size, 1)  # 至少为 1，避免非法核大小

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化：每通道压缩为 1 个值
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,  # 一维卷积跨通道学习注意力权重
                              padding=(k_size - 1) // 2, bias=False)  # 两侧补零保持长度，不使用偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 将权重映射到 0~1

    def forward(self, x):  # 前向传播：对通道做注意力加权
        # x: [batch, seq_len, channels]
        x_permuted = x.permute(0, 2, 1)            # [batch, channels, seq_len]，转置使通道维位于第 1 维
        y = self.avg_pool(x_permuted)              # [batch, channels, 1]，全局平均池化得到通道描述
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 一维卷积后转回原形状，得到通道注意力权重
        y = self.sigmoid(y)  # 归一化权重到 0~1
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将权重广播乘回各通道
        return x_permuted.permute(0, 2, 1)         # [batch, seq_len, channels]，转回原维度后返回


class GRUWaveletModel(nn.Module):  # GRU + 小波特征 + ECA 注意力的预测模型
    """
    GRU + 小波变换 + ECA注意力机制。

    参数:
        input_dim: 输入特征维度。
        time_step: 时间窗口长度。
        gru_hidden: GRU 隐藏维度，可为整数（各层相同）或列表（逐层不同）。
        gru_layers: GRU 层数（gru_hidden 为整数时生效）。
        dropout: GRU 之后的丢弃率。
    前向返回:
        (batch, 1) 的预测张量。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):  # 初始化模型结构与超参数
        super(GRUWaveletModel, self).__init__()  # 调用父类构造函数
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层维度

        self.eca = ECALayer(channels=input_dim)  # 构建 ECA 注意力层

        # 若gru_hidden为列表，则逐层构建GRU，每层hidden_size可不同
        if isinstance(gru_hidden, list):  # 若隐藏维度以列表给出
            hidden_sizes = gru_hidden  # 直接采用列表配置
        else:  # 否则所有层使用相同隐藏维度
            hidden_sizes = [gru_hidden] * gru_layers  # 复制 gru_layers 份

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多层 GRU
        in_size = input_dim  # 第一层输入维度等于特征数
        for h_size in hidden_sizes:  # 逐层构建 GRU
            self.gru_layers_list.append(  # 追加一层 GRU
                nn.GRU(  # 单层 GRU 定义
                    input_size=in_size,  # 输入维度
                    hidden_size=h_size,  # 隐藏维度
                    num_layers=1,  # 每层 GRU 只含一个单元
                    dropout=0.0,  # 层内 dropout 关闭，由外部统一控制
                    batch_first=True,  # 输入形状为 (batch, seq, feature)
                    bidirectional=False  # 单向 GRU
                )
            )
            in_size = h_size  # 下一层输入维度等于本层输出维度

        self.dropout = nn.Dropout(dropout)  # 定义 dropout 层
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 全连接层将最后隐藏状态映射为 1 个预测值

    def forward(self, x):  # 前向传播：ECA 加权 -> 多层 GRU -> 取末时刻 -> 全连接
        x = self.eca(x)  # ECA 注意力增强输入
        out = x  # 初始化 GRU 输入
        for gru in self.gru_layers_list:  # 逐层通过 GRU
            out, _ = gru(out)  # 取序列输出，忽略隐藏状态
            out = self.dropout(out)  # 施加 dropout

        out_last = out[:, -1, :]  # 取序列最后一个时间步的输出
        return self.fc(out_last)  # 经全连接层得到预测


class EarlyStopping:  # 早停策略：验证损失长期不降则停止训练并保留最佳权重
    """早停策略。"""
    def __init__(self, patience=10, min_delta=0):  # 初始化：patience 为容忍轮数，min_delta 为最小改善量
        self.patience = patience  # 保存容忍轮数
        self.min_delta = min_delta  # 保存最小改善阈值
        self.counter = 0  # 未改善计数器
        self.best_loss = None  # 记录最佳验证损失
        self.early_stop = False  # 是否触发早停标志
        self.best_model = None  # 保存最佳模型权重副本

    def __call__(self, val_loss, model):  # 每轮验证后调用：更新最佳模型或累计未改善计数
        if self.best_loss is None:  # 首轮验证，直接记录为最佳
            self.best_loss = val_loss  # 初始化最佳损失
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝当前模型权重
        elif val_loss > self.best_loss - self.min_delta:  # 本轮损失未优于最佳（考虑最小改善量）
            self.counter += 1  # 未改善计数器加一
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
        else:  # 本轮有改善
            self.best_loss = val_loss  # 更新最佳损失
            self.best_model = copy.deepcopy(model.state_dict())  # 更新最佳权重
            self.counter = 0  # 重置计数器


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 训练模型：带验证与早停，返回训练/验证损失曲线
    """
    训练模型。

    参数:
        model: 待训练模型。
        train_loader / val_loader: 训练与验证数据加载器。
        epochs: 最大训练轮数。
        lr: Adam 优化器学习率。
        patience: 早停容忍轮数。
    返回:
        (train_losses, val_losses): 每轮训练与验证平均损失列表。
    """
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam 优化器
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器

    train_losses = []  # 记录每轮训练损失
    val_losses = []  # 记录每轮验证损失

    for epoch in range(epochs):  # 按轮次训练
        model.train()  # 切换训练模式
        train_loss = 0.0  # 本轮累计训练损失

        for batch_x, batch_y in train_loader:  # 遍历训练批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 批次数据送到设备

            optimizer.zero_grad()  # 清零上一轮梯度
            outputs = model(batch_x)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()  # 累加批次损失

        train_loss /= len(train_loader)  # 求训练损失均值
        train_losses.append(train_loss)  # 记录本轮训练损失

        model.eval()  # 切换评估模式
        val_loss = 0.0  # 本轮累计验证损失
        with torch.no_grad():  # 关闭梯度计算
            for batch_x, batch_y in val_loader:  # 遍历验证批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 批次数据送到设备
                outputs = model(batch_x)  # 前向预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加批次损失

        val_loss /= len(val_loader)  # 求验证损失均值
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次日志
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前轮次与损失

        early_stopping(val_loss, model)  # 调用早停检查
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印触发轮次
            model.load_state_dict(early_stopping.best_model)  # 恢复最佳权重
            break  # 跳出训练循环

    if early_stopping.best_model is not None:  # 训练结束后再次加载最佳权重
        model.load_state_dict(early_stopping.best_model)  # 确保最终模型为最优状态

    return train_losses, val_losses  # 返回损失记录


# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU
print(f"使用设备: {device}")  # 打印使用的设备
if torch.cuda.is_available():  # 若有 GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印 GPU 型号
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印 GPU 数量

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ===================== 参数配置 =====================
TRAIN_FILE = '温湿度数据.csv'  # 训练数据文件名
EXTERNAL_FILE = '部分温湿度数据.csv'  # 外部评估数据文件名

# 如果“部分温湿度数据.csv”和测试集本质是同一段数据，这里建议改为False
EVALUATE_EXTERNAL_DATA = True  # 是否评估外部数据（若与测试集同源可设为 False）

time_step = 36    # 36步 × 5分钟 = 3小时，GRU 输入窗口长度
pred_offset = 6   # 6步 × 5分钟 = 30分钟，预测目标相对窗口末尾的偏移

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印 GRU 窗口对应时长
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移对应时长


# ===================== 数据加载 =====================
data = load_csv_with_multiple_encodings(TRAIN_FILE)  # 读取训练数据（多编码尝试）
data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按起始时间与 5 分钟频率生成时间戳列

features = ['日龄', 'THI', '当前温度', '目标温度']  # 候选特征列表
target = 'THI'  # 目标变量为 THI
features = [f for f in features if f in data.columns]  # 过滤掉数据中不存在的特征

print(f"\n【GRU + 小波变换模型】")  # 打印模型标题
print(f"使用特征: {features}")  # 打印实际使用的特征
print(f"特征数量: {len(features)}")  # 打印特征数量


# ===================== 先按时间划分数据集，再分别填补缺失值 =====================
split_index = int(len(data) * 0.8)  # 按 8:2 划分，计算切分位置
train_data_raw = data.iloc[:split_index].copy()  # 前 80% 作为训练集原始副本
test_data_raw = data.iloc[split_index:].copy()  # 后 20% 作为测试集原始副本

print(f"\n数据集划分:")  # 打印划分结果标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw) / len(data) * 100:.1f}%)")  # 打印训练集样本数与占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw) / len(data) * 100:.1f}%)")  # 打印测试集样本数与占比

train_data_raw = fill_missing_values(train_data_raw, features, target, dataset_name="训练集")  # 训练集内部缺失值处理
test_data_raw = fill_missing_values(test_data_raw, features, target, dataset_name="测试集")  # 测试集内部缺失值处理


# ===================== 小波分解（分别在训练集和测试集内部做） =====================
print("\n对训练集进行小波分解（仅分解THI）...")  # 提示开始训练集小波分解
wavelet_window = time_step  # 小波窗口与 GRU 时间窗口保持一致
print(f"小波窗口大小设置为: {wavelet_window} 步 (与GRU时间窗口一致)")  # 打印小波窗口大小

train_wavelet_features = wavelet_decompose_sliding(  # 对训练集特征做滑动窗口小波分解
    train_data_raw[features],  # 输入训练集特征列
    target_col='THI',  # 仅分解 THI
    wavelet='db4',  # 使用 db4 小波基
    level=3,  # 分解 3 层
    window_size=wavelet_window  # 滑动窗口大小
)

print("\n对测试集进行小波分解（仅分解THI）...")  # 提示开始测试集小波分解
test_wavelet_features = wavelet_decompose_sliding(  # 对测试集特征做滑动窗口小波分解
    test_data_raw[features],  # 输入测试集特征列
    target_col='THI',  # 仅分解 THI
    wavelet='db4',  # 使用 db4 小波基
    level=3,  # 分解 3 层
    window_size=wavelet_window  # 滑动窗口大小
)

first_feature = 'THI'  # 可视化选用的特征（THI）
visualize_wavelet_decomposition(  # 绘制并保存小波分解可视化图
    train_data_raw[first_feature].values,  # 原始信号
    train_wavelet_features,  # 小波特征表
    first_feature,  # 特征名
    'wavelet_decomposition_temperature_deviation.png'  # 图片保存路径
)

print(f"\n原始特征数: {len(features)}")  # 打印原始特征数
print(f"小波分解后特征数: {len(train_wavelet_features.columns)}")  # 打印分解后特征数
print(f"  - THI的小波特征: 4 个（1个近似系数 + 3个细节系数）")  # 打印 THI 小波特征数量
print(f"  - 其他原始特征: {len(features) - 1} 个")  # 打印保留的原始特征数量

train_combined_features = train_wavelet_features.copy()  # 训练集组合特征（小波特征 + 原始特征）
test_combined_features = test_wavelet_features.copy()  # 测试集组合特征
feature_names = list(train_combined_features.columns)  # 记录特征名顺序，供归一化使用

print(f"\n使用模式: THI小波分解 + 其他原始特征")  # 打印特征组合模式
print(f"总特征数: {len(feature_names)}")  # 打印总特征数
print(f"特征列表: {feature_names}")  # 打印特征名列表


# ===================== 归一化（只在训练集上fit） =====================
print("\n归一化处理（避免数据泄漏）...")  # 提示开始归一化
feature_scaler = MinMaxScaler()  # 特征归一化器
target_scaler = MinMaxScaler()  # 目标归一化器

feature_scaler.fit(train_combined_features)  # 仅在训练集上拟合特征归一化器
target_scaler.fit(train_data_raw[[target]])  # 仅在训练集上拟合目标归一化器

train_combined_features = pd.DataFrame(  # 变换训练集特征
    feature_scaler.transform(train_combined_features),  # 归一化数值
    columns=feature_names,  # 保持列名
    index=train_combined_features.index  # 保持索引
)

test_combined_features = pd.DataFrame(  # 变换测试集特征
    feature_scaler.transform(test_combined_features),  # 用训练集归一化器变换
    columns=feature_names,  # 保持列名
    index=test_combined_features.index  # 保持索引
)

# 单独保留“原始尺度”的目标值用于最终评估
y_train_raw_original = train_data_raw[target].values.copy()  # 训练集目标原始值副本
y_test_raw_original = test_data_raw[target].values.copy()  # 测试集目标原始值副本

# 训练用目标转成归一化
train_target_scaled = target_scaler.transform(train_data_raw[[target]]).flatten()  # 训练集目标归一化并拉平
test_target_scaled = target_scaler.transform(test_data_raw[[target]]).flatten()  # 测试集目标归一化并拉平


# ===================== 构造序列数据 =====================
print("\n创建训练集序列...")  # 提示创建训练集序列
X_train = create_sequences(train_combined_features, time_step, pred_offset)  # 构造训练集输入序列
y_train = create_targets(train_target_scaled, time_step, pred_offset)  # 构造训练集归一化目标

print("\n创建测试集序列...")  # 提示创建测试集序列
X_test = create_sequences(test_combined_features, time_step, pred_offset)  # 构造测试集输入序列
y_test = create_targets(test_target_scaled, time_step, pred_offset)  # 构造测试集归一化目标

# 与原始尺度目标严格对齐（用于评估）
y_train_raw_eval = create_targets(y_train_raw_original, time_step, pred_offset)  # 训练集原始尺度目标（评估用）
y_test_raw_eval = create_targets(y_test_raw_original, time_step, pred_offset)  # 测试集原始尺度目标（评估用）

print(f"\n序列数据形状:")  # 打印序列形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集序列与目标形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集序列与目标形状


# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印训练标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试样本数

X_train_tensor = torch.FloatTensor(X_train)  # 训练集输入转张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练目标转张量并增加特征维

X_test_tensor = torch.FloatTensor(X_test)  # 测试集输入转张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试目标转张量并增加特征维

# 从训练集中分割验证集（按时间顺序）
# 可选：留一个gap，减少训练集和验证集在滑窗边界上的高度重叠
train_val_split = int(len(X_train) * 0.8)  # 训练集再按 8:2 切出验证集
gap = time_step + pred_offset - 1  # 间隔长度，避免滑窗样本重叠

if train_val_split + gap < len(X_train):  # 样本足够时留出 gap
    X_train_split = X_train_tensor[:train_val_split]  # 实际训练输入
    y_train_split = y_train_tensor[:train_val_split]  # 实际训练目标

    X_val_split = X_train_tensor[train_val_split + gap:]  # 验证输入（跳过 gap）
    y_val_split = y_train_tensor[train_val_split + gap:]  # 验证目标
else:  # 样本不足，不留 gap
    print("\n警告: 样本量不足以设置gap，验证集将直接从分割点后开始。")  # 打印警告
    X_train_split = X_train_tensor[:train_val_split]  # 训练输入
    y_train_split = y_train_tensor[:train_val_split]  # 训练目标

    X_val_split = X_train_tensor[train_val_split:]  # 验证输入
    y_val_split = y_train_tensor[train_val_split:]  # 验证目标

train_dataset = TensorDataset(X_train_split, y_train_split)  # 封装训练数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 封装验证数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 封装测试数据集

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练数据加载器，打乱顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试数据加载器

input_dim = len(feature_names)  # 输入维度等于特征数
final_model = GRUWaveletModel(  # 构建最终模型
    input_dim=input_dim,  # 输入特征维度
    time_step=time_step,  # 时间窗口
    gru_hidden=128,  # GRU 隐藏维度
    gru_layers=1,  # GRU 层数
    dropout=0.2  # dropout 比例
).to(device)  # 将模型移至设备

print(final_model)  # 打印模型结构
print("\n开始训练最终模型...")  # 打印训练开始提示
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印参数总量
print(f"模型大小: {get_model_size(final_model):.2f} MB")  # 打印模型大小
print(f"训练集样本数: {len(X_train_split)}")  # 打印训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

training_start_time = time.time()  # 记录训练开始时间

train_losses, val_losses = train_model(  # 调用训练函数
    final_model,  # 待训练模型
    train_loader,  # 训练数据加载器
    val_loader,  # 验证数据加载器
    epochs=200,  # 最大训练轮数
    lr=0.001,  # 学习率
    patience=10  # 早停容忍轮数
)

training_end_time = time.time()  # 记录训练结束时间
total_training_time = training_end_time - training_start_time  # 计算总耗时（秒）

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印最小验证损失
print(f"训练总时长: {format_time(total_training_time)}")  # 打印格式化后的训练时长
print(f"训练总时长（秒）: {total_training_time:.2f}秒")  # 打印秒数形式的训练时长


# ===================== 绘制训练损失曲线 =====================
plt.figure(figsize=(12, 8))  # 创建损失曲线画布

plt.plot(train_losses, label='Train Loss', linewidth=4)  # 绘制训练损失曲线
plt.plot(val_losses, label='Validation Loss', linewidth=4)  # 绘制验证损失曲线

plt.title('Model Training - Loss Curves')  # 设置标题
plt.xlabel('Epoch')  # 设置横轴标签
plt.ylabel('Loss (MSE)')  # 设置纵轴标签

ax = plt.gca()  # 获取当前坐标轴对象
ax.xaxis.set_major_locator(MultipleLocator(5))  # 横轴主刻度每 5 轮
ax.set_ylim(0, 0.05)  # 固定纵轴范围
ax.yaxis.set_major_locator(MultipleLocator(0.005))  # 纵轴主刻度间隔 0.005
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 纵轴标签保留 3 位小数
ax.tick_params(axis='both', labelsize=12)  # 设置刻度字体大小

plt.legend()  # 显示图例
plt.grid(True, alpha=0.2)  # 开启淡网格
plt.tight_layout()  # 紧凑布局
plt.savefig('training_loss_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 保存损失曲线图
plt.close()  # 关闭画布


# ===================== 测试集预测与评估 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset - 1:]  # 目标位置对应的时间戳，与 y_test_raw_eval 对齐

print(f"\n测试集数据检查:")  # 打印数据检查标题
print(f"X_test shape: {X_test.shape}")  # 打印输入形状
print(f"y_test shape: {y_test.shape}")  # 打印目标形状
print(f"test_timestamps length: {len(test_timestamps)}")  # 打印时间戳长度

test_results = evaluate_and_plot(  # 评估并绘制测试集预测结果
    model=final_model,  # 待评估模型
    X_data=X_test,  # 测试输入
    y_true_raw=y_test_raw_eval,  # 原始尺度真实值
    target_scaler=target_scaler,  # 目标归一化器
    timestamps=test_timestamps,  # 时间戳
    dataset_name='测试集',  # 数据集名称
    plot_path='test_prediction_GRUWavelet_pytorch.png'  # 图片保存路径
)


# ===================== 模型保存 =====================
torch.save({  # 保存模型权重与配置到检查点文件
    'model_state_dict': final_model.state_dict(),  # 模型权重
    'input_dim': input_dim,  # 输入维度
    'time_step': time_step,  # 时间窗口
    'feature_names': feature_names,  # 小波组合后的特征名
    'wavelet': 'db4',  # 小波基
    'level': 3,  # 分解层数
    'wavelet_window': time_step,  # 小波窗口
    'wavelet_target_col': 'THI',  # 分解目标列
    'feature_scaler': feature_scaler,  # 特征归一化器
    'target_scaler': target_scaler,  # 目标归一化器
    'no_data_leakage': True,  # 标记无数据泄漏
    'original_features': features,  # 原始特征列表
    'model_architecture': 'gru_layers_list',  # 模型结构类型标识
}, 'Model_GRUWavelet_no_leakage_pytorch.pth')  # 检查点保存路径

print("\nGRU模型已保存: Model_GRUWavelet_no_leakage_pytorch.pth")  # 打印保存提示


# ===================== 外部评估数据（可选） =====================
if EVALUATE_EXTERNAL_DATA:  # 是否执行外部数据评估
    print("\n开始处理外部评估数据...")  # 提示开始处理

    external_data = load_csv_with_multiple_encodings(EXTERNAL_FILE)  # 读取外部数据

    # 这里沿用你原来的时间起点
    external_data['timestamp'] = pd.to_datetime(  # 按索引×5分钟生成时间戳
        external_data.index * 5,  # 索引对应的分钟数
        unit='m',  # 单位为分钟
        origin=pd.Timestamp('2025/6/21 0:04')  # 时间起点
    )

    # 外部数据也单独做缺失值处理
    external_data = fill_missing_values(  # 外部数据缺失值处理
        external_data,  # 待处理数据
        features,  # 特征列表
        target,  # 目标列
        dataset_name="外部评估数据"  # 数据集名称
    )

    print("\n对外部评估数据进行滑动窗口小波分解（仅分解THI）...")  # 提示开始外部数据小波分解
    external_wavelet_features = wavelet_decompose_sliding(  # 外部数据小波分解
        external_data[features],  # 输入特征
        target_col='THI',  # 分解目标列
        wavelet='db4',  # 小波基
        level=3,  # 分解层数
        window_size=time_step  # 窗口大小
    )

    external_combined_features = pd.DataFrame(  # 用训练集归一化器变换外部数据
        feature_scaler.transform(external_wavelet_features),  # 归一化数值
        columns=feature_names,  # 保持列名
        index=external_wavelet_features.index  # 保持索引
    )

    y_external_raw_original = external_data[target].values.copy()  # 外部目标原始值副本

    X_external = create_sequences(external_combined_features, time_step, pred_offset)  # 构造外部输入序列
    y_external_raw_eval = create_targets(y_external_raw_original, time_step, pred_offset)  # 构造外部原始目标

    valid_mask = ~np.isnan(y_external_raw_eval)  # 找出目标非空的有效样本
    if valid_mask.sum() < len(y_external_raw_eval):  # 若存在 NaN
        print(f"警告: 外部评估数据目标列中有 {(~valid_mask).sum()} 个 NaN，已过滤")  # 打印过滤数量

    X_external = X_external[valid_mask]  # 过滤输入序列
    y_external_raw_eval = y_external_raw_eval[valid_mask]  # 过滤目标

    print("\nGRU+小波+ECA模型预测外部评估数据...")  # 提示开始预测

    # 数据分布诊断
    external_pred_tensor = torch.FloatTensor(X_external).to(device)  # 外部输入转张量并送设备
    final_model.eval()  # 切换评估模式
    with torch.no_grad():  # 关闭梯度计算
        y_external_pred_norm = final_model(external_pred_tensor).cpu().numpy()  # 前向预测（归一化尺度）
    y_external_pred_rescaled = target_scaler.inverse_transform(y_external_pred_norm)  # 反归一化到原始尺度

    print(f"\n[诊断] 训练集THI范围: {target_scaler.data_min_[0]:.4f} ~ {target_scaler.data_max_[0]:.4f}")  # 打印训练目标范围
    print(f"[诊断] 外部评估数据真实THI范围: {y_external_raw_eval.min():.4f} ~ {y_external_raw_eval.max():.4f}")  # 打印外部真实范围
    print(f"[诊断] 模型预测THI范围（反归一化后）: {y_external_pred_rescaled.min():.4f} ~ {y_external_pred_rescaled.max():.4f}")  # 打印预测范围

    external_timestamps = external_data['timestamp'].values[time_step + pred_offset - 1:]  # 外部数据目标位置时间戳
    external_timestamps = external_timestamps[:len(y_external_raw_eval)]  # 与过滤后目标长度对齐

    external_results = evaluate_and_plot(  # 评估并绘制外部数据预测
        model=final_model,  # 待评估模型
        X_data=X_external,  # 外部输入
        y_true_raw=y_external_raw_eval,  # 原始尺度真实值
        target_scaler=target_scaler,  # 目标归一化器
        timestamps=external_timestamps,  # 时间戳
        dataset_name='外部评估数据',  # 数据集名称
        plot_path='new_data_prediction_GRUWavelet_pytorch.png'  # 图片保存路径
    )

print("\n========== 所有任务完成（GRU+小波+ECA模型） ==========")  # 打印全部完成提示
