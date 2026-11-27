"""
GRU + 小波变换 + ECA 通道注意力的温湿度预测训练脚本（备份版本，dropout=0.2）。

整体流程：
1. 多编码尝试读取温湿度数据，补全时间戳，并对连续/离散变量分别做缺失值处理；
2. 在划分数据集之后，仅对相对湿度做滑动窗口小波分解（避免未来信息泄漏），其余变量原样保留；
3. 只在训练集上拟合 MinMax 归一化器，构造 (batch, time_step, feature) 序列样本；
4. 定义 ECA 注意力层与 GRUWaveletModel，先用 5 折交叉验证评估，再用训练集内部 8:2 划分做早停并训练最终模型；
5. 在测试集与新数据上反归一化评估 R2/RMSE/MAE，绘制对比图，并把模型与归一化器一并保存为 .pth 检查点。
"""

import pandas as pd  # 导入 pandas 库并简写为 pd，用于读取 CSV 与 DataFrame 操作
import numpy as np  # 导入 NumPy 并简写为 np，用于数组构造与数值计算
import time  # 导入 time 模块，用于统计训练耗时
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，以便在使用 pyplot 前切换后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片；这样无需图形界面即可批量出图
import matplotlib.pyplot as plt  # 导入 pyplot 并简写为 plt，用于绘制损失曲线与预测对比图

from sklearn.preprocessing import MinMaxScaler  # 从 sklearn.preprocessing 导入 MinMaxScaler，把特征缩放到 0~1
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 从 sklearn.metrics 导入 R2、MSE、MAE 三个回归评估指标

import torch  # 导入 PyTorch 主库，提供张量运算与 GPU 支持
import torch.nn as nn  # 导入 torch.nn 并简写为 nn，用于搭建 GRU、Dropout、Linear 等网络层
import torch.optim as optim  # 导入 torch.optim 并简写为 optim，这里使用 Adam 优化器
from torch.utils.data import DataLoader, TensorDataset  # 从 torch.utils.data 导入 DataLoader 与 TensorDataset，用于组织批次数据

import pywt  # 小波变换库；这里用 wavedec 对历史窗口信号做多尺度分解
import copy  # 导入 copy 模块，用于深拷贝模型权重以保存最佳模型

# ===================== 辅助函数 =====================
def get_model_size(model):
    """计算模型大小（MB）。

    参数:
        model: 待统计的 PyTorch 模型
    返回:
        参数与缓冲区占用字节数换算后的 MB 数值
    """
    param_size = 0  # 初始化参数总字节数为 0
    for param in model.parameters():  # 遍历模型的所有可训练参数
        param_size += param.nelement() * param.element_size()  # 累加当前参数的元素个数乘以单个元素字节数
    buffer_size = 0  # 初始化缓冲区（如归一化统计量）总字节数为 0
    for buffer in model.buffers():  # 遍历模型的所有缓冲区张量
        buffer_size += buffer.nelement() * buffer.element_size()  # 累加当前缓冲区的字节数
    size_all_mb = (param_size + buffer_size) / 1024**2  # 把参数与缓冲区的总字节数换算为 MB（除以 1024 的平方）
    return size_all_mb  # 返回模型大小（MB）


def format_time(seconds):
    """把秒数格式化为易读的时间字符串。

    参数: seconds 为耗时秒数。
    返回: 不足 1 分钟显示“秒”，不足 1 小时显示“分钟”，否则显示“小时”，均保留两位小数。
    """
    if seconds < 60:  # 耗时不足 60 秒时按秒显示
        return f"{seconds:.2f}秒"  # 返回保留两位小数的秒数字符串
    elif seconds < 3600:  # 耗时在 1 分钟到 1 小时之间时按分钟显示
        minutes = seconds / 60  # 把秒数换算为分钟
        return f"{minutes:.2f}分钟"  # 返回保留两位小数的分钟字符串
    else:  # 耗时超过 1 小时时按小时显示
        hours = seconds / 3600  # 把秒数换算为小时
        return f"{hours:.2f}小时"  # 返回保留两位小数的小时字符串


# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测 CUDA 是否可用，可用则选 GPU，否则选 CPU
print(f"使用设备: {device}")  # 打印本次使用的计算设备
if torch.cuda.is_available():  # 如果存在可用 GPU，则进一步输出显卡信息
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置matplotlib默认字体为黑体，保证中文标题可正常显示
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免坐标轴负号变成方块


# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义候选编码列表，依次尝试直到成功解码 CSV
data = None  # 初始化数据变量为 None，用于标记是否读取成功
for enc in encodings:  # 逐个尝试候选编码
    try:  # try 块用于捕获解码异常
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据 CSV
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码名称
        break  # 读取成功后跳出编码尝试循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误（含其子类）
        continue  # 当前编码失败，继续尝试下一种编码

if data is None:  # 若所有编码都读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出无法读取训练数据文件的异常

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔生成与样本数等长的时间戳列

features = ['日龄', 'CO2', '当前温度', '目标温度','外界温度','相对湿度','风速']  # 定义候选输入特征列名列表（日龄、CO2、温度、湿度、风速等）
target = '相对湿度'  # 指定预测目标列为相对湿度
features = [f for f in features if f in data.columns]  # 只保留数据中真实存在的特征列，避免后续 KeyError

# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印一行分隔线
print("缺失值处理")  # 打印“缺失值处理”阶段标题
print(f"{'=' * 60}")  # 再打印一行分隔线
print(f"原始数据形状: {data.shape}")  # 输出原始数据行数与列数
print(f"\n各列缺失值统计:")  # 打印各列缺失值统计的标题
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅统计数据集中实际存在的列
        nan_count = data[col].isna().sum()  # 统计当前列缺失值（NaN）的个数
        nan_pct = nan_count / len(data) * 100  # 计算当前列缺失值占样本总数的百分比
        if nan_count > 0:  # 只打印存在缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 输出该列缺失值数量与占比

# 定义连续变量和离散变量
continuous_vars = ['CO2', '当前温度', '目标温度','外界温度','相对湿度','风速']  # 定义连续变量列表，这类变量适合用插值填充
discrete_vars = ['日龄']  # 定义离散变量列表，这类变量用前后向填充

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 只处理数据中存在且有缺失值的连续列
        print(f"\n处理 {col}:")  # 打印当前正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值个数

        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 对连续列做线性插值，limit_direction=both 表示首尾两端也填充
        data[col] = data[col].fillna(method='ffill')  # 用前一个有效值（forward fill）填补剩余缺失
        data[col] = data[col].fillna(method='bfill')  # 用后一个有效值（backward fill）填补剩余缺失

        after_nan = data[col].isna().sum()  # 统计插值填充后剩余的缺失值个数
        print(f"  插值前: {before_nan} 个缺失值")  # 输出填充前的缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 输出填充后的缺失值数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 只处理数据中存在且有缺失值的离散列
        print(f"\n处理 {col}:")  # 打印当前正在处理的离散列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值个数

        data[col] = data[col].ffill()  # 用前一个有效值填充离散列的缺失值
        data[col] = data[col].bfill()  # 用后一个有效值填充离散列开头的缺失值

        after_nan = data[col].isna().sum()  # 统计填充后剩余的缺失值个数
        print(f"  填充前: {before_nan} 个缺失值")  # 输出填充前的缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 输出填充后的缺失值数量

remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 如果仍有缺失值
    print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 警告将删除含缺失值的样本行
    data = data.dropna(subset=features + [target])  # 删除特征或目标为 NaN 的行，保证后续建模数据完整

print(f"\n处理后数据形状: {data.shape}")  # 输出清洗后的数据规模
print(f"{'=' * 60}\n")  # 打印缺失值处理阶段结束的分隔线

# 保存原始数据副本（用于后续测试集预测）
data_original = data.copy()  # 深拷贝一份清洗后的数据，供后续测试集绘图与新数据对照使用

print(f"\n【GRU + 小波变换模型】")  # 打印本次实验采用的模型方案标题
print(f"使用特征: {features}")  # 输出实际使用的特征列表
print(f"特征数量: {len(features)}")  # 输出特征个数


# ===================== 小波变换特征提取（仅对相对湿度） =====================
def wavelet_decompose_sliding(data, target_col='相对湿度', wavelet='db4', level=3, window_size=None):
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（相对湿度）进行小波分解，其他列保持原样
    每个时间点只使用历史数据
    参数:
        data: 输入 DataFrame（已按训练集/测试集划分）
        target_col: 需要做小波分解的列名，这里为'相对湿度'
        wavelet: 小波基函数，如'db4'
        level: 分解层数
        window_size: 滑动窗口长度，None 时自动计算
    返回:
        含原始特征与小波系数的 DataFrame
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印开始滑动窗口小波分解的提示与目标列
    print(f"小波基函数: {wavelet}")  # 打印所选小波基函数
    print(f"分解层数: {level}")  # 打印小波分解层数

    if window_size is None:  # 若调用时未指定窗口大小
        window_size = min(2 ** (level + 6), len(data))  # 按分解层数自动计算窗口大小（2^(level+6)），且不超过数据总长度

    print(f"滑动窗口大小: {window_size}")  # 输出最终采用的滑动窗口长度

    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空 DataFrame，用于存放生成的小波特征

    for col in data.columns:  # 遍历数据中的每一列
        if col == 'timestamp':  # 遇到 timestamp 时间戳列时
            continue  # 跳过该列，不做小波分解

        signal = data[col].values  # 取当前列的值并转为 NumPy 数组作为信号

        if col == target_col:  # 若当前列正是需要分解的目标列（相对湿度）
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在对哪一列做小波分解
            approx_list = []  # 初始化近似系数（低频）列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数各初始化一个空列表

            for i in range(len(signal)):  # 逐时间点滑动，构造截至当前时刻的历史窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证只使用当前及更早的数据，杜绝未来信息泄漏
                window_signal = signal[start_idx:i + 1]  # 截取从 start_idx 到当前点（含当前点）的信号片段

                if len(window_signal) < window_size:  # 若历史数据不足一个完整窗口
                    pad_width = window_size - len(window_signal)  # 计算需要向前填充的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用窗口前端边缘值（mode=edge）把信号补齐到指定长度

                try:  # try 块：尝试执行小波分解，失败时走退避逻辑
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 pywt.wavedec 对窗口信号做 level 层小波分解

                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值，代表当前时刻的低频趋势
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层细节系数的最后一个值作为当前时刻特征，若为空则补 0.0
                except Exception:  # 分解过程抛出异常时进入退避分支
                    approx_list.append(window_signal[-1])  # 用窗口最后一个原始信号值代替近似系数
                    for j in range(level):  # 遍历每一层细节系数
                        detail_lists[j].append(0.0)  # 分解失败时用 0.0 填充该层细节系数

            wavelet_features[f'{col}_approx'] = approx_list  # 把近似系数序列写入结果表，列名为“列名_approx”
            for j in range(level):  # 遍历每一层细节系数
                wavelet_features[f'{col}_detail{j + 1}'] = detail_lists[j]  # 把第 j+1 层细节系数写入结果表，列名为“列名_detailN”
        else:  # 当前列不是目标列时的分支
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列保持原样、不做分解
            wavelet_features[col] = signal  # 把该列信号原样写入结果表

    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 输出小波分解完成后的特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 输出来自目标列的小波特征数量（1 个近似 + level 个细节）
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 输出保持不变的原始特征数量（排除 timestamp）
    return wavelet_features  # 返回包含原始特征与小波特征的 DataFrame


# ===================== 数据集划分（在小波分解之前！） =====================
time_step = 36   # 36步 × 5分钟 = 3小时，即用过去 3 小时的数据做输入
pred_offset = 6  # 6步 × 5分钟 = 30分钟，即预测 30 分钟后的相对湿度

print(f"\n时间窗口配置:")  # 打印时间窗口配置的标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 输出 GRU 输入序列长度及对应的分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 输出预测偏移步数及对应的分钟数

split_index = int(len(data) * 0.8)  # 按 8:2 计算训练集与测试集的切分下标
train_data_raw = data.iloc[:split_index].copy()  # 取前 80% 数据作为原始训练集
test_data_raw = data.iloc[split_index:].copy()  # 取后 20% 数据作为原始测试集

print(f"\n数据集划分:")  # 打印数据集划分结果的标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw) / len(data) * 100:.1f}%)")  # 输出训练集样本数与占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw) / len(data) * 100:.1f}%)")  # 输出测试集样本数与占比

print("\n对训练集进行小波分解（仅分解相对湿度）...")  # 打印“对训练集做小波分解”的提示
wavelet_window = time_step  # 令小波窗口长度与 GRU 时间步长一致
print(f"小波窗口大小设置为: {wavelet_window} 步 (与GRU时间窗口一致)")  # 输出实际使用的小波窗口大小
train_wavelet_features = wavelet_decompose_sliding(  # 调用滑动窗口分解函数处理训练集特征
    train_data_raw[features],  # 传入训练集的原始特征列
    target_col='相对湿度',  # 指定仅对相对湿度列做分解
    wavelet='db4',  # 使用 db4 小波基函数
    level=3,  # 设定分解层数为 3
    window_size=wavelet_window  # 窗口大小取前面设置的 wavelet_window
)  # 函数调用结束

print("\n对测试集进行小波分解（仅分解相对湿度）...")  # 打印“对测试集做小波分解”的提示
test_wavelet_features = wavelet_decompose_sliding(  # 调用同一个分解函数处理测试集特征（参数与训练集一致）
    test_data_raw[features],  # 传入测试集的原始特征列
    target_col='相对湿度',  # 指定仅分解相对湿度列
    wavelet='db4',  # 同样使用 db4 小波基
    level=3,  # 同样分解 3 层
    window_size=wavelet_window  # 窗口大小与训练集保持一致，避免分布不一致
)  # 函数调用结束


def visualize_wavelet_decomposition(original_signal, wavelet_features, feature_name,
                                    save_path='wavelet_decomposition.png'):  # save_path 参数指定分解图的保存路径（函数签名的续行）
    """可视化小波分解结果：原始信号、近似系数与各层细节系数。

    参数: original_signal 原始信号，wavelet_features 小波特征表，feature_name 特征名，save_path 图片路径。
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))  # 创建 5 行 1 列的画布，分别画原始信号、近似系数和 3 层细节系数

    axes[0].plot(original_signal, linewidth=1, color='black')  # 第 1 个子图用黑色细线绘制原始信号
    axes[0].set_title(f'原始信号: {feature_name}', fontsize=12, fontweight='bold')  # 设置原始信号子图标题
    axes[0].set_ylabel('幅值')  # 设置原始信号子图 y 轴标签为“幅值”
    axes[0].grid(True, alpha=0.3)  # 为原始信号子图添加半透明网格

    axes[1].plot(wavelet_features[f'{feature_name}_approx'].values, linewidth=1, color='blue')  # 第 2 个子图绘制近似系数（低频趋势成分）
    axes[1].set_title('近似系数 (低频成分 - 趋势)', fontsize=12)  # 设置近似系数子图标题
    axes[1].set_ylabel('幅值')  # 设置近似系数子图 y 轴标签
    axes[1].grid(True, alpha=0.3)  # 为近似系数子图添加网格

    colors = ['green', 'orange', 'red']  # 定义 3 层细节系数的绘图颜色：绿、橙、红
    for i in range(1, 4):  # 循环绘制第 1 到第 3 层细节系数
        axes[i + 1].plot(  # 在第 i+1 个子图绘制第 i 层细节系数（plot 调用开始）
            wavelet_features[f'{feature_name}_detail{i}'].values,  # 取第 i 层细节系数的时间序列
            linewidth=1,  # 设置线宽为 1
            color=colors[i - 1]  # 按层序选择对应颜色
        )  # plot 调用结束
        axes[i + 1].set_title(f'细节系数 Level {i} (高频成分 - 波动)', fontsize=12)  # 设置细节系数子图标题，标明层数
        axes[i + 1].set_ylabel('幅值')  # 设置细节系数子图 y 轴标签
        axes[i + 1].grid(True, alpha=0.3)  # 为细节系数子图添加网格

    axes[-1].set_xlabel('时间步', fontsize=11)  # 在最后一个子图设置 x 轴标签为“时间步”
    plt.tight_layout()  # 自动调整各子图间距，避免文字重叠
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 按指定路径保存小波分解图，dpi=150
    print(f"小波分解可视化已保存: {save_path}")  # 打印分解图保存路径
    plt.close()  # 关闭图形释放内存


first_feature = '相对湿度'  # 指定要可视化的特征为相对湿度
visualize_wavelet_decomposition(  # 调用小波分解可视化函数
    train_data_raw[first_feature].values,  # 传入训练集的原始相对湿度信号
    train_wavelet_features,  # 传入训练集的小波特征表
    first_feature,  # 传入特征名
    'wavelet_decomposition_temperature_deviation.png'  # 指定输出图片文件名
)  # 函数调用结束

print(f"\n原始特征数: {len(features)}")  # 输出小波分解前的原始特征数量
print(f"小波分解后特征数: {len(train_wavelet_features.columns)}")  # 输出小波分解后的特征总数
print(f"  - 相对湿度的小波特征: 4 个（1个近似系数 + 3个细节系数）")  # 说明新增的 4 个小波特征构成（1 近似 + 3 细节）
print(f"  - 其他原始特征: {len(features) - 1} 个")  # 说明保留的其他原始特征数量

train_combined_features = train_wavelet_features  # 训练集组合特征直接使用小波特征表（其中已包含原始列）
test_combined_features = test_wavelet_features  # 测试集组合特征同样使用小波特征表
feature_names = list(train_wavelet_features.columns)  # 记录最终特征名列表，供归一化与建模使用

print(f"\n使用模式: 相对湿度小波分解 + 其他原始特征")  # 打印当前特征使用模式
print(f"总特征数: {len(feature_names)}")  # 输出最终特征总数
print(f"特征列表: {feature_names}")  # 输出最终特征名清单

# ===================== 归一化（只在训练集上fit！） =====================
print("\n归一化处理（避免数据泄漏）...")  # 打印归一化阶段提示，强调避免数据泄漏
feature_scaler = MinMaxScaler()  # 创建特征归一化器（最小-最大缩放）
target_scaler = MinMaxScaler()  # 创建目标值归一化器

feature_scaler.fit(train_combined_features)  # 只在训练集特征上拟合归一化器（关键：不在全量数据上 fit）
target_scaler.fit(train_data_raw[[target]])  # 只在训练集目标值上拟合目标归一化器

train_combined_features = pd.DataFrame(  # 对训练集特征做归一化并重建 DataFrame（调用开始）
    feature_scaler.transform(train_combined_features),  # 用训练集拟合的 scaler 变换训练集特征
    columns=feature_names  # 保持原有列名不变
)  # DataFrame 构造结束
test_combined_features = pd.DataFrame(  # 对测试集特征做归一化并重建 DataFrame（调用开始）
    feature_scaler.transform(test_combined_features),  # 用同一个训练集 scaler 变换测试集特征，不再重新 fit
    columns=feature_names  # 列名与训练集保持一致
)  # DataFrame 构造结束

train_data_raw[target] = target_scaler.transform(train_data_raw[[target]])  # 用目标 scaler 把训练集目标值缩放到 0~1
test_data_raw[target] = target_scaler.transform(test_data_raw[[target]])  # 用同一个 scaler 把测试集目标值缩放到 0~1


# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):
    """构造监督学习的序列样本。
    参数:
        data: 已归一化的特征表，每一行是一个时间点
        time_step: 每个样本包含的历史时间步数
    返回:
        形状为 (样本数, time_step, 特征数) 的三维 NumPy 数组
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点范围，预留 pred_offset 以保证每个样本都有对应标签
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 步历史特征作为一个样本
    return np.array(sequences)  # 把所有样本堆叠成三维数组并返回


print("\n创建训练集序列...")  # 打印创建训练集序列的提示
X_train = create_sequences(train_combined_features, time_step)  # 调用 create_sequences 构造训练集特征序列
y_train = train_data_raw[target].values[time_step + pred_offset:]  # 取偏移之后的训练目标值作为标签
y_train = y_train[:len(X_train)]  # 截断标签长度，使其与特征样本数严格一致

print("\n创建测试集序列...")  # 打印创建测试集序列的提示
X_test = create_sequences(test_combined_features, time_step)  # 调用 create_sequences 构造测试集特征序列
y_test = test_data_raw[target].values[time_step + pred_offset:]  # 取偏移之后的测试目标值作为标签
y_test = y_test[:len(X_test)]  # 截断测试标签长度，使其与样本数一致

print(f"\n序列数据形状:")  # 打印序列数据形状的标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 输出训练集特征与标签的形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 输出测试集特征与标签的形状


# ===================== PyTorch模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    参数:
        channels: 输入特征通道数
        gamma: 用于计算卷积核大小的缩放系数
        b: 用于计算卷积核大小的偏置项
    说明:
        通过全局平均池化 + 1D 卷积学习每个通道的注意力权重
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 通道注意力层。
        参数: channels 为输入通道数，gamma/b 用于自适应推导 1D 卷积核大小。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应计算卷积核参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数（便于左右对称 padding）

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 定义自适应全局平均池化，把时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 定义 1D 卷积，通过相邻通道交互学习注意力权重
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活，把注意力权重映射到 0~1

    def forward(self, x):
        """前向传播：对 (batch, time, channels) 输入施加通道注意力。
        参数: x 为三维输入张量；返回同形状的加权后张量。
        """
        x_permuted = x.permute(0, 2, 1)  # 把输入从 (batch, time, channels) 转置为 (batch, channels, time)
        y = self.avg_pool(x_permuted)  # 沿时间维做全局平均池化，得到每个通道的全局描述
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 用 1D 卷积学习通道注意力，再转回原维度顺序
        y = self.sigmoid(y)  # 用 Sigmoid 得到 0~1 的通道权重
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 把权重按通道广播相乘，实现特征重标定
        return x_permuted.permute(0, 2, 1)  # 转回 (batch, time, channels) 并返回


class GRUWaveletModel(nn.Module):
    """
    GRU + 小波变换 + ECA注意力机制
    参数:
        input_dim: 输入特征维度
        time_step: 时间步长
        gru_hidden: 隐藏层维度，可为整数或列表（逐层指定）
        gru_layers: GRU 层数
        dropout: Dropout 比率（本备份版本为 0.2）
    说明:
        输入先经 ECA 注意力增强，再逐层通过 GRU，取最后时刻输出经全连接层回归
    """
    def __init__(self, input_dim, time_step, gru_hidden=[128], gru_layers=1, dropout=0.2):
        """初始化 GRU+小波+ECA 模型。
        参数: input_dim 输入维度，time_step 时间步，gru_hidden 隐藏维度（可列表），
              gru_layers 层数，dropout 丢弃率；构建 ECA、多层 GRU、Dropout 与全连接层。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 记录时间步长配置
        self.input_dim = input_dim  # 记录输入特征维度
        self.gru_hidden = gru_hidden  # 记录隐藏层配置

        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 通道注意力模块，通道数等于输入特征数

        # 若 gru_hidden 为列表，则逐层构建 GRU，每层hidden_size可不同
        if isinstance(gru_hidden, list):  # 判断 gru_hidden 是否为列表（支持每层不同隐藏维度）
            hidden_sizes = gru_hidden  # 是列表则直接作为各层隐藏维度
        else:  # 否则是单一整数
            hidden_sizes = [gru_hidden] * gru_layers  # 把同一隐藏维度复制 gru_layers 次

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多层 GRU，保证参数能被正确注册
        in_size = input_dim  # 记录当前层的输入维度，初值为输入特征数
        for i, h_size in enumerate(hidden_sizes):  # 逐层遍历隐藏维度配置
            self.gru_layers_list.append(  # 把新建的 GRU 层加入 ModuleList（调用开始）
                nn.GRU(  # 实例化单层 GRU（调用开始）
                    input_size=in_size,  # 输入维度为上一层的输出维度
                    hidden_size=h_size,  # 当前层的隐藏单元个数
                    num_layers=1,  # 每次只创建 1 层，多层通过循环堆叠实现
                    dropout=0.0,  # 单层 GRU 内部不额外加 dropout（dropout 由外部层负责）
                    batch_first=True,  # batch_first=True 表示输入形状为 (batch, time, feature)
                    bidirectional=False  # 使用单向 GRU
                )  # GRU 实例化结束
            )  # append 调用结束
            in_size = h_size  # 把下一层的输入维度更新为当前隐藏维度

        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层，训练时按概率随机丢弃神经元输出
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 定义全连接层，把最后一层隐藏状态映射为单个预测值

    def forward(self, x):
        """前向传播：ECA -> 逐层 GRU -> 取末时刻 -> 全连接回归。
        参数: x 形状为 (batch, time_step, input_dim)；返回形状为 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 先用 ECA 注意力增强输入特征
        out = x  # 把增强后的特征作为 GRU 的输入
        for gru in self.gru_layers_list:  # 逐层执行 GRU 前向传播
            out, _ = gru(out)  # 取 GRU 的输出序列，忽略末隐藏状态
            out = self.dropout(out)  # 对每层输出施加 Dropout，抑制过拟合
        out_last = out[:, -1, :]  # 取序列最后一个时间步的输出作为整段序列的汇总表示
        return self.fc(out_last)  # 经全连接层输出最终标量预测


class EarlyStopping:
    """早停策略：验证损失连续 patience 轮无明显下降时停止训练，并保留最佳权重。"""
    def __init__(self, patience=10, min_delta=0):
        """初始化早停控制器。
        参数: patience 为容忍验证损失不改善的轮数，min_delta 为认定改善的最小下降量。
        """
        self.patience = patience  # 保存容忍验证损失不改善的轮数
        self.min_delta = min_delta  # 保存认定“有改善”所需的最小损失下降量
        self.counter = 0  # 初始化“未改善”计数器为 0
        self.best_loss = None  # 初始化最佳验证损失为空
        self.early_stop = False  # 初始化早停标志为 False
        self.best_model = None  # 初始化最佳模型权重容器为空

    def __call__(self, val_loss, model):
        """根据当前验证损失更新最佳模型并判断是否早停。
        参数: val_loss 为本轮验证损失，model 为当前模型；无返回值，通过属性反映状态。
        """
        if self.best_loss is None:  # 首次调用时还没有历史最佳损失
            self.best_loss = val_loss  # 把当前验证损失记为最佳
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝当前模型权重并保存为最佳模型
        elif val_loss > self.best_loss - self.min_delta:  # 若验证损失没有比最佳值下降超过 min_delta
            self.counter += 1  # 未改善计数器加一
            if self.counter >= self.patience:  # 若累计未改善轮数达到 patience
                self.early_stop = True  # 把早停标志置为 True
        else:  # 验证损失确实改善了
            self.best_loss = val_loss  # 更新最佳验证损失
            self.best_model = copy.deepcopy(model.state_dict())  # 重新深拷贝当前模型权重作为最佳模型
            self.counter = 0  # 重置未改善计数器


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):
    """训练模型并返回每轮损失记录。

    参数: model 待训练模型，train_loader/val_loader 为训练与验证数据加载器，epochs 最大轮数，lr 学习率，patience 早停容忍轮数。
    返回: (train_losses, val_losses) 两个损失列表。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数作为训练目标
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新模型参数，学习率由参数传入
    early_stopping = EarlyStopping(patience=patience)  # 实例化早停控制器，传入容忍轮数

    train_losses = []  # 初始化每轮训练损失记录列表
    val_losses = []  # 初始化每轮验证损失记录列表

    for epoch in range(epochs):  # 按设定最大轮数循环训练
        model.train()  # 切换模型为训练模式（启用 Dropout）
        train_loss = 0.0  # 初始化当前 epoch 的累计训练损失
        for batch_x, batch_y in train_loader:  # 按批次遍历训练数据加载器
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把一个批次的输入和标签迁移到目标设备

            optimizer.zero_grad()  # 清空优化器中上一批次累积的梯度
            outputs = model(batch_x)  # 前向传播，得到当前批次的预测值
            loss = criterion(outputs, batch_y)  # 计算预测值与真实标签的 MSE 损失
            loss.backward()  # 反向传播计算各参数梯度
            optimizer.step()  # 优化器根据梯度更新一次模型参数

            train_loss += loss.item()  # 累加当前批次的损失值（乘 batch_size 即总损失）

        train_loss /= len(train_loader)  # 除以批次数，得到当前 epoch 的平均训练损失
        train_losses.append(train_loss)  # 把本轮平均训练损失追加到记录列表

        model.eval()  # 切换模型为评估模式（关闭 Dropout）
        val_loss = 0.0  # 初始化当前 epoch 的累计验证损失
        with torch.no_grad():  # 关闭梯度计算，节省显存并加速验证
            for batch_x, batch_y in val_loader:  # 按批次遍历验证数据加载器
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把验证批次数据迁移到目标设备
                outputs = model(batch_x)  # 前向传播得到验证集预测
                loss = criterion(outputs, batch_y)  # 计算验证集上的 MSE 损失
                val_loss += loss.item()  # 累加当前批次的验证损失

        val_loss /= len(val_loader)  # 求当前 epoch 的平均验证损失
        val_losses.append(val_loss)  # 把本轮平均验证损失追加到记录列表

        if (epoch + 1) % 10 == 0:  # 每 10 个 epoch 打印一次训练日志
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 输出当前轮次、训练损失与验证损失

        early_stopping(val_loss, model)  # 调用早停控制器，传入本轮验证损失与模型
        if early_stopping.early_stop:  # 若早停标志被触发
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停发生的具体 epoch
            model.load_state_dict(early_stopping.best_model)  # 加载之前保存的最佳模型权重
            break  # 跳出训练循环

    if early_stopping.best_model is not None:  # 训练正常结束后，若曾保存过最佳权重
        model.load_state_dict(early_stopping.best_model)  # 加载最佳权重作为最终模型状态

    return train_losses, val_losses  # 返回训练损失与验证损失两条曲线记录


# ===================== K折交叉验证训练（仅在训练集上） =====================
from sklearn.model_selection import KFold  # 从 sklearn.model_selection 导入 KFold，用于 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建 5 折划分器，打乱样本并固定随机种子 42 保证可复现
val_losses_all = []  # 初始化各折最佳验证损失列表
history_list = []  # 初始化各折训练/验证损失曲线记录列表
input_dim = len(feature_names)  # 输入维度等于最终特征数

print("\n========== 开始5折交叉验证（仅训练集） ==========")  # 打印交叉验证开始的提示
print(f"交叉验证数据量: {len(X_train)} 样本")  # 输出参与交叉验证的训练样本总数

for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):  # enumerate 从 1 开始，逐折取训练索引与验证索引
    print(f"\n{'=' * 50}")  # 打印本折的分隔线
    print(f"第 {fold}/5 折训练")  # 打印当前是第几折训练
    print(f"{'=' * 50}")  # 再打印一条分隔线

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # 按索引切出本折的训练特征与验证特征
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # 按索引切出本折的训练标签与验证标签

    print(f"训练集样本数: {len(X_train_fold)}")  # 输出本折训练样本数
    print(f"验证集样本数: {len(X_val_fold)}")  # 输出本折验证样本数

    X_train_fold_tensor = torch.FloatTensor(X_train_fold)  # 本折训练特征转为 FloatTensor
    y_train_fold_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)  # 本折训练标签转为张量，并用 unsqueeze(1) 变成列向量
    X_val_fold_tensor = torch.FloatTensor(X_val_fold)  # 本折验证特征转为 FloatTensor
    y_val_fold_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)  # 本折验证标签转为张量并补一维

    fold_train_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)  # 用训练张量构造 TensorDataset
    fold_val_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)  # 用验证张量构造 TensorDataset
    fold_train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True)  # 构造训练 DataLoader，batch_size=32 且每轮打乱
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=32, shuffle=False)  # 构造验证 DataLoader，batch_size=32 且不打乱

    fold_model = GRUWaveletModel(input_dim, time_step, gru_hidden=[128], gru_layers=1, dropout=0.2  # 实例化本折 GRUWaveletModel：隐藏层 128、1 层、dropout=0.2
                                 ).to(device)  # 把本折模型迁移到目标设备（续行）

    if fold == 1:  # 仅在第一折打印模型结构相关信息
        print(f"模型参数量: {sum(p.numel() for p in fold_model.parameters()):,}")  # 输出模型可训练参数量（用千分位格式）

    fold_train_losses, fold_val_losses = train_model(fold_model, fold_train_loader, fold_val_loader,  # 调用 train_model 训练本折模型
                                                     epochs=200, lr=0.001, patience=10)  # 指定 200 轮、学习率 0.001、早停容忍 10 轮（续行）
    history_list.append({  # 把本折损失记录以字典形式追加到历史列表（调用开始）
        'train_losses': fold_train_losses,  # 保存本折训练损失序列
        'val_losses': fold_val_losses  # 保存本折验证损失序列
    })  # 字典与 append 调用结束

    best_val_loss = min(fold_val_losses)  # 取本折验证损失的最小值作为最佳验证损失
    val_losses_all.append(best_val_loss)  # 把本折最佳验证损失加入汇总列表
    print(f"\n第 {fold} 折最佳验证损失: {best_val_loss:.6f}")  # 输出本折最佳验证损失

print(f"\n{'=' * 50}")  # 打印交叉验证结果汇总的分隔线
print("5折交叉验证结果汇总")  # 打印汇总标题
print(f"{'=' * 50}")  # 再打印一条分隔线
for i, loss in enumerate(val_losses_all, 1):  # 遍历各折的最佳验证损失
    print(f"第 {i} 折验证损失: {loss:.6f}")  # 逐折输出最佳验证损失
print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")  # 输出 5 折验证损失的平均值
print(f"验证损失标准差: {np.std(val_losses_all):.6f}")  # 输出 5 折验证损失的标准差

# ===================== 绘制K折交叉验证训练过程曲线 =====================
plt.figure(figsize=(15, 10))  # 创建 15x10 英寸的交叉验证汇总画布

plt.subplot(2, 2, 1)  # 选中第 1 个子图（训练损失）
for i, history in enumerate(history_list, 1):  # 遍历每折的训练历史
    plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制某一折的训练损失曲线，alpha=0.7 表示半透明
plt.title('5-Fold Cross Validation - Training Loss', fontsize=14)  # 设置训练损失子图标题
plt.xlabel('Epoch')  # 设置 x 轴标签为 Epoch
plt.ylabel('Loss (MSE)')  # 设置 y 轴标签为 Loss (MSE)
plt.legend()  # 显示图例（各折）
plt.grid(True, alpha=0.3)  # 添加半透明网格线

plt.subplot(2, 2, 2)  # 选中第 2 个子图（验证损失）
for i, history in enumerate(history_list, 1):  # 再次遍历每折的训练历史
    plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制某一折的验证损失曲线
plt.title('5-Fold Cross Validation - Validation Loss', fontsize=14)  # 设置验证损失子图标题
plt.xlabel('Epoch')  # 设置 x 轴标签
plt.ylabel('Loss (MSE)')  # 设置 y 轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格

plt.subplot(2, 2, 3)  # 选中第 3 个子图（各折最佳损失柱状图）
folds = [f'Fold {i}' for i in range(1, 6)]  # 生成 Fold 1 到 Fold 5 的标签列表
plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)  # 用柱状图展示每折的最佳验证损失
plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',  # 画一条红色虚线表示平均验证损失（调用开始）
            label=f'Mean: {np.mean(val_losses_all):.6f}')  # 为均值参考线设置图例文字（续行）
plt.title('Best Validation Loss per Fold', fontsize=14)  # 设置柱状图子图标题
plt.xlabel('Fold')  # 设置 x 轴标签为 Fold
plt.ylabel('Validation Loss (MSE)')  # 设置 y 轴标签为 Validation Loss (MSE)
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3, axis='y')  # 仅沿 y 轴添加网格

plt.subplot(2, 2, 4)  # 选中第 4 个子图（跨折平均损失曲线）
max_epochs = max(len(h['train_losses']) for h in history_list)  # 求所有折中最长的训练轮数，作为横轴长度
avg_train_losses = []  # 初始化跨折平均训练损失列表
avg_val_losses = []  # 初始化跨折平均验证损失列表
for epoch in range(max_epochs):  # 逐 epoch 计算各折的平均损失
    train_vals = [h['train_losses'][epoch] for h in history_list if epoch < len(h['train_losses'])]  # 收集本 epoch 所有折的训练损失（只取有记录的折）
    val_vals = [h['val_losses'][epoch] for h in history_list if epoch < len(h['val_losses'])]  # 收集本 epoch 所有折的验证损失
    if train_vals:  # 若本 epoch 存在训练损失记录
        avg_train_losses.append(np.mean(train_vals))  # 计算平均值并追加到平均训练损失列表
    if val_vals:  # 若本 epoch 存在验证损失记录
        avg_val_losses.append(np.mean(val_vals))  # 计算平均值并追加到平均验证损失列表
plt.plot(avg_train_losses, label='Avg Train Loss', linewidth=2)  # 绘制平均训练损失曲线，加粗显示
plt.plot(avg_val_losses, label='Avg Val Loss', linewidth=2)  # 绘制平均验证损失曲线，加粗显示
plt.title('5-Fold Average Loss Curves', fontsize=14)  # 设置平均损失子图标题
plt.xlabel('Epoch')  # 设置 x 轴标签
plt.ylabel('Loss (MSE)')  # 设置 y 轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格

plt.tight_layout()  # 自动调整 4 个子图间距
plt.savefig('kfold_training_GRUWaveletECA_pytorch.png', dpi=150, bbox_inches='tight')  # 把交叉验证汇总图保存为 png 文件
print("\nK折交叉验证损失曲线已保存: kfold_training_GRUWaveletECA_pytorch.png")  # 打印汇总图保存路径

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印“训练最终模型”阶段标题
print(f"{'=' * 50}")  # 再打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 输出最终训练集样本数
print(f"测试集样本数: {len(X_test)}")  # 输出测试集样本数

X_train_tensor = torch.FloatTensor(X_train)  # 训练集特征转为 FloatTensor
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练集标签转为张量并补一维
X_test_tensor = torch.FloatTensor(X_test)  # 测试集特征转为 FloatTensor
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试集标签转为张量并补一维

# 从训练集中分割验证集（80-20），避免使用测试集进行早停
train_val_split = int(len(X_train) * 0.8)  # 在训练集内部按 8:2 计算切分点（验证集不来自测试集）
X_train_split = X_train_tensor[:train_val_split]  # 取前 80% 训练张量作为最终训练输入
y_train_split = y_train_tensor[:train_val_split]  # 取前 80% 训练张量作为最终训练标签
X_val_split = X_train_tensor[train_val_split:]  # 取后 20% 训练张量作为内部验证输入
y_val_split = y_train_tensor[train_val_split:]  # 取后 20% 训练张量作为内部验证标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 构造最终训练数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 构造内部验证数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 构造测试数据集

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构造打乱顺序的最终训练 DataLoader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 构造顺序读取的内部验证 DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 构造顺序读取的测试 DataLoader

input_dim = len(feature_names)  # 确认模型输入维度为特征数
final_model = GRUWaveletModel(input_dim, time_step, gru_hidden=[128], gru_layers=1, dropout=0.2).to(device)  # 实例化最终 GRUWaveletModel 并迁移到目标设备

print(final_model)  # 打印最终模型结构

print("\n开始训练最终模型...")  # 打印开始训练最终模型的提示
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 输出最终模型的可训练参数量
print(f"模型大小: {get_model_size(final_model):.2f} MB")  # 调用 get_model_size 输出模型占用空间（MB）
print(f"训练集样本数: {len(X_train_split)}")  # 输出实际用于训练的样本数
print(f"验证集样本数: {len(X_val_split)}")  # 输出内部验证样本数

training_start_time = time.time()  # 记录训练开始时间（时间戳）

train_losses, val_losses = train_model(  # 调用 train_model 开始训练最终模型（调用开始）
    final_model,  # 传入最终模型
    train_loader,  # 传入训练数据加载器
    val_loader,  # 传入内部验证数据加载器
    epochs=200,  # 最大训练轮数 200
    lr=0.001,  # 学习率 0.001
    patience=10  # 早停容忍轮数 10
)  # 函数调用结束

training_end_time = time.time()  # 记录训练结束时间（时间戳）
total_training_time = training_end_time - training_start_time  # 用结束时间减开始时间得到总训练秒数

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 输出训练完成信息与最佳（最小）验证损失
print(f"训练总时长: {format_time(total_training_time)}")  # 用 format_time 以易读格式输出总耗时
print(f"训练总时长（秒）: {total_training_time:.2f}秒")  # 以秒为单位再次输出总耗时

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12x6 英寸的训练曲线画布
plt.plot(train_losses, label='Train Loss', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='Validation Loss', linewidth=2)  # 绘制验证损失曲线
plt.title('Model Training - Loss Curves')  # 设置图表标题
plt.xlabel('Epoch')  # 设置 x 轴标签为 Epoch
plt.ylabel('Loss (MSE)')  # 设置 y 轴标签为 Loss (MSE)
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
plt.tight_layout()  # 调整布局防止文字被裁切
plt.savefig('training_loss_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把训练曲线保存为 png 文件
print("Training loss curve saved: training_loss_GRUWavelet_pytorch.png")  # 打印训练曲线保存路径


# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 打印“开始测试集预测”的提示

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 重新构造测试集 DataLoader（保持原有代码）

y_test_normalized = y_test  # 用一个变量保存归一化后的测试标签
y_test_rescaled = target_scaler.inverse_transform(y_test_normalized.reshape(-1, 1))  # 把测试标签反归一化回原始量纲，便于和业务指标比较

print(f"\n测试集数据检查:")  # 打印测试集数据检查的标题
print(f"X_test shape: {X_test.shape}")  # 输出测试特征数组的形状
print(f"y_test shape: {y_test.shape}")  # 输出测试标签数组的形状
print(f"y_test_rescaled shape: {y_test_rescaled.shape}")  # 输出反归一化后测试标签的形状

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试特征转为 FloatTensor 并迁移到目标设备

final_model.eval()  # 把最终模型切换为评估模式（关闭 Dropout）
with torch.no_grad():  # 关闭梯度计算以加速预测
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 前向预测，并把结果从 GPU 转回 CPU 再转 NumPy 数组

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 把预测值反归一化回原始量纲

print(f"y_pred_rescaled shape: {y_pred_rescaled.shape}")  # 输出反归一化预测值的形状

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse_test = np.sqrt(mse_test)  # 对 MSE 开平方得到 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印评估结果的分隔线
print("测试集评估结果（GRU+小波+ECA）")  # 打印评估标题（GRU+小波+ECA）
print(f"{'=' * 60}")  # 再打印一条分隔线
print(f"测试集 R2:   {r2:.4f}")  # 输出测试集 R2
print(f"测试集 RMSE: {rmse_test:.4f}")  # 输出测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 输出测试集 MAE
print(f"{'=' * 60}")  # 打印评估结束分隔线

plt.figure(figsize=(15, 5))  # 创建 15x5 英寸的测试预测对比图画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 把反归一化的真实值展平成一维数组
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 把反归一化的预测值展平成一维数组

test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset:]  # 按 time_step+pred_offset 偏移截取与预测对应的测试时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳长度，使其与真实值点数一致

print(f"\n绘图数据检查:")  # 打印绘图数据检查标题
print(f"test_timestamps length: {len(test_timestamps)}")  # 输出时间戳数组长度
print(f"y_test_rescaled_flat length: {len(y_test_rescaled_flat)}")  # 输出真实值数组长度
print(f"y_pred_rescaled_flat length: {len(y_pred_rescaled_flat)}")  # 输出预测值数组长度

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实相对湿度', linewidth=2)  # 用时间戳作横轴绘制真实相对湿度曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+小波+ECA预测', linewidth=2)  # 在同一张图绘制模型预测曲线
plt.xlabel('时间')  # 设置 x 轴标签为“时间”
plt.ylabel('相对湿度 ')  # 设置 y 轴标签为“相对湿度”
plt.title(f'测试集相对湿度结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置标题并附带 R2 与 RMSE 指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
plt.savefig('test_prediction_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把测试预测对比图保存为 png
print("测试集预测结果已保存: test_prediction_GRUWavelet_pytorch.png")  # 打印测试预测图保存路径
plt.close()  # 关闭图形释放内存


# ===================== 模型保存 =====================
torch.save({  # 构造检查点字典并调用 torch.save 写入文件（调用开始）
    'model_state_dict': final_model.state_dict(),  # 保存最终模型的权重状态字典
    'input_dim': input_dim,  # 保存输入特征维度配置
    'time_step': time_step,  # 保存时间步长配置
    'feature_names': feature_names,  # 保存特征名列表，便于加载时对齐
    'wavelet': 'db4',  # 保存使用的小波基函数名称
    'level': 3,  # 保存小波分解层数
    'wavelet_window': time_step,  # 保存小波滑动窗口大小
    'wavelet_target_col': '相对湿度',  # 保存被分解的目标列名
    'feature_scaler': feature_scaler,  # 保存特征归一化器对象
    'target_scaler': target_scaler,  # 保存目标归一化器对象
    'no_data_leakage': True,  # 标记该模型训练过程无数据泄漏
    'original_features': features,  # 保存原始特征列表
}, 'Model_GRUWavelet_no_leakage_pytorch.pth')  # 指定检查点文件名并完成保存
print("\nGRU模型已保存: Model_GRUWavelet_no_leakage_pytorch.pth")  # 打印模型保存成功的提示


# ===================== 新数据预测 =====================
print("\n开始处理新数据")  # 打印“开始处理新数据”的提示

new_data = None  # 初始化新数据变量为 None
for enc in encodings:  # 逐个尝试候选编码读取新数据
    try:  # try 块用于捕获解码异常
        new_data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取新数据 CSV
        print(f"成功使用 {enc} 编码加载新数据")  # 打印成功读取新数据所用编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 继续尝试下一种编码

if new_data is None:  # 若所有编码都读取失败
    raise ValueError("无法使用任何已知编码读取新数据文件")  # 抛出新数据文件读取失败异常

new_data['timestamp'] = pd.to_datetime(  # 为新数据构造时间戳（调用开始）
    new_data.index * 5,  # 用数据行索引乘以 5，表示每行相隔 5 分钟
    unit='m',  # 指定乘数的单位为分钟
    origin=pd.Timestamp('2025/6/21 0:04')  # 指定时间起点为 2025/6/21 0:04
)  # pd.to_datetime 调用结束

print("\n对新数据进行滑动窗口小波分解（仅分解相对湿度）...")  # 打印对新数据做滑动窗口小波分解的提示
new_wavelet_features = wavelet_decompose_sliding(  # 调用分解函数处理新数据特征（调用开始）
    new_data[features],  # 传入新数据的原始特征列
    target_col='相对湿度',  # 指定仅分解相对湿度列
    wavelet='db4',  # 使用与训练一致的 db4 小波基
    level=3,  # 分解层数与训练一致，为 3 层
    window_size=time_step  # 窗口大小取训练时的时间步长
)  # 函数调用结束

new_combined_features = new_wavelet_features  # 新数据组合特征取小波特征表（已含原始列）
new_combined_features = pd.DataFrame(  # 用训练集 scaler 对新数据特征归一化并重建 DataFrame
    feature_scaler.transform(new_combined_features),  # 用训练集拟合的 scaler 变换新数据，不再重新 fit
    columns=feature_names  # 列名保持与训练一致
)  # DataFrame 构造结束

y_new_original = new_data[target].values  # 取出新数据中目标的原始值数组

X_new = create_sequences(new_combined_features, time_step)  # 为新数据构造时序样本
y_new = y_new_original[time_step + pred_offset:]  # 按偏移量截取对应的新数据真实标签
X_new = X_new[:len(y_new)]  # 截断特征序列使其与标签数量一致

valid_mask = ~np.isnan(y_new)  # 用 ~isnan 生成标签有效的布尔掩码
if valid_mask.sum() < len(y_new):  # 若存在无效（NaN）标签
    print(f"警告: 新数据目标列中有 {(~valid_mask).sum()} 个 NaN，已过滤")  # 输出将过滤掉多少个 NaN 样本的警告

X_new = X_new[valid_mask]  # 用掩码过滤特征序列，只保留标签有效的样本
y_new = y_new[valid_mask]  # 用掩码过滤标签，去掉 NaN 值

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 把新数据特征转为 FloatTensor 并迁移到目标设备

print("\nGRU+小波+ECA模型预测新数据...")  # 打印模型开始预测新数据的提示
final_model.eval()  # 将模型切换为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据进行预测并把结果转回 CPU NumPy 数组

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 把新数据预测值反归一化回原始量纲
y_new_rescaled = y_new.reshape(-1, 1)  # 把新数据真实标签整理成 (n, 1) 列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R2
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MSE
rmse_new = np.sqrt(mse_new)  # 对新数据 MSE 开方得到 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印新数据评估的分隔线
print("新数据评估结果（GRU+小波+ECA）")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 再打印一条分隔线
print(f"新数据 R²:   {r2_new:.4f}")  # 输出新数据 R2
print(f"新数据 RMSE: {rmse_new:.4f}")  # 输出新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 输出新数据 MAE
print(f"{'=' * 60}")  # 打印评估结束分隔线

timestamps = new_data['timestamp'][time_step + pred_offset:]  # 按偏移量截取新数据对应的时间戳

plt.figure(figsize=(15, 5))  # 创建 15x5 英寸的新数据预测对比图画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 展平新数据真实值
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 展平新数据预测值

plt.plot(  # 绘制新数据真实值曲线（调用开始）
    timestamps[:len(y_new_rescaled_flat)],  # 取与真实值等长的时间戳切片
    y_new_rescaled_flat,  # 真实值序列
    label='真实相对湿度',  # 图例文字为“真实相对湿度”
    linewidth=2,  # 设置线宽为 2
    alpha=0.8  # 设置透明度为 0.8
)  # plot 调用结束
plt.plot(  # 绘制新数据预测值曲线（调用开始）
    timestamps[:len(y_new_pred_rescaled_flat)],  # 取与预测值等长的时间戳切片
    y_new_pred_rescaled_flat,  # 预测值序列
    label='GRU+小波+ECA预测',  # 图例文字为“GRU+小波+ECA预测”
    linewidth=2,  # 设置线宽为 2
    alpha=0.8  # 设置透明度为 0.8
)  # plot 调用结束
plt.xlabel('时间')  # 设置 x 轴标签为“时间”
plt.ylabel('相对湿度 ')  # 设置 y 轴标签为“相对湿度”
plt.title(f'新数据集相对湿度结果 (R²: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置标题并附带 R2 与 RMSE 指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
plt.savefig('new_data_prediction_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把新数据预测图保存为 png
print("\n新数据预测结果已保存: new_data_prediction_GRUWavelet_pytorch.png")  # 打印新数据预测图保存路径
plt.close()  # 关闭图形释放内存

print("\n========== 所有任务完成（GRU+小波+ECA模型） ==========")  # 打印所有任务完成的提示