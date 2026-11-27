"""
GRU模型测试脚本（支持小波变换）
用于加载训练好的GRU模型并进行预测
支持：GRU-Only模型、GRU+小波变换模型

本脚本是推理/评估入口：多编码读取测试数据与训练数据、缺失值处理、按检查点配置
自动识别模型类型（是否使用小波特征、是否无数据泄漏），构造对应特征与时间序列样本，
加载权重后输出 R²/MSE/RMSE/MAE，并绘制预测对比图与四联误差分析图。
若使用了小波特征，会对训练集与测试集分别做滑动窗口小波分解以避免未来信息泄漏。
"""
import pandas as pd  # 导入 pandas 库，用于表格数据读取与处理（别名 pd）
import numpy as np  # 导入 NumPy 库，用于数组与数值计算（别名 np）
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用 Agg 非交互式后端，允许在无界面环境保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口（别名 plt）
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于必要时重新拟合归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R²、均方误差、平均绝对误差
import torch  # 导入 PyTorch 主库，用于张量计算与模型加载
import torch.nn as nn  # 导入神经网络模块，用于定义各类模型结构
import pywt  # 小波变换库，用于 pywt.wavedec 对 THI 做多尺度分解
import os  # 导入标准库 os，用于判断模型文件是否存在

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU
print(f"使用设备: {device}")  # 打印使用的设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ===================== 模型定义 =====================
class ECALayer(nn.Module):  # 高效通道注意力模块（ECA-Net）
    """
    ECA-Net: Efficient Channel Attention
    通过1D卷积实现高效的通道注意力机制

    参数:
        channels: 输入通道数。
        gamma: 自适应计算卷积核大小的缩放系数（默认 2）。
        b: 卷积核大小计算的偏置项（默认 1）。
    前向返回:
        与输入同形状、各通道按注意力权重加权后的张量。
    """
    def __init__(self, channels, gamma=2, b=1):  # 初始化：channels 为输入通道数
        super(ECALayer, self).__init__()  # 调用父类构造函数
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应计算卷积核参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化：每通道压缩为 1 个值
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 一维卷积学习通道权重，两侧补零
        self.sigmoid = nn.Sigmoid()  # Sigmoid 将权重映射到 0~1
    
    def forward(self, x):  # 前向传播：对通道做注意力加权
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step)
        x_permuted = x.permute(0, 2, 1)  # 转置使通道维位于第 1 维
        
        # 全局平均池化
        y = self.avg_pool(x_permuted)  # 得到通道描述 (batch, channels, 1)
        
        # 1D卷积
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 一维卷积后转回原形状，得到通道注意力权重
        
        # Sigmoid激活
        y = self.sigmoid(y)  # 归一化权重到 0~1
        
        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将权重广播乘回各通道
        
        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 转回原维度后返回


class GRUOnlyModel(nn.Module):  # 仅使用 ECA 注意力的 GRU 预测模型
    """GRU-Only 模型 + ECA-Net注意力机制（与 gru.py 保持一致）

    参数:
        input_dim: 输入特征维度。
        time_step: 时间窗口长度。
        gru_hidden: GRU 隐藏维度，可为整数或逐层列表。
        gru_layers: GRU 层数（gru_hidden 为整数时生效）。
        dropout: GRU 之后的丢弃率。
    前向返回:
        (batch, 1) 的预测张量。
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):  # 初始化模型结构
        super(GRUOnlyModel, self).__init__()  # 调用父类构造函数
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层维度

        # ECA-Net注意力模块
        self.eca = ECALayer(channels=input_dim)  # 构建 ECA 注意力层
        
        # 若 gru_hidden 为列表，则逐层构建 GRU，每层hidden_size可不同
        if isinstance(gru_hidden, list):  # 若隐藏维度以列表给出
            hidden_sizes = gru_hidden  # 直接采用列表配置
        else:  # 否则所有层使用相同隐藏维度
            hidden_sizes = [gru_hidden] * gru_layers  # 复制 gru_layers 份

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多层 GRU
        in_size = input_dim  # 第一层输入维度等于特征数
        for i, h_size in enumerate(hidden_sizes):  # 逐层构建 GRU
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
        # ECA注意力增强
        x = self.eca(x)  # 对输入做通道注意力加权
        
        # 逐层GRU编码
        out = x  # 初始化 GRU 输入
        for gru in self.gru_layers_list:  # 逐层通过 GRU
            out, _ = gru(out)  # 取序列输出，忽略隐藏状态
            out = self.dropout(out)  # 施加 dropout
        out_last = out[:, -1, :]  # 取序列最后一个时间步的输出

        # 预测输出
        return self.fc(out_last)  # 经全连接层得到预测


class GRUWaveletModel(nn.Module):  # GRU + 小波特征 + ECA 注意力的预测模型
    """GRU + 小波变换 + ECA注意力机制

    参数:
        input_dim: 输入特征维度（小波组合后的特征数）。
        time_step: 时间窗口长度。
        gru_hidden: GRU 隐藏维度，可为整数或逐层列表。
        gru_layers: GRU 层数（gru_hidden 为整数时生效）。
        dropout: GRU 之后的丢弃率。
    前向返回:
        (batch, 1) 的预测张量。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):  # 初始化模型结构
        super(GRUWaveletModel, self).__init__()  # 调用父类构造函数
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层维度

        # ECA-Net注意力模块
        self.eca = ECALayer(channels=input_dim)  # 构建 ECA 注意力层
        
        # 若 gru_hidden 为列表，则逐层构建 GRU，每层hidden_size可不同
        if isinstance(gru_hidden, list):  # 若隐藏维度以列表给出
            hidden_sizes = gru_hidden  # 直接采用列表配置
        else:  # 否则所有层使用相同隐藏维度
            hidden_sizes = [gru_hidden] * gru_layers  # 复制 gru_layers 份

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多层 GRU
        in_size = input_dim  # 第一层输入维度等于特征数
        for i, h_size in enumerate(hidden_sizes):  # 逐层构建 GRU
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
        # ECA注意力增强
        x = self.eca(x)  # 对输入做通道注意力加权
        
        # 逐层GRU编码
        out = x  # 初始化 GRU 输入
        for gru in self.gru_layers_list:  # 逐层通过 GRU
            out, _ = gru(out)  # 取序列输出，忽略隐藏状态
            out = self.dropout(out)  # 施加 dropout
        out_last = out[:, -1, :]  # 取序列最后一个时间步的输出

        # 预测输出
        return self.fc(out_last)  # 经全连接层得到预测
# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始加载测试数据
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的编码列表
data = None  # 初始化读取结果变量
for enc in encodings:  # 逐个编码尝试
    try:  # 捕获解码异常
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 读取成功即跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 换下一个编码继续尝试

if data is None:  # 所有编码都失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常终止
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 按索引×5分钟生成时间戳（分钟为单位）
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 时间起点
# 特征配置
features = ['日龄','THI','目标温度','当前温度']  # 候选特征列表（默认顺序）
target = 'THI'  # 目标变量为 THI

# 过滤特征列（只保留数据中存在的列）
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征

print(f"使用特征: {features}")  # 打印实际使用的特征
print(f"数据形状: {data.shape}")  # 打印数据行列数

# 检查目标列是否存在
if target not in data.columns:  # 若缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带THI的CSV文件。")  # 抛出异常提示

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印处理标题
print(f"{'='*60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前数据行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计表头
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅处理数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比（百分比）
        if nan_count > 0:  # 仅打印存在缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与占比

# 定义连续变量和离散变量
continuous_vars = ['THI','目标温度','当前温度']  # 连续变量：线性插值 + 前后填充
discrete_vars = ['日龄']  # 离散变量：前后填充

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 该列存在且有缺失时才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录处理前缺失值个数
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值，双向填充内部缺失
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].ffill()  # 前向填充开头残留缺失
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].bfill()  # 后向填充结尾残留缺失
        
        after_nan = data[col].isna().sum()  # 记录处理后缺失值个数
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 该列存在且有缺失时才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失值个数
        
        # 前向填充
        data[col] = data[col].ffill()  # 前向填充
        # 后向填充
        data[col] = data[col].bfill()  # 后向填充
        
        after_nan = data[col].isna().sum()  # 记录填充后缺失值个数
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列剩余缺失总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除特征/目标中含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后数据行列数
print(f"{'='*60}\n")  # 打印结束分隔线

# 重置索引
data = data.reset_index(drop=True)  # 删除缺失行后重置索引，保证序列构造连续

# ===================== 小波变换函数（无数据泄漏版本，仅分解THI） =====================
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
        if col == 'timestamp':  # 时间戳列不参与分解
            continue  # 跳过时间戳列

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


# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 提示开始加载模型
# 优先加载无数据泄漏版本的模型
USE_WAVELET = False  # 是否使用小波特征（加载检查点后按配置更新）
NO_LEAKAGE = False  # 是否为无数据泄漏版本（加载检查点后按配置更新）
# 尝试多个可能的模型路径
model_paths = [  # 候选检查点路径列表
    'Model_GRUWavelet_no_leakage_pytorch.pth',  # 无数据泄漏的小波模型路径
]

checkpoint = None  # 初始化检查点变量
MODEL_PATH = None  # 初始化实际加载的模型路径

for path in model_paths:  # 遍历候选路径
    if os.path.exists(path):  # 仅处理存在的文件
        try:  # 捕获加载异常
            checkpoint = torch.load(path, map_location=device, weights_only=False)  # 加载检查点到指定设备
            MODEL_PATH = path  # 记录实际加载的路径
            
            # 判断模型类型
            if 'no_leakage' in path or checkpoint.get('no_data_leakage', False):  # 无数据泄漏的小波模型
                USE_WAVELET = True  # 启用小波特征
                NO_LEAKAGE = True  # 标记为无泄漏版本
                print(f"成功加载小波变换模型（无数据泄漏版本）: {MODEL_PATH}")  # 打印加载信息
            elif 'Wavelet' in path:  # 旧版小波模型
                USE_WAVELET = True  # 启用小波特征
                NO_LEAKAGE = False  # 可能存在数据泄漏
                print(f"成功加载小波变换模型（旧版本）: {MODEL_PATH}")  # 打印加载信息
                print("⚠️  警告：该模型可能存在数据泄漏问题")  # 打印泄漏警告
            else:  # 普通 GRU 模型
                USE_WAVELET = False  # 不启用小波特征
                NO_LEAKAGE = False  # 非无泄漏版本
                print(f"成功加载普通GRU模型: {MODEL_PATH}")  # 打印加载信息
            break  # 加载成功即跳出循环
        except Exception as e:  # 捕获加载过程中的任意异常
            print(f"尝试加载 {path} 失败: {e}")  # 打印失败原因
            continue  # 继续尝试下一个路径

if checkpoint is None:  # 没有任何检查点加载成功
    raise FileNotFoundError(f"无法找到模型文件。尝试的路径: {model_paths}")  # 抛出异常终止

input_dim = checkpoint['input_dim']  # 从检查点读取输入维度
time_step = checkpoint['time_step']  # 从检查点读取时间步长

# ⚠️ 重要：从模型中获取训练时使用的原始特征列表
if 'original_features' in checkpoint:  # 检查点保存了原始特征列表
    # 如果模型保存了原始特征列表，使用它
    model_features = checkpoint['original_features']  # 直接取训练时的原始特征
    print(f"\n从模型加载原始特征列表: {model_features}")  # 打印加载的特征
else:  # 检查点未保存原始特征列表
    # 如果没有保存，尝试从feature_names反推（去掉小波后缀）
    feature_names_saved = checkpoint.get('feature_names', None)  # 尝试读取保存的特征名
    if feature_names_saved:  # 存在特征名列表
        # 从小波特征名中提取原始特征名
        model_features = []  # 初始化反推结果列表
        for fname in feature_names_saved:  # 遍历保存的特征名
            if '_approx' in fname:  # 近似系数特征名形如“THI_approx”
                original_name = fname.replace('_approx', '')  # 去掉后缀得到原始特征名
                if original_name not in model_features:  # 避免重复添加
                    model_features.append(original_name)  # 加入原始特征列表
        print(f"\n从小波特征名反推原始特征: {model_features}")  # 打印反推结果
    else:  # 两种信息都没有
        # 如果都没有，使用默认特征（但会给出警告）
        model_features = features  # 退而使用当前测试数据的特征列表
        print(f"\n⚠️  警告：模型未保存特征列表，使用测试数据的特征: {model_features}")  # 打印警告

# 检查测试数据是否包含所需特征
missing_features = [f for f in model_features if f not in data.columns]  # 找出测试数据中缺失的特征
if missing_features:  # 若存在缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常

# 使用模型训练时的特征
features = model_features  # 覆盖为模型训练时的特征列表
print(f"\n最终使用特征: {features}")  # 打印最终特征

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的编码列表
train_data = None  # 初始化训练数据变量
for enc in encodings:  # 逐个编码尝试
    try:  # 捕获解码异常
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功即跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 换下一个编码继续尝试
if train_data is None:  # 所有编码都失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止

# 检查训练数据中是否包含所需特征
print(f"\n训练数据列名: {list(train_data.columns)}")  # 打印训练数据所有列名
missing_in_train = [f for f in features if f not in train_data.columns]  # 找出训练数据中缺失的特征
if missing_in_train:  # 若训练数据缺少特征
    print(f"\n⚠️ 警告：训练数据缺少以下特征: {missing_in_train}")  # 打印警告
    print(f"可用的特征: {[col for col in train_data.columns if col != 'timestamp']}")  # 打印可用特征
    # 只使用训练数据中存在的特征
    features = [f for f in features if f in train_data.columns]  # 过滤为训练数据中存在的特征
    print(f"调整后使用的特征: {features}")  # 打印调整后的特征
    
    if len(features) == 0:  # 没有任何可用特征
        raise ValueError("训练数据中没有可用的特征！")  # 抛出异常终止

if USE_WAVELET:  # 使用小波特征分支
    print("\n使用小波变换特征...")  # 打印分支提示
    
    # 获取小波参数
    wavelet_type = checkpoint.get('wavelet', 'db4')  # 从检查点读取小波基
    wavelet_level = checkpoint.get('level', 3)  # 从检查点读取分解层数
    wavelet_target_col = checkpoint.get('wavelet_target_col', 'THI')  # 获取小波分解的目标列
    feature_names = checkpoint.get('feature_names', None)  # 读取训练时保存的特征名
    
    print(f"  小波基函数: {wavelet_type}")  # 打印小波基
    print(f"  分解层数: {wavelet_level}")  # 打印分解层数
    print(f"  小波分解目标列: {wavelet_target_col}")  # 打印分解目标列
    print(f"  数据泄漏防护: {'是' if NO_LEAKAGE else '否'}")  # 打印是否无泄漏
    
    # 从checkpoint中获取小波窗口大小，与训练时保持一致（训练时为time_step=48）
    wavelet_window_size = checkpoint.get('wavelet_window', time_step)  # 读取训练时的小波窗口
    print(f"  小波窗口大小: {wavelet_window_size}（与训练时一致）")  # 打印窗口大小
    
    # 使用滑动窗口小波分解（无数据泄漏，仅分解THI）
    print("\n对训练数据进行滑动窗口小波分解（仅分解THI）...")  # 提示开始训练集分解
    train_wavelet_features = wavelet_decompose_sliding(  # 训练集滑动窗口小波分解
        train_data[features],  # 输入训练集特征
        target_col=wavelet_target_col,  # 分解目标列
        wavelet=wavelet_type,  # 小波基
        level=wavelet_level,  # 分解层数
        window_size=wavelet_window_size  # 窗口大小
    )
    print("\n对测试数据进行滑动窗口小波分解（仅分解THI）...")  # 提示开始测试集分解
    test_wavelet_features = wavelet_decompose_sliding(  # 测试集滑动窗口小波分解
        data[features],  # 输入测试集特征
        target_col=wavelet_target_col,  # 分解目标列
        wavelet=wavelet_type,  # 小波基
        level=wavelet_level,  # 分解层数
        window_size=wavelet_window_size  # 窗口大小
    )
    # 直接使用小波分解后的特征（已包含原始特征和小波特征）
    train_combined = train_wavelet_features  # 训练集组合特征
    test_combined = test_wavelet_features  # 测试集组合特征
    # 归一化（如果模型保存了scaler，直接使用）
    if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 检查点含归一化器
        print("\n使用模型保存的归一化器...")  # 打印提示
        feature_scaler = checkpoint['feature_scaler']  # 直接取保存的特征归一化器
        target_scaler = checkpoint['target_scaler']  # 直接取保存的目标归一化器
    else:  # 未保存归一化器
        print("\n使用训练数据重新拟合归一化器...")  # 打印提示
        feature_scaler = MinMaxScaler()  # 新建特征归一化器
        target_scaler = MinMaxScaler()  # 新建目标归一化器
        feature_scaler.fit(train_combined)  # 用训练集组合特征拟合
        target_scaler.fit(train_data[[target]])  # 用训练集目标拟合
    # 保存原始列名
    original_columns = test_combined.columns  # 记录归一化前的列名
    test_combined = pd.DataFrame(  # 变换测试集组合特征
        feature_scaler.transform(test_combined),  # 归一化数值
        columns=original_columns  # 保持原列名
    )
    print(f"  特征数量: {len(test_combined.columns)}")  # 打印特征数量
    print(f"  特征列表: {list(test_combined.columns)}")  # 打印特征列表
else:  # 不使用小波特征的分支
    print("\n使用原始特征...")  # 打印分支提示
    # 普通模型：仅使用原始特征
    feature_scaler = MinMaxScaler()  # 新建特征归一化器
    target_scaler = MinMaxScaler()  # 新建目标归一化器
    feature_scaler.fit(train_data[features])  # 用训练集特征拟合
    target_scaler.fit(train_data[[target]])  # 用训练集目标拟合
    test_combined = pd.DataFrame(  # 变换测试数据特征
        feature_scaler.transform(data[features]),  # 归一化数值
        columns=features  # 保持列名
    )
    print(f"  特征数量: {len(features)}")  # 打印特征数量
# ===================== 序列构造 =====================
pred_offset = 6   # 与训练脚本 gru.py 保持一致（6步 × 5分钟 = 30分钟）
def create_sequences(data, time_step):  # 创建时间序列样本（对单个数据集构造，不涉及训练/测试划分）
    """
    创建时间序列样本
    注意：这里是对单个数据集（机器人新数据）进行序列构造，不涉及训练/测试划分。

    参数:
        data: 特征表（行代表时间步）。
        time_step: 输入窗口长度。
    返回:
        ndarray: 形状为 (样本数, time_step, 特征数) 的三维数组。
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，保证目标位置不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行作为一个样本
    return np.array(sequences)  # 转换为三维数组返回
print("\n创建序列数据...")  # 提示创建序列数据
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 对目标列做归一化（二维）
# 创建序列
X_test = create_sequences(test_combined, time_step)  # 构造测试输入序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 取对齐的归一化目标并拉平
y_test = y_test[:len(X_test)]  # 确保长度匹配
print(f"\n测试数据形状:")  # 打印测试数据形状标题
print(f"  X_test: {X_test.shape}")  # 打印输入序列形状
print(f"  y_test: {y_test.shape}")  # 打印目标形状
# ===================== 构建模型 =====================
if USE_WAVELET:  # 使用小波特征，构建 GRUWaveletModel
    model = GRUWaveletModel(  # 实例化 GRU+小波+ECA 模型
        input_dim=input_dim,  # 输入维度
        time_step=time_step,  # 时间步长
        gru_hidden=128,  # GRU 隐藏维度
        gru_layers=1,  # GRU 层数
        dropout=0.2  # dropout 比例
    ).to(device)  # 将模型移至设备
else:  # 使用原始特征，构建 GRUOnlyModel
    model = GRUOnlyModel(  # 实例化 GRU-Only+ECA 模型
        input_dim=input_dim,  # 输入维度
        time_step=time_step,  # 时间步长
        gru_hidden=128,  # GRU 隐藏维度
        gru_layers=1,  # GRU 层数
        dropout=0.2  # dropout 比例
    ).to(device)  # 将模型移至设备

model.load_state_dict(checkpoint['model_state_dict'])  # 加载训练好的权重
model.eval()  # 切换评估模式
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印参数总量
# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始预测
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试输入转张量并送设备
with torch.no_grad():  # 关闭梯度计算
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并转回 NumPy
# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 真实目标反归一化到原始尺度
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 预测值反归一化到原始尺度
# 计算评估指标
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差
rmse = np.sqrt(mse)  # 均方误差开方得 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差
if USE_WAVELET:  # 根据模型类型生成名称
    model_name = "GRU+Wavelet+ECA (No Leakage)" if NO_LEAKAGE else "GRU+Wavelet+ECA"  # 小波模型名称
else:  # 普通模型
    model_name = "GRU-Only"  # 纯 GRU 模型名称
print("\n" + "=" * 60)  # 打印分隔线
print(f"{model_name} Model Test Results")  # 打印带模型名的评估结果标题
print("=" * 60)  # 再次打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R²
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线
# ===================== 结果可视化 =====================
plot_length = min(1600, len(y_test_rescaled))  # 限制绘图点数，最多 1600
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的预测值

# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成横轴步数索引

plt.figure(figsize=(15, 8))  # 创建预测对比图画布
y_test_flat = y_test_plot.flatten()  # 真实值拉平为一维
y_pred_flat = y_pred_plot.flatten()  # 预测值拉平为一维
plt.tick_params(axis='both', labelsize=16)  # 设置刻度字体大小
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线（蓝色）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值曲线（橙色）
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签
plt.ylabel('THI', fontsize=16)  # 设置纵轴标签
# plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.tight_layout()  # 紧凑布局
plt.text(0.5, 0.95, '(f)', transform=plt.gca().transAxes,  # 在图上添加子图编号“(f)”
         fontsize=16, fontweight='bold', va='top', ha='center')
plt.tight_layout()  # 再次紧凑布局
if USE_WAVELET:  # 按模型类型选择预测图文件名
    output_filename = 'test_GRUWavelet_no_leakage_prediction.png' if NO_LEAKAGE else 'test_GRUWavelet_prediction.png'  # 小波模型预测图名
else:  # 普通模型
    output_filename = 'test_GRUOnly_prediction.png'  # 纯 GRU 模型预测图名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 保存预测图，600 dpi
print(f"\n预测结果图已保存: {output_filename}")  # 打印保存提示
plt.close()  # 关闭画布释放内存

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算全体样本的预测误差（预测-真实）

# 仅用于绘图：取前300个点
max_points = min(300, len(errors))  # 限制误差分析绘图点数，最多 300
y_test_plot_err = y_test_rescaled[:max_points].flatten()  # 误差分析用真实值（前 300）
y_pred_plot_err = y_pred_rescaled[:max_points].flatten()  # 误差分析用预测值（前 300）
errors_plot = y_pred_plot_err - y_test_plot_err  # 误差分析用误差序列

print(f"\n误差分析图使用前 {max_points} 个数据点")  # 打印实际绘图点数

plt.figure(figsize=(15, 10))  # 创建四联误差分析画布

# 子图1: True value vs Predicted value（前300个点）
plt.subplot(2, 2, 1)  # 选中第 1 个子图
plt.scatter(  # 绘制真值-预测值散点图
    y_test_plot_err,  # 横轴：真实值
    y_pred_plot_err,  # 纵轴：预测值
    alpha=0.35,  # 点透明度
    s=45,  # 点大小
    color='#b58f86',  # 点颜色
    label='Predicted value'  # 图例文字
)
min_val = min(y_test_plot_err.min(), y_pred_plot_err.min())  # 取真实/预测的最小值作为坐标下界
max_val = max(y_test_plot_err.max(), y_pred_plot_err.max())  # 取真实/预测的最大值作为坐标上界
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')  # 绘制 y=x 理想参考线
plt.xlabel('True value', fontsize=11)  # 设置横轴标签
plt.ylabel('Predicted value', fontsize=11)  # 设置纵轴标签
plt.title('True value vs Predicted value', fontsize=12)  # 设置子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.4)  # 开启网格
margin = (max_val - min_val) * 0.05  # 计算坐标轴 5% 边距
plt.xlim(min_val - margin, max_val + margin)  # 设置横轴范围
plt.ylim(min_val - margin, max_val + margin)  # 设置纵轴范围

# 子图2: 误差分布直方图（前300个点）
plt.subplot(2, 2, 2)  # 选中第 2 个子图
plt.hist(errors_plot, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差直方图（50 个柱子）
plt.xlabel('Prediction Error', fontsize=11)  # 设置横轴标签
plt.ylabel('Frequency', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考竖线
plt.grid(True, alpha=0.3, axis='y')  # 开启纵轴网格

# 子图3: 误差时序图（前300个点）
plt.subplot(2, 2, 3)  # 选中第 3 个子图
error_steps = np.arange(len(errors_plot))  # 生成误差时序的横轴索引
plt.plot(error_steps, errors_plot, alpha=0.7, linewidth=1)  # 绘制误差随时间变化曲线
plt.xlabel('Timestep', fontsize=11)  # 设置横轴标签
plt.ylabel('Prediction Error', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Time Series (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考横线
plt.grid(True, alpha=0.3)  # 开启网格

# 子图4: 误差箱线图（前300个点）
plt.subplot(2, 2, 4)  # 选中第 4 个子图
plt.boxplot([errors_plot], tick_labels=[model_name])  # 绘制误差箱线图，标签用模型名
plt.ylabel('Prediction Error', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Statistics Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考横线
plt.grid(True, alpha=0.3, axis='y')  # 开启纵轴网格

plt.tight_layout()  # 紧凑布局
if USE_WAVELET:  # 按模型类型选择误差图文件名
    error_filename = 'test_GRUWavelet_no_leakage_error_analysis.png' if NO_LEAKAGE else 'test_GRUWavelet_error_analysis.png'  # 小波模型误差图名
else:  # 普通模型
    error_filename = 'test_GRUOnly_error_analysis.png'  # 纯 GRU 模型误差图名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 保存误差分析图，600 dpi
print(f"误差分析图已保存: {error_filename}")  # 打印保存提示
plt.close()  # 关闭画布释放内存

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("Error Statistics")  # 打印统计标题
print("=" * 60)  # 再次打印分隔线
print(f"Mean Error:     {np.mean(errors):.4f}")  # 打印误差均值
print(f"Std Deviation:  {np.std(errors):.4f}")  # 打印误差标准差
print(f"Max Error:      {np.max(errors):.4f}")  # 打印误差最大值
print(f"Min Error:      {np.min(errors):.4f}")  # 打印误差最小值
print(f"Median Error:   {np.median(errors):.4f}")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\nTest Completed!")  # 打印全部完成提示




