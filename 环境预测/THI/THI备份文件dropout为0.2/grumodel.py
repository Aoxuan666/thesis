"""
GRU模型测试脚本（支持小波变换）
用于加载训练好的GRU模型并进行预测
支持：GRU-Only模型、GRU+小波变换模型

本脚本只做推理与评估，不参与训练：
1. 读取“部分温湿度数据.csv”，做时间戳生成与缺失值填充；
2. 根据 .pth 检查点中的配置，重建 GRU-Only 或 GRU+小波+ECA 模型并加载权重；
3. 用模型训练时保存的归一化器（或重新拟合）处理数据，构造序列并预测 THI；
4. 输出 R2/MSE/RMSE/MAE 指标，绘制预测对比图、小波分解效果图与误差分析图。
注意：模型与特征列表以检查点为准，避免测试阶段特征顺序与训练不一致。
"""
import pandas as pd  # 导入 pandas 并简写为 pd，用于读取 CSV 与构造 DataFrame
import numpy as np  # 导入 NumPy 并简写为 np，用于数组构造与数值计算
import matplotlib  # 导入 matplotlib 基础模块，以便在使用 pyplot 前切换后端
matplotlib.use('Agg')  # 使用非交互式 Agg 后端，无图形界面时也能保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 并简写为 plt，用于绘制预测与误差图
from sklearn.preprocessing import MinMaxScaler  # 从 sklearn.preprocessing 导入 MinMaxScaler，用于特征和目标归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 从 sklearn.metrics 导入 R2、MSE、MAE 三个回归评估指标
import torch  # 导入 PyTorch 主库，提供张量与设备管理能力
import torch.nn as nn  # 导入 torch.nn 并简写为 nn，用于定义网络层
import pywt  # 小波变换库；无泄漏版本用 wavedec，旧版还用 upcoef 重构分量
import os  # 导入 os 模块，用 os.path.exists 判断模型文件是否存在

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测 CUDA 是否可用，可用则选 GPU，否则选 CPU
print(f"使用设备: {device}")  # 打印当前使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图默认字体为黑体，保证中文标签可正常显示
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号变成方块


# ===================== 模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    通过1D卷积实现高效的通道注意力机制
    参数:
        channels: 输入特征通道数
        gamma: 推导卷积核大小时的缩放系数
        b: 推导卷积核大小时的偏置项
    返回:
        无显式返回值；forward 会对输入做通道注意力重标定
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 层：按通道数自适应确定 1D 卷积核大小。
        参数 channels 为通道数，gamma/b 用于推导奇数卷积核。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 根据通道数自适应计算卷积核参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数，便于对称 padding
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 定义自适应全局平均池化，把时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 定义 1D 卷积层，通过相邻通道交互学习注意力权重
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活，把注意力权重映射到 0~1
    
    def forward(self, x):
        """前向传播：对 (batch, time, channels) 输入施加通道注意力。
        参数 x 为输入张量，返回同形状的加权张量。
        """
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step)
        x_permuted = x.permute(0, 2, 1)  # 把输入维度转置为 (batch, channels, time_step)
        
        # 全局平均池化
        y = self.avg_pool(x_permuted)  # 沿时间维做全局平均池化，得到每个通道的全局描述
        
        # 1D卷积
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 用 1D 卷积学习通道注意力，并转回原维度顺序
        
        # Sigmoid激活
        y = self.sigmoid(y)  # Sigmoid 把权重归一化到 0~1 之间
        
        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 把通道权重广播乘回原特征，完成特征重标定
        
        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 转回 (batch, time_step, channels) 并返回


class GRUOnlyModel(nn.Module):
    """GRU-Only 模型 + ECA-Net 注意力机制。

    仅使用原始特征，输入先经 ECA 增强，再由 GRU 编码，最后用全连接层输出单值预测。
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):
        """初始化 GRU-Only + ECA 模型。
        参数: input_dim 输入维度，time_step 时间步，gru_hidden 隐藏维度，
              gru_layers 层数（大于 1 时启用 dropout），dropout 丢弃率。
        """
        super(GRUOnlyModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 记录时间步长配置
        self.input_dim = input_dim  # 记录输入特征维度
        self.gru_hidden = gru_hidden  # 记录 GRU 隐藏层维度

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块，在 GRU 之前增强输入特征
        
        self.gru = nn.GRU(  # 定义 GRU 层（调用开始）
            input_size=input_dim,  # GRU 输入维度等于特征数
            hidden_size=gru_hidden,  # GRU 隐藏单元数
            num_layers=gru_layers,  # GRU 堆叠层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅当层数大于 1 时才在 GRU 内部启用 dropout
            batch_first=True,  # batch_first=True 表示输入形状为 (batch, time, feature)
            bidirectional=False  # 使用单向 GRU
        )  # GRU 定义结束

        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接层，把隐藏状态映射为单个预测值

    def forward(self, x):
        """前向传播：ECA 增强 -> GRU 编码 -> 取末时刻 -> 全连接回归。
        参数 x 形状 (batch, time_step, input_dim)，返回 (batch, 1) 预测值。
        """
        # ECA注意力增强输入特征
        x = self.eca(x)  # 先用 ECA 模块增强输入特征
        
        # GRU编码
        out, h = self.gru(x)  # GRU 前向传播，同时得到输出序列与最终隐藏状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # 取序列最后一个时间步的输出作为整段序列的表示
        
        return self.fc(out_last)  # 经全连接层输出最终预测值


class GRUWaveletModel(nn.Module):
    """GRU + 小波变换 + ECA 注意力机制。

    小波特征已在数据预处理阶段生成，这里模型结构与 GRU-Only 类似但支持多层不同隐藏维度。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):
        """初始化 GRU+小波+ECA 模型。
        参数: input_dim 输入维度，time_step 时间步，gru_hidden 隐藏维度（可为列表），
              gru_layers 层数，dropout 丢弃率；逐层构建 GRU 并加入 ECA 与全连接层。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 记录时间步长配置
        self.input_dim = input_dim  # 记录输入特征维度
        self.gru_hidden = gru_hidden  # 记录隐藏层配置

        # ECA-Net注意力模块
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 通道注意力模块
        
        # 若 gru_hidden 为列表，则逐层构建 GRU，每层hidden_size可不同
        if isinstance(gru_hidden, list):  # 判断隐藏层配置是否为列表（支持逐层不同维度）
            hidden_sizes = gru_hidden  # 是列表则直接作为各层隐藏维度
        else:  # 否则视为单一整数
            hidden_sizes = [gru_hidden] * gru_layers  # 把同一隐藏维度重复 gru_layers 次

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多层 GRU，确保参数被正确注册
        in_size = input_dim  # 记录当前层的输入维度，初值为输入特征数
        for i, h_size in enumerate(hidden_sizes):  # 逐层遍历隐藏维度配置
            self.gru_layers_list.append(  # 把新建的 GRU 加入 ModuleList（调用开始）
                nn.GRU(  # 实例化单层 GRU（调用开始）
                    input_size=in_size,  # 输入维度为上一层输出维度
                    hidden_size=h_size,  # 当前层隐藏单元数
                    num_layers=1,  # 每次只建一层，多层由循环实现
                    dropout=0.0,  # 单层 GRU 内部不加 dropout
                    batch_first=True,  # batch_first=True，输入按 batch 优先排列
                    bidirectional=False  # 单向 GRU
                )  # GRU 实例化结束
            )  # append 调用结束
            in_size = h_size  # 把下一层输入维度更新为当前隐藏维度

        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层，抑制过拟合
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 定义全连接层，输入为最后一层隐藏维度

    def forward(self, x):
        """前向传播：ECA -> 逐层 GRU -> 末时刻全连接回归。
        参数 x 形状 (batch, time_step, input_dim)，返回 (batch, 1) 预测值。
        """
        # ECA注意力增强
        x = self.eca(x)  # 用 ECA 注意力增强输入
        
        # 逐层GRU编码
        out = x  # 把增强后的特征作为 GRU 输入
        for gru in self.gru_layers_list:  # 逐层执行 GRU 前向传播
            out, _ = gru(out)  # 取当前 GRU 层的输出序列，忽略隐藏状态
            out = self.dropout(out)  # 对该层输出施加 Dropout
        out_last = out[:, -1, :]  # 取最后时刻输出作为序列汇总

        # 预测输出
        return self.fc(out_last)  # 全连接层输出预测值
# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印“加载测试数据”的提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义候选编码列表，依次尝试解码
data = None  # 初始化数据变量为空
for enc in encodings:  # 逐个尝试候选编码
    try:  # try 块捕获解码异常
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据 CSV
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功读取所用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 继续尝试下一种编码

if data is None:  # 若所有编码都读取失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出无法读取测试数据的异常
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 用行索引乘 5 表示时间间隔（单位分钟），生成时间戳（调用开始）
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 以指定时刻为起点，完成时间戳生成
# 特征配置
features = ['日龄','THI','目标温度','当前温度']  # 定义候选输入特征列名列表
target = 'THI'  # 指定预测目标列为 THI

# 过滤特征列（只保留数据中存在的列）
features = [f for f in features if f in data.columns]  # 只保留测试数据中实际存在的特征列

print(f"使用特征: {features}")  # 输出实际使用的特征列表
print(f"数据形状: {data.shape}")  # 输出测试数据的行数与列数

# 检查目标列是否存在
if target not in data.columns:  # 若测试数据中缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带THI的CSV文件。")  # 抛出 KeyError 提示需要带 THI 的 CSV

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印一行分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'='*60}")  # 再打印一行分隔线
print(f"原始数据形状: {data.shape}")  # 输出原始数据的行数与列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计的标题
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 只统计数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计当前列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占样本总数的百分比
        if nan_count > 0:  # 只打印确有缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 输出该列缺失数量与占比

# 定义连续变量和离散变量
continuous_vars = ['日龄','THI','目标温度','当前温度']  # 连续变量，用线性插值+前后填充处理
discrete_vars = ['日龄']  # 离散变量，仅用前后向填充处理

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且有缺失值的连续列
        print(f"\n处理 {col}:")  # 打印当前正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 用线性插值填补缺失值，双向填充首尾
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].ffill()  # 再用前向填充补齐开头剩余的缺失值
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].bfill()  # 最后用后向填充补齐结尾剩余的缺失值
        
        after_nan = data[col].isna().sum()  # 统计填充后剩余的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 输出填充前的缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 输出填充后的缺失值数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且有缺失值的离散列
        print(f"\n处理 {col}:")  # 打印当前正在处理的离散列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 前向填充
        data[col] = data[col].ffill()  # 用前一个有效值填充离散列的缺失
        # 后向填充
        data[col] = data[col].bfill()  # 用后一个有效值填充开头的缺失
        
        after_nan = data[col].isna().sum()  # 统计填充后剩余的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 输出填充前的缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 输出填充后的缺失值数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印将删除含缺失值行的警告
    data = data.dropna(subset=features + [target])  # 删除特征或目标为 NaN 的样本行

print(f"\n处理后数据形状: {data.shape}")  # 输出清洗后的数据规模
print(f"{'='*60}\n")  # 打印缺失值处理阶段结束的分隔线

# 重置索引
data = data.reset_index(drop=True)  # 重置索引并丢弃原索引，保证后续按位置切片不错位

# ===================== 小波变换函数（无数据泄漏版本，仅分解THI） =====================
def wavelet_decompose_sliding(data, target_col='THI', wavelet='db4', level=3, window_size=None, plot_decomposition=False, plot_feature='THI'):
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（THI）进行小波分解，其他列保持原样
    每个时间点只使用历史数据
    参数:
        data: 输入数据 (DataFrame)
        target_col: 需要进行小波分解的列名（默认'THI'）
        wavelet: 小波基函数 ('db4', 'sym5', 'coif3' 等)
        level: 分解层数
        window_size: 滑动窗口大小，None则自动计算
        plot_decomposition: 是否绘制小波分解效果图
        plot_feature: 要绘制的特征名称（默认为'THI'）
    返回:
        分解后的特征 (DataFrame)
    参数:
        data: 输入 DataFrame
        target_col: 需要分解的列名（默认'THI'）
        wavelet: 小波基函数（'db4' 等）
        level: 分解层数
        window_size: 滑动窗口大小，None 时取 min(36, len(data))
        plot_decomposition: 是否绘制分解效果图
        plot_feature: 绘图所用的特征名
    返回:
        含原始列与小波系数列的 DataFrame
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印开始分解的提示与目标列名
    print(f"小波基函数: {wavelet}")  # 打印所用小波基函数
    print(f"分解层数: {level}")  # 打印分解层数
    # 自动计算窗口大小：确保足够大以进行有效的小波分解
    if window_size is None:  # 若调用时未指定窗口大小
        window_size = min(36, len(data))  # 使用36作为窗口大小，与训练脚本的时间步设置保持一致
    
    print(f"滑动窗口大小: {window_size}")  # 输出最终使用的滑动窗口长度
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空 DataFrame，存放小波特征
    
    # 用于绘图的数据
    plot_data = {}  # 初始化用于绘图的数据字典
    
    for col in data.columns:  # 遍历数据的每一列
        if col == 'timestamp':  # 跳过 timestamp 时间戳列
            continue  # 进入下一列，不处理时间戳
        
        signal = data[col].values  # 取出当前列的值作为一维信号
        
        # 只对目标列进行小波分解
        if col == target_col:  # 只有目标列（THI）才做小波分解
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 初始化近似系数列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数各建一个空列表
            
            for i in range(len(signal)):  # 逐时间点滑动窗口，只用历史信息避免泄漏
                # 只使用当前时间点之前的数据（包括当前点）
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证不引用未来数据
                window_signal = signal[start_idx:i+1]  # 截取从起点到当前点（含）的历史信号
                
                # 如果窗口太小，用边界值填充到window_size
                if len(window_signal) < window_size:  # 若历史数据不足一个完整窗口
                    pad_width = window_size - len(window_signal)  # 计算需要向前填充的点数
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值（mode=edge）把窗口补齐到指定长度
                
                # 小波分解
                try:  # try 块：尝试对窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 pywt.wavedec 进行 level 层小波分解
                    
                    # 取最后一个值作为当前时间点的特征
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为当前时刻特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层细节系数末值，若为空则补 0.0
                except Exception as e:  # 捕获分解过程中出现的异常
                    # 如果分解失败，使用原始值
                    approx_list.append(window_signal[-1])  # 分解失败时用窗口最后一个原始值代替近似系数
                    for j in range(level):  # 遍历每一层细节系数
                        detail_lists[j].append(0.0)  # 失败层用 0.0 填充细节系数
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 把近似系数序列写入结果表，列名为“列名_approx”
            for j in range(level):  # 逐层写入细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 把第 j+1 层细节系数写入结果表
            
            # 保存指定特征的数据用于绘图
            if plot_decomposition and col == plot_feature:  # 若需要绘图且当前列正是绘图目标特征
                plot_data['col_name'] = col  # 在绘图字典中记录列名
                plot_data['original'] = signal  # 保存原始信号用于绘图
                plot_data['approx'] = np.array(approx_list)  # 保存近似系数数组
                plot_data['details'] = [np.array(detail_lists[j]) for j in range(level)]  # 把各层细节系数转成数组保存
                print(f"已保存 '{col}' 的小波分解数据用于绘图")  # 打印已保存绘图数据的提示
        else:  # 当前列不是需要分解的目标列
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列保持原样不分解
            wavelet_features[col] = signal  # 把原始信号直接写入结果表
    
    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 输出分解后特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 输出来自目标列的小波特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 输出保持不变的原始特征数量
    
    # 绘制小波分解效果图
    if plot_decomposition and plot_data:  # 若需要绘图且已收集到绘图数据
        plot_wavelet_decomposition(plot_data, wavelet, level)  # 调用绘图函数绘制 SCI 风格分解图
    elif plot_decomposition and not plot_data:  # 若需要绘图但没有找到目标特征
        print(f"⚠️  警告：未找到特征 '{plot_feature}'，无法绘制分解图")  # 输出未找到特征的警告
    
    return wavelet_features  # 返回含原始列与小波特征列的 DataFrame


def plot_wavelet_decomposition(plot_data, wavelet, level):
    """
    Plot wavelet decomposition for SCI publication
    Parameters:
        plot_data: Dictionary containing original signal and decomposition results
        wavelet: Wavelet basis function name
        level: Decomposition level
    返回:
        无显式返回值；把 SCI 风格的分解图保存为 PNG 与 PDF 两个文件
    """
    # Set publication-quality parameters
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置 SCI 出版风格的英文字体 Times New Roman
    plt.rcParams['font.size'] = 11  # 设置全局字号为 11
    plt.rcParams['axes.linewidth'] = 1.2  # 设置坐标轴边框线宽为 1.2
    
    col_name = plot_data['col_name']  # 从绘图字典取出列名
    original = plot_data['original']  # 取出原始信号
    approx = plot_data['approx']  # 取出近似系数
    details = plot_data['details']  # 取出各层细节系数列表
    
    # Plot first 1000 points for better visualization
    plot_length = min(800, len(original))  # 取前 800 个点绘图，避免曲线过于密集
    x_axis = np.arange(plot_length)  # 生成对应的横轴序号
    
    # Create subplots
    fig, axes = plt.subplots(level + 2, 1, figsize=(12, 10))  # 创建 level+2 行 1 列画布（原始 + 近似 + 各层细节）
    
    # Plot original signal
    axes[0].plot(x_axis, original[:plot_length], linewidth=1.2, color='#2C3E50')  # 第 1 个子图绘制原始信号，颜色为深蓝灰
    axes[0].set_title('(a) Original Signal: Temperature Deviation',  # 设置原始信号子图标题（调用开始）
                      fontsize=13, fontweight='bold', loc='left')  # 标题样式：字号 13、加粗、左对齐（续行）
    axes[0].set_ylabel('Amplitude (°C)', fontsize=11)  # 设置 y 轴标签“Amplitude (°C)”
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 添加半透明虚线网格
    axes[0].set_xlim(0, plot_length)  # 限定 x 轴显示范围为 0 到 plot_length
    
    # Plot approximation coefficients (low frequency)
    axes[1].plot(x_axis, approx[:plot_length], linewidth=1.2, color='#27AE60')  # 第 2 个子图绘制近似系数（低频趋势），绿色
    axes[1].set_title('(b) Approximation Coefficients (Low Frequency)',   # 设置近似系数子图标题（调用开始）
                      fontsize=13, fontweight='bold', loc='left')  # 标题样式（续行）
    axes[1].set_ylabel('Amplitude (°C)', fontsize=11)  # 设置 y 轴标签
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 添加虚线网格
    axes[1].set_xlim(0, plot_length)  # 限定 x 轴范围
    
    # Plot detail coefficients (high frequency)
    colors = ['#E74C3C', '#F39C12', '#9B59B6', '#34495E', '#E91E63']  # 定义细节系数可用的 5 种颜色
    detail_labels = ['D₁', 'D₂', 'D₃', 'D₄', 'D₅']  # 定义细节层的下标标签 D₁~D₅（原代码保留，未参与绘图）
    titles = [  # 定义各细节子图的标题列表（调用开始）
        '(c) Detail Coefficients D₁ (High Frequency)',  # 第 3 个子图（D₁，高频）标题
        '(d) Detail Coefficients D₂ (Medium Frequency)',  # 第 4 个子图（D₂，中频）标题
        '(e) Detail Coefficients D₃ (Highest Frequency)',  # 第 5 个子图（D₃，最高频）标题
        '(f) Detail Coefficients D₄',  # 第 6 个子图（D₄）标题
        '(g) Detail Coefficients D₅'  # 第 7 个子图（D₅）标题
    ]  # 标题列表定义结束
    
    for i in range(level):  # 逐层遍历细节系数并绘图
        axes[i + 2].plot(x_axis, details[i][:plot_length],  # 绘制第 i 层细节系数曲线（调用开始）
                        linewidth=1.2, color=colors[i % len(colors)])  # 设置线宽 1.2，并按层数循环取色
        axes[i + 2].set_title(titles[i], fontsize=13, fontweight='bold', loc='left')  # 设置该子图标题
        axes[i + 2].set_ylabel('Amplitude (°C)', fontsize=11)  # 设置 y 轴标签
        axes[i + 2].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 添加虚线网格
        axes[i + 2].set_xlim(0, plot_length)  # 限定 x 轴范围
    
    axes[-1].set_xlabel('Timestep', fontsize=11)  # 在最后一个子图设置 x 轴标签“Timestep”
    
    plt.tight_layout()  # 自动调整子图间距，防止重叠
    
    # Save in both PNG and PDF formats for publication
    filename_png = f'wavelet_decomposition_{col_name}_sci.png'  # 构造 PNG 输出文件名
    filename_pdf = f'wavelet_decomposition_{col_name}_sci.pdf'  # 构造 PDF 输出文件名
    
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')  # 以 300 dpi 保存 PNG 位图
    plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')  # 保存 PDF 矢量图，便于论文插入
    
    print(f"\n[OK] Wavelet decomposition figure saved (SCI format):")  # 打印“图片已保存”的提示
    print(f"  - {filename_png}")  # 输出 PNG 文件路径
    print(f"  - {filename_pdf}")  # 输出 PDF 文件路径
    plt.close()  # 关闭图形释放内存
    
    # Restore default font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 恢复默认中文字体为黑体
    plt.rcParams['font.family'] = 'sans-serif'  # 恢复字体族为无衬线（保持与其他图一致）

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 打印“加载训练好的模型”的提示
# 优先加载无数据泄漏版本的模型
USE_WAVELET = False  # 初始化“是否使用小波”标志为 False
NO_LEAKAGE = False  # 初始化“无数据泄漏”标志为 False
# 尝试多个可能的模型路径
model_paths = [  # 定义候选模型检查点路径列表（调用开始）
    'Model_GRUWavelet_no_leakage_pytorch.pth',  # 无数据泄漏版本的检查点文件名
]  # 列表定义结束

checkpoint = None  # 初始化检查点变量为空
MODEL_PATH = None  # 初始化实际加载的模型路径为空

for path in model_paths:  # 逐个尝试候选模型路径
    if os.path.exists(path):  # 仅当文件确实存在时才尝试加载
        try:  # try 块捕获加载过程中的异常
            checkpoint = torch.load(path, map_location=device, weights_only=False)  # 加载检查点，map_location 指定设备，weights_only=False 允许保存 scaler 对象
            MODEL_PATH = path  # 记录成功加载的模型路径
            
            # 判断模型类型
            if 'no_leakage' in path or checkpoint.get('no_data_leakage', False):  # 若路径含 no_leakage，或检查点标记了 no_data_leakage
                USE_WAVELET = True  # 启用小波特征
                NO_LEAKAGE = True  # 标记该模型为无数据泄漏版本
                print(f"成功加载小波变换模型（无数据泄漏版本）: {MODEL_PATH}")  # 打印无泄漏小波模型加载成功
            elif 'Wavelet' in path:  # 否则若路径名中包含 Wavelet
                USE_WAVELET = True  # 启用小波特征
                NO_LEAKAGE = False  # 按旧版本处理，可能存在数据泄漏
                print(f"成功加载小波变换模型（旧版本）: {MODEL_PATH}")  # 打印旧版小波模型加载成功
                print("⚠️  警告：该模型可能存在数据泄漏问题")  # 警告该模型可能存在数据泄漏
            else:  # 既不是小波无泄漏也不是旧小波，则视为普通 GRU
                USE_WAVELET = False  # 不使用小波特征
                NO_LEAKAGE = False  # 无泄漏标记保持 False
                print(f"成功加载普通GRU模型: {MODEL_PATH}")  # 打印普通 GRU 模型加载成功
            break  # 加载成功后跳出路径尝试循环
        except Exception as e:  # 捕获加载异常
            print(f"尝试加载 {path} 失败: {e}")  # 打印当前路径加载失败的原因
            continue  # 继续尝试下一个候选路径

if checkpoint is None:  # 若所有候选路径都未能加载
    raise FileNotFoundError(f"无法找到模型文件。尝试的路径: {model_paths}")  # 抛出 FileNotFoundError，列出尝试过的路径

input_dim = checkpoint['input_dim']  # 从检查点中读取输入特征维度
time_step = checkpoint['time_step']  # 从检查点中读取时间步长

# ⚠️ 重要：从模型中获取训练时使用的原始特征列表
if 'original_features' in checkpoint:  # 若检查点中保存了训练时的原始特征列表
    # 如果模型保存了原始特征列表，使用它
    model_features = checkpoint['original_features']  # 直接采用模型保存的原始特征列表，保证顺序一致
    print(f"\n从模型加载原始特征列表: {model_features}")  # 打印特征列表的来源
else:  # 否则检查点没有保存原始特征
    # 如果没有保存，尝试从feature_names反推（去掉小波后缀）
    feature_names_saved = checkpoint.get('feature_names', None)  # 尝试读取保存的特征名列表
    if feature_names_saved:  # 若确实保存了特征名
        # 从小波特征名中提取原始特征名
        model_features = []  # 初始化反推得到的原始特征列表
        for fname in feature_names_saved:  # 遍历保存的每个特征名
            if '_approx' in fname:  # 只处理以 _approx 结尾的小波近似特征名
                original_name = fname.replace('_approx', '')  # 去掉 _approx 后缀，还原原始列名
                if original_name not in model_features:  # 同一原始特征只保留一次，避免重复
                    model_features.append(original_name)  # 把还原出的原始特征加入列表
        print(f"\n从小波特征名反推原始特征: {model_features}")  # 打印反推得到的原始特征
    else:  # 连特征名也未保存
        # 如果都没有，使用默认特征（但会给出警告）
        model_features = features  # 退而使用测试数据的默认特征配置
        print(f"\n⚠️  警告：模型未保存特征列表，使用测试数据的特征: {model_features}")  # 警告特征来源不可靠，可能与训练不一致

# 检查测试数据是否包含所需特征
missing_features = [f for f in model_features if f not in data.columns]  # 找出测试数据中缺失的模型所需特征
if missing_features:  # 若存在缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出 ValueError，避免后续特征不匹配

# 使用模型训练时的特征
features = model_features  # 把特征列表替换为模型训练时使用的特征
print(f"\n最终使用特征: {features}")  # 打印最终确定的特征列表

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 重新定义候选编码列表，用于读取训练数据
train_data = None  # 初始化训练数据变量为空
for enc in encodings:  # 逐个尝试候选编码
    try:  # try 块捕获解码异常
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 读取训练数据，主要用于获取归一化参数
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印读取训练数据所用编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 继续尝试下一种编码
if train_data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出无法读取训练数据的异常

# 检查训练数据中是否包含所需特征
print(f"\n训练数据列名: {list(train_data.columns)}")  # 打印训练数据的全部列名，便于排查
missing_in_train = [f for f in features if f not in train_data.columns]  # 找出训练数据中缺失的特征
if missing_in_train:  # 若训练数据缺少部分特征
    print(f"\n⚠️ 警告：训练数据缺少以下特征: {missing_in_train}")  # 警告将只使用训练数据中存在的特征
    print(f"可用的特征: {[col for col in train_data.columns if col != 'timestamp']}")  # 打印训练数据中可用的特征清单
    # 只使用训练数据中存在的特征
    features = [f for f in features if f in train_data.columns]  # 过滤特征列表，只保留训练数据也有的列
    print(f"调整后使用的特征: {features}")  # 打印调整后的特征列表
    
    if len(features) == 0:  # 若调整后没有任何可用特征
        raise ValueError("训练数据中没有可用的特征！")  # 抛出 ValueError 终止程序

if USE_WAVELET:  # 若模型使用小波特征
    print("\n使用小波变换特征...")  # 打印“使用小波变换特征”的提示
    
    # 获取小波参数
    wavelet_type = checkpoint.get('wavelet', 'db4')  # 从小波检查点读取小波基函数
    wavelet_level = checkpoint.get('level', 3)  # 读取小波分解层数
    wavelet_target_col = checkpoint.get('wavelet_target_col', 'THI')  # 获取小波分解的目标列，确保与训练一致
    feature_names = checkpoint.get('feature_names', None)  # 读取保存的特征名列表
    
    print(f"  小波基函数: {wavelet_type}")  # 输出小波基函数
    print(f"  分解层数: {wavelet_level}")  # 输出分解层数
    print(f"  小波分解目标列: {wavelet_target_col}")  # 输出小波分解的目标列
    print(f"  数据泄漏防护: {'是' if NO_LEAKAGE else '否'}")  # 输出是否启用了数据泄漏防护
    
    # 从checkpoint中获取小波窗口大小，与训练时保持一致（训练时为time_step=48）
    wavelet_window_size = checkpoint.get('wavelet_window', time_step)  # 读取小波窗口大小；若未保存则用 time_step 兜底
    print(f"  小波窗口大小: {wavelet_window_size}（与训练时一致）")  # 输出实际使用的小波窗口大小
    
    # 根据模型版本选择小波分解方法
    if NO_LEAKAGE:  # 若为无数据泄漏版本
        # 使用滑动窗口小波分解（无数据泄漏，仅分解THI）
        print("\n对训练数据进行滑动窗口小波分解（仅分解THI）...")  # 打印对训练数据做滑动窗口分解的提示
        train_wavelet_features = wavelet_decompose_sliding(train_data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window_size, plot_decomposition=False)  # 分解训练数据（plot_decomposition=False，不出图）
        
        print("\n对测试数据进行滑动窗口小波分解（仅分解THI）...")  # 打印对测试数据做滑动窗口分解的提示
        test_wavelet_features = wavelet_decompose_sliding(data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window_size, plot_decomposition=True, plot_feature=wavelet_target_col)  # 分解测试数据并开启绘图，目标特征为小波目标列
    else:  # 若为旧版模型（可能有数据泄漏）
        # 使用传统小波分解（可能有数据泄漏）
        print("\n⚠️  使用传统小波分解（可能存在数据泄漏）")  # 警告将使用传统小波分解
        
        def wavelet_decompose_legacy(data, wavelet='db4', level=3, plot_decomposition=False, plot_feature='THI'):
            """传统小波分解方法（向后兼容）：对整段信号分解并重构，可能包含未来信息。"""
            wavelet_features = pd.DataFrame()  # 初始化空的结果 DataFrame
            plot_data = {}  # 初始化绘图数据字典
            
            for col in data.columns:  # 遍历数据的每一列
                if col == 'timestamp':  # 跳过时间戳列
                    continue  # 进入下一列
                signal = data[col].values  # 取出当前列的一维信号
                coeffs = pywt.wavedec(signal, wavelet, level=level)  # 对整段信号直接做小波分解（不滑动窗口，可能泄漏未来信息）
                approx = coeffs[0]  # 取出近似系数
                approx_recon = pywt.upcoef('a', approx, wavelet, level=level, take=len(signal))  # 用 upcoef 把近似系数重构回原信号长度
                wavelet_features[f'{col}_approx'] = approx_recon[:len(signal)]  # 把重构的近似分量写入结果表
                
                detail_recons = []  # 初始化细节重构分量列表
                for i, detail in enumerate(coeffs[1:], 1):  # 遍历每一层细节系数（编号从 1 开始）
                    detail_recon = pywt.upcoef('d', detail, wavelet, level=level-i+1, take=len(signal))  # 用 upcoef 把细节系数重构回原长度
                    wavelet_features[f'{col}_detail{i}'] = detail_recon[:len(signal)]  # 把第 i 层细节分量写入结果表
                    detail_recons.append(detail_recon[:len(signal)])  # 把该层细节分量保存下来用于绘图
                
                # 保存指定特征的数据用于绘图
                if plot_decomposition and col == plot_feature:  # 若需要绘图且当前列为指定特征
                    plot_data['col_name'] = col  # 记录列名
                    plot_data['original'] = signal  # 保存原始信号
                    plot_data['approx'] = approx_recon[:len(signal)]  # 保存重构后的近似分量
                    plot_data['details'] = detail_recons  # 保存各层细节分量
                    print(f"已保存 '{col}' 的小波分解数据用于绘图")  # 打印已保存绘图数据的提示
            
            if plot_decomposition and plot_data:  # 若需要绘图且已有绘图数据
                plot_wavelet_decomposition(plot_data, wavelet, level)  # 调用 SCI 风格绘图函数
            elif plot_decomposition and not plot_data:  # 若需要绘图但未找到目标特征
                print(f"⚠️  警告：未找到特征 '{plot_feature}'，无法绘制分解图")  # 输出未找到特征的警告
            
            return wavelet_features  # 返回结果特征表
        
        train_wavelet_features = wavelet_decompose_legacy(train_data[features], wavelet=wavelet_type, level=wavelet_level, plot_decomposition=False)  # 用传统方法分解训练数据
        test_wavelet_features = wavelet_decompose_legacy(data[features], wavelet=wavelet_type, level=wavelet_level, plot_decomposition=True, plot_feature='THI')  # 用传统方法分解测试数据，并绘制 THI 的分解图
    
    # 直接使用小波分解后的特征（已包含原始特征和小波特征）
    train_combined = train_wavelet_features  # 训练集组合特征直接使用小波特征表
    test_combined = test_wavelet_features  # 测试集组合特征直接使用小波特征表
    
    # 归一化（如果模型保存了scaler，直接使用）
    if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若检查点中保存了特征与目标的归一化器
        print("\n使用模型保存的归一化器...")  # 打印“使用模型保存的归一化器”的提示
        feature_scaler = checkpoint['feature_scaler']  # 直接复用训练时保存的特征 scaler
        target_scaler = checkpoint['target_scaler']  # 直接复用训练时保存的目标 scaler
    else:  # 否则检查点未保存归一化器
        print("\n使用训练数据重新拟合归一化器...")  # 打印“重新拟合归一化器”的提示
        feature_scaler = MinMaxScaler()  # 新建特征 scaler
        target_scaler = MinMaxScaler()  # 新建目标 scaler
        feature_scaler.fit(train_combined)  # 用训练组合特征拟合特征 scaler
        target_scaler.fit(train_data[[target]])  # 用训练数据目标列拟合目标 scaler
    
    # 保存原始列名
    original_columns = test_combined.columns  # 保存归一化前的列名顺序
    
    test_combined = pd.DataFrame(  # 用特征 scaler 变换测试组合特征并重建 DataFrame（调用开始）
        feature_scaler.transform(test_combined),  # 执行归一化变换
        columns=original_columns  # 保持列名与变换前一致
    )  # DataFrame 构造结束
    
    print(f"  特征数量: {len(test_combined.columns)}")  # 输出测试特征数量
    print(f"  特征列表: {list(test_combined.columns)}")  # 输出测试特征列表
    
else:  # 不使用小波时进入此分支
    print("\n使用原始特征...")  # 打印“使用原始特征”的提示
    
    # 普通模型：仅使用原始特征
    feature_scaler = MinMaxScaler()  # 新建特征 scaler
    target_scaler = MinMaxScaler()  # 新建目标 scaler
    
    feature_scaler.fit(train_data[features])  # 用训练数据原始特征拟合
    target_scaler.fit(train_data[[target]])  # 用训练数据目标列拟合
    
    test_combined = pd.DataFrame(  # 对测试数据原始特征做归一化并重建 DataFrame
        feature_scaler.transform(data[features]),  # 用训练集 scaler 变换测试特征
        columns=features  # 列名为原始特征名
    )  # DataFrame 构造结束
    
    print(f"  特征数量: {len(features)}")  # 输出使用的特征数量

# ===================== 序列构造 =====================
pred_offset = 6   # 与训练脚本 gru.py 保持一致（6步 × 5分钟 = 30分钟），标签取偏移后的值

def create_sequences(data, time_step):
    """
    创建时间序列样本
    注意：这里是对单个数据集（机器人新数据）进行序列构造，不涉及训练/测试划分
    参数:
        data: 单个数据集（这里为新采集数据），不做训练/测试划分
        time_step: 每个样本的历史时间步数
    返回:
        形状为 (样本数, time_step, 特征数) 的三维数组
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点范围，预留 pred_offset 保证标签存在
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 步历史特征作为一个样本
    return np.array(sequences)  # 堆叠为三维数组返回

print("\n创建序列数据...")  # 打印“创建序列数据”的提示
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 用目标 scaler 把测试数据的目标列归一化

# 创建序列
X_test = create_sequences(test_combined, time_step)  # 构造测试集特征序列
y_test = data_target_normalized[time_step + pred_offset:].flatten()  # 取偏移后的目标值并展平成一维数组
X_test = X_test[:len(y_test)]  # 确保长度匹配：序列数可能比标签数多 1，截断对齐

print(f"\n测试数据形状:")
print(f"  X_test: {X_test.shape}")  # 输出测试特征序列的形状
print(f"  y_test: {y_test.shape}")  # 输出测试标签的形状

# ===================== 构建模型 =====================
if USE_WAVELET:  # 若使用小波模型
    model = GRUWaveletModel(  # 实例化 GRUWaveletModel（调用开始）
        input_dim=input_dim,  # 输入维度来自检查点
        time_step=time_step,  # 时间步长来自检查点
        gru_hidden=128,  # 隐藏单元数 128
        gru_layers=1,  # GRU 层数 1
        dropout=0.2  # dropout 比率 0.2
    ).to(device)  # 把模型迁移到目标设备并结束构造
else:  # 否则构造普通模型
    model = GRUOnlyModel(  # 实例化 GRUOnlyModel（调用开始）
        input_dim=input_dim,  # 输入维度
        time_step=time_step,  # 时间步长
        gru_hidden=128,  # 隐藏单元数
        gru_layers=1,  # GRU 层数
        dropout=0.2  # dropout 比率
    ).to(device)  # 迁移到目标设备并结束构造

model.load_state_dict(checkpoint['model_state_dict'])  # 加载检查点中保存的模型权重
model.eval()  # 切换模型为评估模式，关闭 Dropout

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 输出模型的可训练参数量

# ===================== 模型预测 =====================
print("\n开始预测...")  # 打印“开始预测”的提示
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试特征转为 FloatTensor 并迁移到目标设备

with torch.no_grad():  # 关闭梯度计算以加速推理
    y_pred = model(X_test_tensor).cpu().numpy()  # 模型前向预测，结果转回 CPU 并转 NumPy 数组

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 把真实标签反归一化回原始量纲
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 把预测值反归一化回原始量纲

# 计算评估指标
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 对 MSE 开方得到 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

if USE_WAVELET:  # 按模型类型决定显示名称
    model_name = "GRU+Wavelet+ECA (No Leakage)" if NO_LEAKAGE else "GRU+Wavelet+ECA"  # 小波模型：无泄漏时标注 (No Leakage)
else:  # 普通模型分支
    model_name = "GRU-Only"  # 普通模型名称设为 GRU-Only

print("\n" + "=" * 60)  # 打印分隔线
print(f"{model_name} Model Test Results")  # 打印模型名称与结果标题
print("=" * 60)  # 再打印一行分隔线
print(f"R2 Score:  {r2:.4f}")  # 输出 R2 指标
print(f"MSE:       {mse:.4f}")  # 输出 MSE 指标
print(f"RMSE:      {rmse:.4f}")  # 输出 RMSE 指标
print(f"MAE:       {mae:.4f}")  # 输出 MAE 指标
print("=" * 60)  # 打印结束分隔线

# ===================== 结果可视化 =====================
# 获取时间戳信息
timestamps = data['timestamp'].values[time_step + pred_offset:time_step + pred_offset + len(y_test_rescaled)]  # 按偏移量截取与预测值一一对应的时间戳

# 筛选4.22-4.27的数据
import pandas as pd  # 再次导入 pandas（原代码重复导入，保留不改）
timestamps_pd = pd.to_datetime(timestamps)  # 把时间戳字符串转换为 DatetimeIndex
mask = (timestamps_pd.month == 4) & (timestamps_pd.day >= 22) & (timestamps_pd.day <= 27)  # 构造 4 月 22 日至 27 日的布尔筛选掩码
plot_indices = np.where(mask)[0]  # 取出满足日期条件的下标位置

if len(plot_indices) > 0:  # 若找到了目标时间段的数据
    plot_start = plot_indices[0]  # 记录绘制区间的起始下标
    plot_end = plot_indices[-1] + 1  # 记录绘制区间的结束下标（不含）
    plot_length = plot_end - plot_start  # 计算绘制区间长度
    
    y_test_plot = y_test_rescaled[plot_start:plot_end]  # 截取该时间段的真实值
    y_pred_plot = y_pred_rescaled[plot_start:plot_end]  # 截取该时间段的预测值
    timestamps_plot = timestamps_pd[plot_start:plot_end]  # 截取该时间段的时间戳
    print(f"\n预测时间范围: {timestamps_plot[0]} 到 {timestamps_plot[-1]}")  # 输出预测覆盖的起止时间
    print(f"预测数据点数: {len(y_test_plot)}")  # 输出该区间的数据点个数
    # 打印所有预测结果 - 表格格式
    print("\n" + "=" * 100)  # 打印表格的分隔线
    print(f"{'时间':<25} {'真实THI ':<20} {'预测THI ':<20} {'误差 ':<20}")  # 打印表头：时间、真实 THI、预测 THI、误差
    print("=" * 100)  # 打印表头下方分隔线
    for i, (ts, real, pred) in enumerate(zip(timestamps_plot, y_test_plot.flatten(), y_pred_plot.flatten())):  # 逐点遍历时间戳、真实值与预测值
        error = pred - real  # 计算单点预测误差 = 预测值 - 真实值
        ts_str = str(ts)[:19]  # 格式化时间戳：去掉小数秒，只保留到秒
        print(f"{ts_str:<25} {real:<20.4f} {pred:<20.4f} {error:<20.4f}")  # 按固定列宽输出单行结果，便于对齐阅读
    print("=" * 100)  # 打印表格底部分隔线
    
    # 绘制图表
    plt.figure(figsize=(16, 8))  # 创建 16x8 英寸的预测对比图画布
    y_test_flat = y_test_plot.flatten()  # 展平该时间段的真实值
    y_pred_flat = y_pred_plot.flatten()  # 展平该时间段的预测值
    plt.tick_params(axis='both', labelsize=12)  # 设置横纵坐标轴刻度标签字号为 12
    plt.plot(timestamps_plot, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4', marker='o', markersize=3)  # 绘制真实值曲线（蓝色圆点标记）
    plt.plot(timestamps_plot, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e', marker='s', markersize=3)  # 绘制预测值曲线（橙色方块标记）
    plt.xlabel('Time', fontsize=14)  # 设置 x 轴标签为 Time
    plt.ylabel('THI ', fontsize=14)  # 设置 y 轴标签为 THI
    plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)  # 设置标题并附带 R2 与 RMSE
    plt.legend(fontsize=12)  # 显示图例
    plt.grid(True, alpha=0.3)  # 添加半透明网格
    plt.xticks(rotation=45, ha='right')  # 把时间标签旋转 45 度并右对齐，避免重叠
    plt.tight_layout()  # 自动调整布局
    plt.text(0.5, 0.95, '(f)', transform=plt.gca().transAxes,  # 在图中添加子图编号“(f)”（调用开始）
             fontsize=16, fontweight='bold', va='top', ha='center')  # 设置编号文本的字号、加粗与对齐方式（续行）
    if USE_WAVELET:  # 按模型类型选择输出文件名
        output_filename = 'test_GRUWavelet_no_leakage_prediction.png' if NO_LEAKAGE else 'test_GRUWavelet_prediction.png'  # 小波模型根据是否泄漏选择对应文件名
    else:  # 普通模型分支
        output_filename = 'test_GRUOnly_prediction.png'  # 普通模型输出文件名
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')  # 保存预测对比图
    print(f"\n预测结果图已保存: {output_filename}")  # 打印图片保存路径
    plt.close()  # 关闭图形释放内存
else:  # 若未找到 4.22-4.27 的数据
    print("\n警告: 未找到4.22-4.27的数据，使用全部数据绘制")  # 输出警告并改用全部数据绘制
    plot_length = min(600, len(y_test_rescaled))  # 最多取前 600 个点，避免图过密
    y_test_plot = y_test_rescaled[:plot_length]  # 截取前 plot_length 个真实值
    y_pred_plot = y_pred_rescaled[:plot_length]  # 截取前 plot_length 个预测值
    steps = np.arange(plot_length)  # 生成步数横轴
    plt.figure(figsize=(15, 8))  # 创建 15x8 英寸画布
    y_test_flat = y_test_plot.flatten()  # 展平真实值
    y_pred_flat = y_pred_plot.flatten()  # 展平预测值
    plt.tick_params(axis='both', labelsize=16)  # 设置刻度标签字号为 16
    plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 以步数为横轴绘制真实值
    plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 以步数为横轴绘制预测值
    plt.xlabel('Timestep', fontsize=16)  # 设置 x 轴标签为 Timestep
    plt.ylabel('THI ', fontsize=16)  # 设置 y 轴标签为 THI
    plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)  # 设置标题并附带 R2 与 RMSE
    plt.legend(fontsize=16)  # 显示图例
    plt.grid(True, alpha=0.3)  # 添加网格
    plt.tight_layout()  # 自动调整布局
    plt.text(0.5, 0.95, '(f)', transform=plt.gca().transAxes,  # 在图中添加子图编号“(f)”（调用开始）
             fontsize=16, fontweight='bold', va='top', ha='center')  # 设置编号文本样式（续行）
    if USE_WAVELET:  # 按模型类型选择文件名
        output_filename = 'test_GRUWavelet_no_leakage_prediction.png' if NO_LEAKAGE else 'test_GRUWavelet_prediction.png'  # 小波模型文件名
    else:  # 普通模型分支
        output_filename = 'test_GRUOnly_prediction.png'  # 普通模型文件名
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')  # 保存预测图
    print(f"\n预测结果图已保存: {output_filename}")  # 打印保存路径
    plt.close()  # 关闭图形

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算逐点误差：预测值减真实值

plt.figure(figsize=(15, 10))  # 创建 15x10 英寸误差分析画布

# 子图1: 预测值 vs 真实值散点图
plt.subplot(2, 2, 1)  # 选中第 1 个子图
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, s=20)  # 绘制真实值与预测值的散点图
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],  # 绘制理想预测对角线 y=x（调用开始）
         [y_test_rescaled.min(), y_test_rescaled.max()],  # 对角线终点坐标（续行）
         'r--', lw=2, label='理想预测线')  # 红色虚线、线宽 2，并标注图例“理想预测线”
plt.xlabel('真实THI ', fontsize=11)  # 设置 x 轴标签“真实THI”
plt.ylabel('预测THI ', fontsize=11)  # 设置 y 轴标签“预测THI”
plt.title(f'预测值 vs 真实值 (R²={r2:.4f})', fontsize=12)  # 设置标题含 R²
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 选中第 2 个子图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差分布直方图，50 个区间
plt.xlabel('预测误差 ', fontsize=11)  # 设置 x 轴标签“预测误差”
plt.ylabel('频数', fontsize=11)  # 设置 y 轴标签“频数”
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 设置标题含 MAE
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 画零误差红色竖虚线
plt.grid(True, alpha=0.3, axis='y')  # 仅沿 y 轴添加网格
# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 选中第 3 个子图
error_steps = np.arange(min(120, len(errors)))  # 取前 120 个误差生成横轴序号
plt.plot(error_steps, errors[:120], alpha=0.7, linewidth=1)  # 绘制误差随时间变化的折线
plt.xlabel('Timestep', fontsize=11)  # 设置 x 轴标签 Timestep
plt.ylabel('Prediction Error ', fontsize=11)  # 设置 y 轴标签“Prediction Error”
plt.title('Error Time Series', fontsize=12)  # 设置标题“Error Time Series”
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 画零误差红色横虚线
plt.grid(True, alpha=0.3)  # 添加网格
# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 选中第 4 个子图
box_data = [errors]  # 把误差序列包装成箱线图输入列表
plt.boxplot(box_data, tick_labels=[model_name])  # 绘制误差箱线图，x 轴标签为模型名称
plt.ylabel('预测误差 ', fontsize=11)  # 设置 y 轴标签“预测误差”
plt.title('误差统计分布', fontsize=12)  # 设置标题“误差统计分布”
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 画零误差红色横虚线
plt.grid(True, alpha=0.3, axis='y')  # 仅沿 y 轴添加网格
plt.tight_layout()  # 自动调整 4 个子图间距
if USE_WAVELET:  # 按模型类型选择误差图文件名
    error_filename = 'test_GRUWavelet_no_leakage_error_analysis.png' if NO_LEAKAGE else 'test_GRUWavelet_error_analysis.png'  # 小波模型误差图文件名（按是否泄漏区分）
else:  # 普通模型分支
    error_filename = 'test_GRUOnly_error_analysis.png'  # 普通模型误差图文件名
plt.savefig(error_filename, dpi=150, bbox_inches='tight')  # 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印误差图保存路径
plt.close()  # 关闭图形
# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("Error Statistics")  # 打印“Error Statistics”标题
print("=" * 60)  # 再打印一行分隔线
print(f"Mean Error:     {np.mean(errors):.4f} C")  # 输出平均误差（Mean Error）
print(f"THI Deviation:  {np.THI(errors):.4f} C")  # 原代码调用 np.THI 计算所谓“THI Deviation”（NumPy 并无此函数，按要求保留原样不改代码）
print(f"Max Error:      {np.max(errors):.4f} C")  # 输出最大误差
print(f"Min Error:      {np.min(errors):.4f} C")  # 输出最小误差
print(f"Median Error:   {np.median(errors):.4f} C")  # 输出误差中位数
print("=" * 60)  # 打印结束分隔线

print("\nTest Completed!")  # 打印测试流程全部完成的提示






