"""
GRU模型测试脚本（支持小波变换）
用于加载训练好的GRU模型并进行预测
支持：GRU-Only模型、GRU+小波变换模型
本脚本是 gru.py 训练流程对应的推理测试脚本：读取新采集的温湿度数据，做与训练阶段一致的
缺失值处理和小波分解；从 .pth 检查点恢复输入维度、时间窗口、原始特征列表与归一化器，
构造时间窗口序列后加载训练好的权重进行预测，输出 R²/MSE/RMSE/MAE 指标，并绘制
预测对比图和四子图误差分析图。

主要流程:
    1. 多编码读取测试数据并构造时间戳，校验目标列是否存在；
    2. 连续变量插值、离散变量前后填充的缺失值处理；
    3. 尝试多个候选路径加载模型，识别是否为小波/无泄漏版本；
    4. 读取训练数据恢复归一化参数，按需做滑动窗口或传统小波分解；
    5. 构造序列、载入权重、预测并反归一化，计算评估指标；
    6. 绘制预测曲线与误差分析图，并打印误差统计信息。
"""
import pandas as pd  # 导入 pandas 库，用于读取和处理表格数据
import numpy as np  # 导入 numpy 库，用于数值计算与数组操作
import matplotlib  # 导入 matplotlib 基础库，为切换绘图后端做准备
matplotlib.use('Agg')  # 设置非交互式 Agg 后端，只保存图片不弹出图形窗口
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制预测对比图与误差分析图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 用于特征归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R²、MSE、MAE 等评估指标函数
import torch  # 导入 PyTorch 主库，用于张量运算与模型推理
import torch.nn as nn  # 导入 torch.nn 模块，用于定义神经网络层
import pywt  # 小波变换库  # 小波变换库，用于多尺度分解与系数重构
import os  # 导入 os 模块，用于判断模型文件是否存在

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择推理设备：有 GPU 用 CUDA，否则用 CPU
print(f"使用设备: {device}")  # 打印当前使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图的中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，保证负号正常呈现


# ===================== 模型定义 =====================
class ECALayer(nn.Module):  # 定义 ECA 高效通道注意力层
    """
    ECA-Net: Efficient Channel Attention
    通过1D卷积实现高效的通道注意力机制
    参数:
        channels: 输入特征的通道数
        gamma, b: 用于自适应计算一维卷积核大小的系数
    """
    def __init__(self, channels, gamma=2, b=1):  # 初始化 ECA 层，按通道数自适应确定一维卷积核大小
        """
        初始化 ECA 通道注意力层

        参数:
            channels: 输入特征的通道数
            gamma: 计算卷积核尺寸时的缩放系数
            b: 计算卷积核尺寸时的偏置项
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按公式自适应计算卷积核尺寸参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数  # 若 t 为偶数则加 1，保证卷积核尺寸为奇数
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 定义全局平均池化，把时间维压缩成 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 定义一维卷积实现跨通道交互，不降维、无偏置
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活，把注意力权重映射到 0-1
    
    def forward(self, x):  # 定义前向传播：对输入特征进行通道注意力加权
        """
        对输入执行通道注意力加权

        参数:
            x: 形状为 (batch, time_step, channels) 的输入张量
        返回:
            通道加权后的张量，形状保持不变
        """
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step)
        x_permuted = x.permute(0, 2, 1)  # 交换时间维与通道维，使张量满足池化的输入格式
        
        # 全局平均池化
        y = self.avg_pool(x_permuted)  # 全局平均池化，得到每个通道的全局描述向量
        
        # 1D卷积
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 转置后做一维卷积再转回，实现通道间信息交互
        
        # Sigmoid激活
        y = self.sigmoid(y)  # Sigmoid 激活，得到各通道的注意力权重
        
        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 把权重按通道广播后乘回原始输入特征
        
        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 恢复原始维度顺序并返回加权后的结果


class GRUOnlyModel(nn.Module):  # 定义只使用原始特征的 GRU + ECA 预测模型
    """GRU-Only 模型 + ECA-Net注意力机制

    仅使用原始特征：先经 ECA 增强通道，再用 GRU 提取时序特征，
    最后取末尾时刻的输出经全连接层给出预测值。

    参数:
        input_dim: 输入特征维度
        time_step: 时间窗口长度
        gru_hidden: GRU 隐藏层单元数
        gru_layers: GRU 层数
        dropout: 多层 GRU 时的 dropout 比例
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 初始化 GRU-Only 模型结构
        """
        初始化 GRU-Only 模型

        参数:
            input_dim: 输入特征维度
            time_step: 时间窗口长度
            gru_hidden: GRU 隐藏层单元数
            gru_layers: GRU 层数
            dropout: 多层 GRU 时的 dropout 比例
        """
        super(GRUOnlyModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层单元数

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块，在 GRU 前增强输入特征
        
        self.gru = nn.GRU(  # 定义单向 GRU 编码器，提取输入序列的时序特征
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接输出层，把隐藏状态映射为单一预测值

    def forward(self, x):  # 定义前向传播：ECA 增强、GRU 编码、全连接输出
        """
        前向传播：ECA 注意力增强 -> GRU 编码 -> 全连接输出

        参数:
            x: 形状为 (batch, time_step, input_dim) 的输入序列
        返回:
            形状为 (batch, 1) 的预测值
        """
        # ECA注意力增强输入特征
        x = self.eca(x)  # 用 ECA 注意力增强输入特征的通道表达
        
        # GRU编码
        out, h = self.gru(x)  # GRU 编码，得到各时刻输出与隐藏状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # 取最后一个时间步的输出作为整段序列的表示
        
        return self.fc(out_last)  # 经全连接层输出最终预测值


class GRUWaveletModel(nn.Module):  # 定义 GRU + 小波特征 + ECA 的预测模型
    """GRU + 小波变换 + ECA注意力机制

    结构与 GRUOnlyModel 相同，但输入为小波分解后的多尺度特征；
    用于加载基于小波特征的检查点进行推理预测。

    参数:
        input_dim: 输入特征维度（小波特征数）
        time_step: 时间窗口长度
        gru_hidden: GRU 隐藏层单元数
        gru_layers: GRU 层数
        dropout: 多层 GRU 时的 dropout 比例
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 初始化 GRU 小波模型结构
        """
        初始化 GRU 小波模型

        参数:
            input_dim: 输入特征维度（小波特征数）
            time_step: 时间窗口长度
            gru_hidden: GRU 隐藏层单元数
            gru_layers: GRU 层数
            dropout: 多层 GRU 时的 dropout 比例
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层单元数

        # ECA-Net注意力模块
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块
        
        # 单层GRU编码器
        self.gru = nn.GRU(  # 定义单向 GRU 编码器，提取时序特征
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        
        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接输出层，输出单一预测值

    def forward(self, x):  # 定义前向传播：ECA 增强、GRU 编码、全连接输出
        """
        前向传播：ECA 注意力增强 -> GRU 编码 -> 全连接输出

        参数:
            x: 形状为 (batch, time_step, input_dim) 的输入序列
        返回:
            形状为 (batch, 1) 的预测值
        """
        # ECA注意力增强
        x = self.eca(x)  # 用 ECA 注意力增强输入特征
        
        # GRU编码
        out, h = self.gru(x)  # GRU 编码得到各时刻的输出与隐藏状态
        out_last = out[:, -1, :]  # 取最后一个时间步的输出作为序列表示

        # 预测输出
        return self.fc(out_last)  # 经全连接层输出最终预测值
# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印加载测试数据的提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义读取 CSV 时依次尝试的编码列表
data = None  # 初始化数据变量为空
for enc in encodings:  # 依次尝试每种编码读取测试数据
    try:  # 捕获解码异常
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据 CSV
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功读取所用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 若解码失败则尝试下一种编码
        continue  # 继续尝试下一种编码

if data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常终止程序
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 按索引乘 5 分钟、以 2025/6/21 0:04 为起点生成时间戳
                                   origin=pd.Timestamp('2025/6/21 0:04'))
# 特征配置
features = ['日龄','温度偏差','CO2','目标温度','外界温度','相对湿度']  # 定义候选特征列表
target = '温度偏差'  # 设定预测目标列为温度偏差

# 过滤特征列（只保留数据中存在的列）
features = [f for f in features if f in data.columns]  # 只保留测试数据中实际存在的特征列

print(f"使用特征: {features}")  # 打印实际使用的特征列表
print(f"数据形状: {data.shape}")  # 打印测试数据的行列形状

# 检查目标列是否存在
if target not in data.columns:  # 检查目标列是否存在于测试数据中
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带温度偏差的CSV文件。")  # 缺少目标列则抛出 KeyError 提示换用带温度偏差的 CSV

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印由等号组成的分隔线
print("缺失值处理")  # 打印缺失值处理标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据形状
print(f"\n各列缺失值统计:")  # 打印各列缺失值统计标题
for col in features + [target]:  # 遍历所有特征列与目标列
    if col in data.columns:  # 只处理数据中存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算该列缺失值占样本的百分比
        if nan_count > 0:  # 只打印含缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失值数量与占比

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度','相对湿度' ]  # 连续变量  # 定义需要线性插值的连续变量列表
discrete_vars = ['日龄']  # 离散变量  # 定义只做前后填充的离散变量列表

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量列表
    if col in data.columns and data[col].isna().sum() > 0:  # 若该列存在且含有缺失值
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录插值前的缺失值数量
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 双向线性插值填补连续变量的缺失值
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].ffill()  # 前向填充开头的缺失值
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].bfill()  # 后向填充结尾的缺失值
        
        after_nan = data[col].isna().sum()  # 记录插值后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前的缺失值个数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后的缺失值个数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量列表
    if col in data.columns and data[col].isna().sum() > 0:  # 若该离散列存在且含有缺失值
        print(f"\n处理 {col}:")  # 打印正在处理的离散列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 前向填充
        data[col] = data[col].ffill()  # 前向填充离散变量的缺失值
        # 后向填充
        data[col] = data[col].bfill()  # 后向填充离散变量的缺失值
        
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前的缺失值个数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后的缺失值个数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计所有特征与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 若处理后仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告并说明将删除含缺失值的行
    data = data.dropna(subset=features + [target])  # 删除特征或目标列中含缺失值的样本行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据形状
print(f"{'='*60}\n")  # 打印分隔线并换行

# 重置索引
data = data.reset_index(drop=True)  # 重置索引，保证行下标从 0 开始连续

# ===================== 小波变换函数（无数据泄漏版本，仅分解温度偏差） =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None, plot_decomposition=False, plot_feature='温度偏差'):  # 定义滑动窗口小波分解函数，仅对目标列分解以避免未来信息泄漏
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（温度偏差）进行小波分解，其他列保持原样
    每个时间点只使用历史数据
    参数:
        data: 输入数据 (DataFrame)
        target_col: 需要进行小波分解的列名（默认'温度偏差'）
        wavelet: 小波基函数 ('db4', 'sym5', 'coif3' 等)
        level: 分解层数
        window_size: 滑动窗口大小，None则自动计算
        plot_decomposition: 是否绘制小波分解效果图
        plot_feature: 要绘制的特征名称（默认为'温度偏差'）
    返回:
        分解后的特征 (DataFrame)
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印开始分解的提示信息
    print(f"小波基函数: {wavelet}")  # 打印所选的小波基函数
    print(f"分解层数: {level}")  # 打印小波分解的层数
    # 自动计算窗口大小：确保足够大以进行有效的小波分解
    if window_size is None:  # 若调用时未指定滑动窗口大小
        window_size = min(36, len(data))  # 使用36作为窗口大小  # 取 36 与数据长度的较小值作为窗口大小
    
    print(f"滑动窗口大小: {window_size}")  # 打印实际使用的滑动窗口大小
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建空的结果表，索引与原数据一致
    
    # 用于绘图的数据
    plot_data = {}  # 初始化用于保存绘图数据的字典
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 跳过时间戳列，不参与小波分解
            continue  # 进入下一列的处理
        
        signal = data[col].values  # 取出当前列的数值数组
        
        # 只对目标列进行小波分解
        if col == target_col:  # 若当前列是需要做小波分解的目标列
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 初始化近似系数列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数初始化一个空列表
            
            for i in range(len(signal)):  # 按时间点逐格滑动窗口
                # 只使用当前时间点之前的数据（包括当前点）
                start_idx = max(0, i - window_size + 1)  # 计算当前窗口的起始下标，保证不小于 0
                window_signal = signal[start_idx:i+1]  # 截取当前时间点及其之前的窗口数据，不使用未来信息
                
                # 如果窗口太小，用边界值填充到window_size
                if len(window_signal) < window_size:  # 若窗口长度不足
                    pad_width = window_size - len(window_signal)  # 计算需要在窗口前端补充的点数
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边界值在窗口前端填充至指定长度
                
                # 小波分解
                try:  # 尝试对当前窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 wavedec 进行 level 层分解
                    
                    # 取最后一个值作为当前时间点的特征
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为当前时刻特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层细节系数最后一个值，为空则补 0
                except Exception as e:  # 若分解过程出现异常
                    # 如果分解失败，使用原始值
                    approx_list.append(window_signal[-1])  # 降级处理：用窗口最后一个原始值作为近似特征
                    for j in range(level):  # 遍历每一层细节系数
                        detail_lists[j].append(0.0)  # 异常时把细节系数补为 0
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 把近似系数序列添加为新特征列
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 把每层细节系数添加为新特征列
            
            # 保存指定特征的数据用于绘图
            if plot_decomposition and col == plot_feature:  # 若要求绘图且当前列是指定的绘图特征
                plot_data['col_name'] = col  # 记录要绘图的列名
                plot_data['original'] = signal  # 保存当前列的原始信号
                plot_data['approx'] = np.array(approx_list)  # 保存近似系数数组
                plot_data['details'] = [np.array(detail_lists[j]) for j in range(level)]  # 保存各层细节系数数组
                print(f"已保存 '{col}' 的小波分解数据用于绘图")  # 打印绘图数据已保存的提示
        else:  # 若当前列不是分解目标列
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列不做分解的提示
            wavelet_features[col] = signal  # 把该列原始值直接复制到结果表中
    
    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印分解完成提示与生成特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印由目标列分解产生的小波特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印保留下来的其他原始特征数量
    
    # 绘制小波分解效果图
    if plot_decomposition and plot_data:  # 若要求绘图且已收集到绘图数据
        plot_wavelet_decomposition(plot_data, wavelet, level)  # 调用绘图函数绘制小波分解效果图
    elif plot_decomposition and not plot_data:  # 若要求绘图但没有收集到绘图数据
        print(f"⚠️  警告：未找到特征 '{plot_feature}'，无法绘制分解图")  # 打印未找到绘图特征的警告
    
    return wavelet_features  # 返回包含原始特征与小波特征的 DataFrame


def plot_wavelet_decomposition(plot_data, wavelet, level):  # 定义学术论文风格的小波分解绘图函数
    """
    Plot wavelet decomposition for SCI publication
    Parameters:
        plot_data: Dictionary containing original signal and decomposition results
        wavelet: Wavelet basis function name
        level: Decomposition level
    中文说明:
        将小波分解得到的原始信号、近似系数与各层细节系数绘制成适合论文发表的多子图，
        并同时保存 PNG 与 PDF 两种格式。绘图前临时使用 Times New Roman 字体并加粗
        坐标轴，绘图结束后恢复中文 SimHei 字体设置。
        参数:
            plot_data: 包含原始信号与分解结果的字典
            wavelet: 小波基函数名称
            level: 分解层数
        返回:
            无返回值；图片保存到本地并关闭画布
    """
    # Set publication-quality parameters
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置绘图字体为 Times New Roman
    plt.rcParams['font.size'] = 11  # 设置全局字号为 11
    plt.rcParams['axes.linewidth'] = 1.2  # 把坐标轴线宽加粗到 1.2
    
    col_name = plot_data['col_name']  # 取出要绘图的列名
    original = plot_data['original']  # 取出原始信号
    approx = plot_data['approx']  # 取出近似系数
    details = plot_data['details']  # 取出各层细节系数
    
    # Plot first 1000 points for better visualization
    plot_length = min(800, len(original))  # 只绘制前 800 个点，避免曲线过密
    x_axis = np.arange(plot_length)  # 生成横轴步长的索引数组
    
    # Create subplots
    fig, axes = plt.subplots(level + 2, 1, figsize=(12, 10))  # 创建 level+2 行 1 列子图：原始信号、近似系数及各层细节
    
    # Plot original signal
    axes[0].plot(x_axis, original[:plot_length], linewidth=1.2, color='#2C3E50')  # 在第一个子图绘制原始信号
    axes[0].set_title('(a) Original Signal: Temperature Deviation',  # 设置原始信号子图的标题
                      fontsize=13, fontweight='bold', loc='left')
    axes[0].set_ylabel('Amplitude (°C)', fontsize=11)  # 设置纵轴标签为幅值（摄氏度）
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 为原始信号子图添加虚线网格
    axes[0].set_xlim(0, plot_length)  # 限定原始信号子图的横轴范围
    
    # Plot approximation coefficients (low frequency)
    axes[1].plot(x_axis, approx[:plot_length], linewidth=1.2, color='#27AE60')  # 绘制近似系数（低频趋势成分）
    axes[1].set_title('(b) Approximation Coefficients (Low Frequency)',   # 设置近似系数子图的标题
                      fontsize=13, fontweight='bold', loc='left')
    axes[1].set_ylabel('Amplitude (°C)', fontsize=11)  # 设置近似系数子图的纵轴标签
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 为近似系数子图添加网格
    axes[1].set_xlim(0, plot_length)  # 限定近似系数子图的横轴范围
    
    # Plot detail coefficients (high frequency)
    colors = ['#E74C3C', '#F39C12', '#9B59B6', '#34495E', '#E91E63']  # 定义各层细节系数使用的配色方案
    detail_labels = ['D₁', 'D₂', 'D₃', 'D₄', 'D₅']  # 定义各层细节系数的下标标签
    titles = [  # 定义各细节系数子图的标题文本
        '(c) Detail Coefficients D₁ (High Frequency)',
        '(d) Detail Coefficients D₂ (Medium Frequency)',
        '(e) Detail Coefficients D₃ (Highest Frequency)',
        '(f) Detail Coefficients D₄',
        '(g) Detail Coefficients D₅'
    ]
    
    for i in range(level):  # 按层循环绘制细节系数
        axes[i + 2].plot(x_axis, details[i][:plot_length],   # 绘制第 i 层细节系数曲线
                        linewidth=1.2, color=colors[i % len(colors)])
        axes[i + 2].set_title(titles[i], fontsize=13, fontweight='bold', loc='left')  # 设置当前细节子图的标题
        axes[i + 2].set_ylabel('Amplitude (°C)', fontsize=11)  # 设置当前细节子图的纵轴标签
        axes[i + 2].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 为当前细节子图添加网格
        axes[i + 2].set_xlim(0, plot_length)  # 限定当前细节子图的横轴范围
    
    axes[-1].set_xlabel('Timestep', fontsize=11)  # 为最后一个子图设置横轴标签为时间步
    
    plt.tight_layout()  # 自动调整所有子图的间距
    
    # Save in both PNG and PDF formats for publication
    filename_png = f'wavelet_decomposition_{col_name}_sci.png'  # 生成 PNG 格式的输出文件名
    filename_pdf = f'wavelet_decomposition_{col_name}_sci.pdf'  # 生成 PDF 格式的输出文件名
    
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')  # 以 300dpi 保存 PNG 图片
    plt.savefig(filename_pdf, dpi=300, bbox_inches='tight')  # 保存 PDF 矢量图，满足论文出版要求
    
    print(f"\n[OK] Wavelet decomposition figure saved (SCI format):")  # 打印图片保存成功的提示
    print(f"  - {filename_png}")  # 打印 PNG 文件路径
    print(f"  - {filename_pdf}")  # 打印 PDF 文件路径
    plt.close()  # 关闭画布释放内存
    
    # Restore default font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 恢复 SimHei 中文字体设置
    plt.rcParams['font.family'] = 'sans-serif'  # 恢复默认字体族为无衬线

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 打印加载训练好模型的提示

# 优先加载无数据泄漏版本的模型
USE_WAVELET = False  # 初始化是否使用小波特征的标志为 False
NO_LEAKAGE = False  # 初始化是否为无数据泄漏模型的标志为 False

# 尝试多个可能的模型路径
model_paths = [  # 定义候选模型文件路径列表，按优先级排列
    'Model_GRUCO2Wavelet_no_leakage_pytorch.pth',
    '../pth/Model_GRUCO2Wavelet_no_leakage_pytorch.pth',
    'pth/Model_GRUCO2Wavelet_no_leakage_pytorch.pth',
    '../pth/Model_GRUWavelet_pytorch.pth',
    'pth/Model_GRUWavelet_pytorch.pth',
    '../pth/Model_GRUOnly_pytorch.pth',
    'pth/Model_GRUOnly_pytorch.pth'
]

checkpoint = None  # 初始化加载的模型检查点为空
MODEL_PATH = None  # 初始化实际命中的模型路径为空

for path in model_paths:  # 依次尝试每一个候选模型路径
    if os.path.exists(path):  # 若当前路径的文件确实存在
        try:  # 捕获加载过程中可能出现的异常
            checkpoint = torch.load(path, map_location=device, weights_only=False)  # 加载模型权重及配置到当前计算设备
            MODEL_PATH = path  # 记录本次成功加载的模型路径
            
            # 判断模型类型
            if 'no_leakage' in path or checkpoint.get('no_data_leakage', False):  # 根据路径名或检查点标记判断是否为无泄漏小波模型
                USE_WAVELET = True  # 标记使用小波特征
                NO_LEAKAGE = True  # 标记为无数据泄漏版本
                print(f"成功加载小波变换模型（无数据泄漏版本）: {MODEL_PATH}")  # 打印无泄漏小波模型加载成功
            elif 'Wavelet' in path:  # 否则若路径名中含有 Wavelet
                USE_WAVELET = True  # 标记使用小波特征
                NO_LEAKAGE = False  # 标记为旧版小波模型，可能存在数据泄漏
                print(f"成功加载小波变换模型（旧版本）: {MODEL_PATH}")  # 打印旧版小波模型加载成功
                print("⚠️  警告：该模型可能存在数据泄漏问题")  # 提示旧模型可能存在数据泄漏问题
            else:  # 否则视为普通 GRU 模型
                USE_WAVELET = False  # 标记不使用小波特征
                NO_LEAKAGE = False  # 标记为非无泄漏版本
                print(f"成功加载普通GRU模型: {MODEL_PATH}")  # 打印普通 GRU 模型加载成功
            break  # 加载成功后跳出路径循环
        except Exception as e:  # 若加载过程中出现异常
            print(f"尝试加载 {path} 失败: {e}")  # 打印当前路径加载失败的原因
            continue  # 继续尝试下一个候选路径

if checkpoint is None:  # 若所有候选路径都加载失败
    raise FileNotFoundError(f"无法找到模型文件。尝试的路径: {model_paths}")  # 抛出 FileNotFoundError 终止程序

input_dim = checkpoint['input_dim']  # 从检查点读取模型输入特征维度
time_step = checkpoint['time_step']  # 从检查点读取模型使用的时间窗口长度

# ⚠️ 重要：从模型中获取训练时使用的原始特征列表
if 'original_features' in checkpoint:  # 若模型中保存了训练时的原始特征列表
    # 如果模型保存了原始特征列表，使用它
    model_features = checkpoint['original_features']  # 直接采用模型保存的原始特征列表
    print(f"\n从模型加载原始特征列表: {model_features}")  # 打印从模型加载到的原始特征列表
else:  # 否则尝试从小波特征名反推原始特征
    # 如果没有保存，尝试从feature_names反推（去掉小波后缀）
    feature_names_saved = checkpoint.get('feature_names', None)  # 尝试读取保存的小波特征名列表
    if feature_names_saved:  # 若检查点中存在特征名列表
        # 从小波特征名中提取原始特征名
        model_features = []  # 初始化反推得到的原始特征列表
        for fname in feature_names_saved:  # 遍历每一个小波特征名
            if '_approx' in fname:  # 若特征名中含有 _approx 后缀
                original_name = fname.replace('_approx', '')  # 去掉后缀还原出原始特征名
                if original_name not in model_features:  # 避免把同一个原始特征重复加入
                    model_features.append(original_name)  # 把还原出的原始特征名加入列表
        print(f"\n从小波特征名反推原始特征: {model_features}")  # 打印从小波特征名反推出的原始特征
    else:  # 若检查点中既无原始特征列表也无特征名列表
        # 如果都没有，使用默认特征（但会给出警告）
        model_features = features  # 退而使用测试数据中的默认特征列表
        print(f"\n⚠️  警告：模型未保存特征列表，使用测试数据的特征: {model_features}")  # 打印未保存特征列表的警告信息

# 检查测试数据是否包含所需特征
missing_features = [f for f in model_features if f not in data.columns]  # 找出测试数据中缺失的模型所需特征
if missing_features:  # 若存在缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出 ValueError 并列出缺失的特征

# 使用模型训练时的特征
features = model_features  # 最终统一采用模型训练时使用的特征列表
print(f"\n最终使用特征: {features}")  # 打印最终确定的特征列表

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义读取训练数据时依次尝试的编码列表
train_data = None  # 初始化训练数据变量为空
for enc in encodings:  # 依次尝试每种编码读取训练数据
    try:  # 捕获解码异常
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据以获取归一化参数
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功读取训练数据所用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 若解码失败则尝试下一种编码
        continue  # 继续尝试下一种编码
if train_data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序

# 检查训练数据中是否包含所需特征
print(f"\n训练数据列名: {list(train_data.columns)}")  # 打印训练数据的所有列名
missing_in_train = [f for f in features if f not in train_data.columns]  # 找出训练数据中缺失的特征
if missing_in_train:  # 若训练数据中存在缺失特征
    print(f"\n⚠️ 警告：训练数据缺少以下特征: {missing_in_train}")  # 打印训练数据缺少特征的警告
    print(f"可用的特征: {[col for col in train_data.columns if col != 'timestamp']}")  # 打印训练数据中实际可用的特征
    # 只使用训练数据中存在的特征
    features = [f for f in features if f in train_data.columns]  # 只保留训练数据中也存在的特征
    print(f"调整后使用的特征: {features}")  # 打印调整后的特征列表
    
    if len(features) == 0:  # 若调整后没有任何可用特征
        raise ValueError("训练数据中没有可用的特征！")  # 抛出 ValueError 终止程序

if USE_WAVELET:  # 若模型使用小波特征
    print("\n使用小波变换特征...")  # 打印正在使用小波变换特征的提示
    
    # 获取小波参数
    wavelet_type = checkpoint.get('wavelet', 'db4')  # 从小波配置读取小波基函数，默认 db4
    wavelet_level = checkpoint.get('level', 3)  # 读取小波分解层数，默认 3
    wavelet_target_col = checkpoint.get('wavelet_target_col', '温度偏差')  # 获取小波分解的目标列  # 读取小波分解的目标列，默认温度偏差
    feature_names = checkpoint.get('feature_names', None)  # 读取训练时保存的小波特征名列表
    
    print(f"  小波基函数: {wavelet_type}")  # 打印本次使用的小波基函数
    print(f"  分解层数: {wavelet_level}")  # 打印本次使用的分解层数
    print(f"  小波分解目标列: {wavelet_target_col}")  # 打印本次分解的目标列
    print(f"  数据泄漏防护: {'是' if NO_LEAKAGE else '否'}")  # 打印模型是否启用了数据泄漏防护
    
    # 根据模型版本选择小波分解方法
    if NO_LEAKAGE:  # 若当前模型是无数据泄漏版本
        # 使用滑动窗口小波分解（无数据泄漏，仅分解温度偏差）
        print("\n对训练数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 提示对训练数据进行滑动窗口小波分解
        train_wavelet_features = wavelet_decompose_sliding(train_data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, plot_decomposition=False)  # 对训练特征执行无泄漏滑动窗口小波分解
        
        print("\n对测试数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 提示对测试数据进行滑动窗口小波分解
        test_wavelet_features = wavelet_decompose_sliding(data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, plot_decomposition=True, plot_feature=wavelet_target_col)  # 对测试特征做分解，同时绘制分解效果图
    else:  # 否则说明是旧版小波模型
        # 使用传统小波分解（可能有数据泄漏）
        print("\n⚠️  使用传统小波分解（可能存在数据泄漏）")  # 提示将使用可能存在数据泄漏的传统小波分解
        
        def wavelet_decompose_legacy(data, wavelet='db4', level=3, plot_decomposition=False, plot_feature='温度偏差'):  # 定义传统小波分解函数，用于向后兼容旧模型
            """传统小波分解方法（向后兼容旧模型）

            对整段信号直接做小波分解并重构回原长度，不使用滑动窗口，因此可能引入未来信息。

            参数:
                data: 输入特征表（DataFrame）
                wavelet: 小波基函数，默认 'db4'
                level: 分解层数，默认 3
                plot_decomposition: 是否绘制分解效果图
                plot_feature: 要绘制的特征名
            返回:
                小波重构后的特征表（DataFrame）
            """
            wavelet_features = pd.DataFrame()  # 创建用于保存小波特征的结果表
            plot_data = {}  # 初始化用于绘图的字典
            
            for col in data.columns:  # 遍历输入数据的每一列
                if col == 'timestamp':  # 跳过时间戳列
                    continue  # 进入下一列
                signal = data[col].values  # 取出当前列的数值数组
                coeffs = pywt.wavedec(signal, wavelet, level=level)  # 对整段信号直接做 level 层小波分解
                approx = coeffs[0]  # 取出近似系数
                approx_recon = pywt.upcoef('a', approx, wavelet, level=level, take=len(signal))  # 用 upcoef 把近似系数重构回原信号长度
                wavelet_features[f'{col}_approx'] = approx_recon[:len(signal)]  # 保存近似重构结果作为新特征列
                
                detail_recons = []  # 初始化细节系数重构结果列表
                for i, detail in enumerate(coeffs[1:], 1):  # 从第一层开始遍历各层细节系数
                    detail_recon = pywt.upcoef('d', detail, wavelet, level=level-i+1, take=len(signal))  # 把第 i 层细节系数重构回原信号长度
                    wavelet_features[f'{col}_detail{i}'] = detail_recon[:len(signal)]  # 保存第 i 层细节重构结果作为特征列
                    detail_recons.append(detail_recon[:len(signal)])  # 记录第 i 层细节重构结果供后续绘图
                
                # 保存指定特征的数据用于绘图
                if plot_decomposition and col == plot_feature:  # 若要求绘图且当前列是指定的绘图特征
                    plot_data['col_name'] = col  # 记录要绘图的列名
                    plot_data['original'] = signal  # 保存原始信号
                    plot_data['approx'] = approx_recon[:len(signal)]  # 保存近似系数重构结果
                    plot_data['details'] = detail_recons  # 保存各层细节系数重构结果
                    print(f"已保存 '{col}' 的小波分解数据用于绘图")  # 打印绘图数据已保存的提示
            
            if plot_decomposition and plot_data:  # 若已经收集到绘图数据
                plot_wavelet_decomposition(plot_data, wavelet, level)  # 绘制小波分解效果图
            elif plot_decomposition and not plot_data:  # 若要求绘图但未收集到数据
                print(f"⚠️  警告：未找到特征 '{plot_feature}'，无法绘制分解图")  # 打印未找到绘图特征的警告
            
            return wavelet_features  # 返回小波重构后的特征表
        
        train_wavelet_features = wavelet_decompose_legacy(train_data[features], wavelet=wavelet_type, level=wavelet_level, plot_decomposition=False)  # 用传统小波分解处理训练数据
        test_wavelet_features = wavelet_decompose_legacy(data[features], wavelet=wavelet_type, level=wavelet_level, plot_decomposition=True, plot_feature='温度偏差')  # 用传统小波分解处理测试数据并绘制分解图
    
    # 直接使用小波分解后的特征（已包含原始特征和小波特征）
    train_combined = train_wavelet_features  # 训练集直接使用小波分解后的特征
    test_combined = test_wavelet_features  # 测试集直接使用小波分解后的特征
    
    # 归一化（如果模型保存了scaler，直接使用）
    if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若检查点中保存了特征与目标归一化器
        print("\n使用模型保存的归一化器...")  # 打印使用模型自带归一化器的提示
        feature_scaler = checkpoint['feature_scaler']  # 取出模型保存的特征归一化器
        target_scaler = checkpoint['target_scaler']  # 取出模型保存的目标归一化器
    else:  # 否则需要重新拟合归一化器
        print("\n使用训练数据重新拟合归一化器...")  # 打印将重新拟合归一化器的提示
        feature_scaler = MinMaxScaler()  # 创建特征归一化器
        target_scaler = MinMaxScaler()  # 创建目标归一化器
        feature_scaler.fit(train_combined)  # 在训练特征上拟合特征归一化器
        target_scaler.fit(train_data[[target]])  # 在训练目标值上拟合目标归一化器
    
    # 保存原始列名
    original_columns = test_combined.columns  # 保存变换前的列名，供重建 DataFrame 使用
    
    test_combined = pd.DataFrame(  # 用特征归一化器变换测试数据
        feature_scaler.transform(test_combined),
        columns=original_columns
    )
    
    print(f"  特征数量: {len(test_combined.columns)}")  # 打印测试数据的特征数量
    print(f"  特征列表: {list(test_combined.columns)}")  # 打印测试数据的特征列表
    
else:  # 若模型使用原始特征（非小波）
    print("\n使用原始特征...")  # 打印使用原始特征的提示
    
    # 普通模型：仅使用原始特征
    feature_scaler = MinMaxScaler()  # 创建特征归一化器
    target_scaler = MinMaxScaler()  # 创建目标归一化器
    
    feature_scaler.fit(train_data[features])  # 在训练数据原始特征上拟合归一化器
    target_scaler.fit(train_data[[target]])  # 在训练目标值上拟合目标归一化器
    
    test_combined = pd.DataFrame(  # 变换测试数据的原始特征
        feature_scaler.transform(data[features]),
        columns=features
    )
    
    print(f"  特征数量: {len(features)}")  # 打印最终特征数量

# ===================== 序列构造 =====================
pred_offset = 6  # 定义预测偏移步数（6 步 × 5 分钟 = 30 分钟）

def create_sequences(data, time_step):  # 定义时间窗口序列构造函数
    """
    创建时间序列样本
    注意：这里是对单个数据集（机器人新数据）进行序列构造，不涉及训练/测试划分
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动切窗并预留预测偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行作为一个输入样本
    return np.array(sequences)  # 把序列列表堆叠成三维 numpy 数组并返回

print("\n创建序列数据...")  # 提示开始创建序列数据
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 用目标归一化器变换测试集的目标值

# 创建序列
X_test = create_sequences(test_combined, time_step)  # 生成测试集的输入序列
y_test = data_target_normalized[time_step + pred_offset:].flatten()  # 取偏移之后的目标值并展平为一维标签
X_test = X_test[:len(y_test)]  # 确保长度匹配  # 截断输入序列，使其与标签数量一致

print(f"\n测试数据形状:")  # 打印测试数据形状标题
print(f"  X_test: {X_test.shape}")  # 打印测试输入的形状
print(f"  y_test: {y_test.shape}")  # 打印测试标签的形状

# ===================== 构建模型 =====================
if USE_WAVELET:  # 若使用小波特征模型
    model = GRUWaveletModel(  # 实例化小波版模型并移动到指定设备
        input_dim=input_dim,
        time_step=time_step,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.1
    ).to(device)
else:  # 否则使用 GRU-Only 模型
    model = GRUOnlyModel(  # 实例化 GRU-Only 模型并移动到指定设备
        input_dim=input_dim,
        time_step=time_step,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.1
    ).to(device)

model.load_state_dict(checkpoint['model_state_dict'])  # 从检查点加载训练好的模型权重
model.eval()  # 把模型切换为评估模式

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型的参数总数

# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始进行预测
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试输入转换为张量并移动到指定设备

with torch.no_grad():  # 关闭梯度计算以加速推理
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并把结果从设备转回 numpy 数组

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 对测试集真实目标值做反归一化
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 对模型预测值做反归一化

# 计算评估指标
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 对 MSE 开平方得到均方根误差 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

if USE_WAVELET:  # 若模型使用小波特征
    model_name = "GRU+Wavelet+ECA (No Leakage)" if NO_LEAKAGE else "GRU+Wavelet+ECA"  # 按是否无泄漏选择对应的模型名称
else:  # 否则使用原始特征模型
    model_name = "GRU-Only"  # 把模型名称设为 GRU-Only

print("\n" + "=" * 60)  # 打印分隔线
print(f"{model_name} Model Test Results")  # 打印模型测试结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R² 指标
print(f"MSE:       {mse:.4f}")  # 打印 MSE 指标
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE 指标
print(f"MAE:       {mae:.4f}")  # 打印 MAE 指标
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
# 只绘制前120个时间步长
plot_length = min(120, len(y_test_rescaled))  # 只绘制前 120 个时间步，避免曲线过密
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的预测值
# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成绘图横轴的步长索引
plt.figure(figsize=(15, 8))  # 创建预测对比图画布
y_test_flat = y_test_plot.flatten()  # 把真实值展平为一维数组
y_pred_flat = y_pred_plot.flatten()  # 把预测值展平为一维数组
plt.tick_params(axis='both', labelsize=16)  # 把坐标轴刻度字号设为 16
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制模型预测值曲线
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签为时间步
plt.ylabel('TD (℃)', fontsize=16)  # 设置纵轴标签为温度偏差（摄氏度）
plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)  # 设置标题并展示 R² 与 RMSE 指标
plt.legend(fontsize=16)  # 显示图例并设置字号
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.tight_layout()  # 调整图表布局
plt.text(0.5, 0.95, '(f)', transform=plt.gca().transAxes,  # 在图表顶部中央添加子图编号 (f)
         fontsize=16, fontweight='bold', va='top', ha='center')  # 设置子图编号文本的对齐方式与加粗样式
plt.tight_layout()  # 再次调整布局以容纳新增文本
if USE_WAVELET:  # 按模型类型选择预测图的输出文件名
    output_filename = 'test_GRUWavelet_no_leakage_prediction.png' if NO_LEAKAGE else 'test_GRUWavelet_prediction.png'  # 无泄漏小波模型对应的输出文件名
else:  # 否则采用其它命名
    output_filename = 'test_GRUOnly_prediction.png'  # GRU-Only 模型对应的输出文件名
plt.savefig(output_filename, dpi=150, bbox_inches='tight')  # 保存预测对比图到指定文件
print(f"\n预测结果图已保存: {output_filename}")  # 打印预测图保存提示
plt.close()  # 关闭画布释放内存

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算预测误差：预测值减去真实值

plt.figure(figsize=(15, 10))  # 创建误差分析画布

# 子图1: 预测值 vs 真实值散点图
plt.subplot(2, 2, 1)  # 子图1：预测值与真实值的散点图
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, s=20)  # 绘制预测值与真实值的散点
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],  # 绘制 y=x 的理想预测参考线
         [y_test_rescaled.min(), y_test_rescaled.max()],
         'r--', lw=2, label='理想预测线')
plt.xlabel('真实温度偏差 (℃)', fontsize=11)  # 设置散点图横轴标签
plt.ylabel('预测温度偏差 (℃)', fontsize=11)  # 设置散点图纵轴标签
plt.title(f'预测值 vs 真实值 (R²={r2:.4f})', fontsize=12)  # 设置散点图标题并显示 R²
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格线
# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 子图2：误差分布直方图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差直方图，分 50 个区间
plt.xlabel('预测误差 (℃)', fontsize=11)  # 设置直方图横轴标签
plt.ylabel('频数', fontsize=11)  # 设置直方图纵轴标签为频数
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 设置直方图标题并显示 MAE
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 用红色虚线标出零误差位置
plt.grid(True, alpha=0.3, axis='y')  # 沿 y 轴方向添加网格线
# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 子图3：误差随时间变化的时序图
error_steps = np.arange(min(120, len(errors)))  # 生成误差时序图的横轴索引
plt.plot(error_steps, errors[:120], alpha=0.7, linewidth=1)  # 绘制前 120 步的误差曲线
plt.xlabel('Timestep', fontsize=11)  # 设置时序图横轴标签
plt.ylabel('Prediction Error (℃)', fontsize=11)  # 设置时序图纵轴标签
plt.title('Error Time Series', fontsize=12)  # 设置时序图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 用红色虚线标出零误差水平线
plt.grid(True, alpha=0.3)  # 添加网格线
# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 子图4：误差箱线图
box_data = [errors]  # 把误差组织为箱线图的输入列表
plt.boxplot(box_data, tick_labels=[model_name])  # 绘制误差箱线图并标注模型名称
plt.ylabel('预测误差 (℃)', fontsize=11)  # 设置箱线图纵轴标签
plt.title('误差统计分布', fontsize=12)  # 设置箱线图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 用红色虚线标出零误差水平线
plt.grid(True, alpha=0.3, axis='y')  # 沿 y 轴方向添加网格线
plt.tight_layout()  # 调整误差分析图的布局
if USE_WAVELET:  # 按模型类型选择误差图的输出文件名
    error_filename = 'test_GRUWavelet_no_leakage_error_analysis.png' if NO_LEAKAGE else 'test_GRUWavelet_error_analysis.png'  # 无泄漏小波模型对应的误差图文件名
else:  # 否则采用其它命名
    error_filename = 'test_GRUOnly_error_analysis.png'  # GRU-Only 模型对应的误差图文件名
plt.savefig(error_filename, dpi=150, bbox_inches='tight')  # 保存误差分析图到指定文件
print(f"误差分析图已保存: {error_filename}")  # 打印误差分析图保存提示
plt.close()  # 关闭画布释放内存
# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("Error Statistics")  # 打印误差统计标题
print("=" * 60)  # 打印分隔线
print(f"Mean Error:     {np.mean(errors):.4f} C")  # 打印误差的平均值
print(f"Std Deviation:  {np.std(errors):.4f} C")  # 打印误差的标准差
print(f"Max Error:      {np.max(errors):.4f} C")  # 打印误差的最大值
print(f"Min Error:      {np.min(errors):.4f} C")  # 打印误差的最小值
print(f"Median Error:   {np.median(errors):.4f} C")  # 打印误差的中位数
print("=" * 60)  # 打印分隔线

print("\nTest Completed!")  # 打印测试全部完成的提示






