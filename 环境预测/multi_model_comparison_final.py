"""
多模型对比脚本 - 完全按照训练逻辑

本脚本按与训练时一致的逻辑加载多个已训练模型（LSTM、TCN、GRU 及其与 ECA/小波的组合），
在同一份测试数据上完成推理：先做缺失值处理与特征对齐，必要时重建归一化器；
再对测试特征做滑动窗口切分、反归一化并计算 R2/RMSE/MAE；
最后把各模型预测曲线与真实值画在同一张图中，并在终端输出性能对比表。

"""
import pandas as pd  # 导入 pandas 数据分析库
import numpy as np  # 导入 numpy 数值计算库
import matplotlib  # 导入 matplotlib 绘图库
matplotlib.use('Agg')  # 指定 Agg 后端以便无界面保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 归一化工具
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、均方误差与平均绝对误差评估函数
import torch  # 导入 PyTorch 主模块
import torch.nn as nn  # 导入 PyTorch 神经网络子模块并简写为 nn
import pywt  # 导入 pywt 用于小波分解
import warnings  # 导入 warnings 用于忽略告警
import os  # 导入 os 用于路径处理
warnings.filterwarnings('ignore')  # 忽略运行过程中的所有警告信息

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据是否有可用 GPU 选择计算设备
print(f"使用设备: {device}")  # 打印当前使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示

# ===================== 模型定义 =====================
class ECALayer(nn.Module):  # 定义 ECA 通道注意力层
    """
    ECA（Efficient Channel Attention）通道注意力层。
    通过自适应平均池化与一维卷积学习各通道权重，对输入特征做通道重标定。
    """
    def __init__(self, channels, gamma=2, b=1):  # 定义构造函数，接收通道数与核大小调节参数
        """
        初始化 ECA 层。
        参数 channels 为输入通道数；gamma 与 b 用来计算一维卷积核大小，
        使卷积核能随通道数自适应变化。无返回值。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        t = int(abs((np.log2(channels) + b) / gamma))  # 根据通道数计算自适应卷积核大小参数 t
        k_size = t if t % 2 else t + 1  # 保证卷积核大小为奇数
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 定义自适应平均池化层，把通道压缩为 1 个值
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 用一维卷积学习通道间依赖关系，padding 保持长度不变
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活函数把权重限制在 0 到 1

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：对输入特征做通道注意力加权。
        参数 x 形状为 (batch, seq_len, channels)；
        返回同形状、各通道被重新加权后的张量。
        """
        x_permuted = x.permute(0, 2, 1)  # 把输入维度置换为 (batch, channels, seq_len) 以便做通道卷积
        y = self.avg_pool(x_permuted)  # 对每个通道做全局平均池化
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 通过一维卷积并转置回原维度，得到各通道权重
        y = self.sigmoid(y)  # 用 Sigmoid 把权重映射到 0 到 1
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将权重广播乘回各通道，实现通道重标定
        return x_permuted.permute(0, 2, 1)  # 把维度置换回 (batch, seq_len, channels) 后返回

class GRUModel(nn.Module):  # 定义基础 GRU 回归模型
    """
    基础 GRU 回归模型：GRU 层加全连接层，取最后时间步输出进行预测。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 定义构造函数，接收输入维度、时间步与网络超参
        """
        初始化模型。
        参数 input_dim 为输入特征维度，time_step 为窗口长度，
        gru_hidden 为隐藏层尺寸，gru_layers 为 GRU 层数，dropout 为丢弃率。
        无返回值。
        """
        super(GRUModel, self).__init__()  # 调用父类初始化方法
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层，多层时启用 dropout 且批次维度在前
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅当 GRU 层数大于 1 时才设置 dropout
        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接层把隐状态映射为单个预测值
    def forward(self, x):  # 定义前向传播方法
        """
        前向传播。
        参数 x 为形状 (batch, time_step, input_dim) 的输入序列；
        返回形状 (batch, 1) 的预测值。
        """
        out, _ = self.gru(x)  # 运行 GRU 取得输出序列，忽略最终隐状态
        return self.fc(out[:, -1, :])  # 取序列最后一个时间步的输出接全连接层得到预测
class GRUECAModel(nn.Module):  # 定义 GRU 加 ECA 注意力模型
    """
    GRU 加 ECA 通道注意力模型：在 GRU 之前先对特征做通道重标定。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 定义构造函数，参数同 GRUModel
        """
        初始化模型，参数含义与 GRUModel 一致。
        无返回值。
        """
        super(GRUECAModel, self).__init__()  # 调用父类初始化方法
        self.eca = ECALayer(channels=input_dim)  # 在 GRU 之前加入 ECA 通道注意力层
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层，多层时启用 dropout 且批次维度在前
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅当层数大于 1 时才设置 dropout
        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接层输出单个预测值

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：先经 ECA 再经 GRU，取最后时间步输出预测。
        参数 x 为输入序列张量；返回形状 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 先对输入应用 ECA 通道注意力
        out, _ = self.gru(x)  # 将注意力加权后的序列送入 GRU
        return self.fc(out[:, -1, :])  # 取最后时间步输出接全连接层得到预测

class GRUWaveletModel(nn.Module):  # 定义 GRU 加小波分解模型（不含 ECA）
    """GRU + 小波分解（不带ECA）"""  # 补充：输入应为小波分解后的特征，其余结构与 GRUModel 相同
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 定义构造函数，参数同 GRUModel
        """
        初始化模型，参数含义与 GRUModel 一致。
        无返回值。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类初始化方法
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层，多层时启用 dropout 且批次维度在前
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅当层数大于 1 时才设置 dropout
        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接层输出单个预测值

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：运行 GRU 并取最后时间步输出预测。
        参数 x 为小波特征序列；返回形状 (batch, 1) 的预测值。
        """
        out, _ = self.gru(x)  # 运行 GRU 取得输出序列
        return self.fc(out[:, -1, :])  # 取最后时间步输出接全连接层得到预测

class GRUWaveletECAModel(nn.Module):  # 定义 GRU 加小波加 ECA 模型，结构与训练时一致
    """GRU + 小波分解 + ECA，使用 ModuleList 与训练时结构一致"""  # 补充：使用 ModuleList 堆叠多层 GRU，输入为小波分解与 ECA 后的特征
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 定义构造函数，gru_hidden 可为整数或整数列表
        """
        初始化模型。
        参数 gru_hidden 可为整数或整数列表；若为列表则为每个元素各建一层 GRU，
        gru_layers 为层数，dropout 为丢弃率。无返回值。
        """
        super(GRUWaveletECAModel, self).__init__()  # 调用父类初始化方法
        self.eca = ECALayer(channels=input_dim)  # 在 GRU 之前加入 ECA 通道注意力层
  # 判断隐藏层尺寸是以列表还是单个整数给出
        if isinstance(gru_hidden, list):  # 若为列表则直接采用该隐藏层尺寸配置
            hidden_sizes = gru_hidden  # 否则复制 gru_layers 份相同隐藏层尺寸
        else:
            hidden_sizes = [gru_hidden] * gru_layers  # 创建用于存放多个 GRU 层的 ModuleList
  # 记录当前层的输入维度
        self.gru_layers_list = nn.ModuleList()  # 逐个隐藏层尺寸构建 GRU 并加入列表
        in_size = input_dim  # 遍历配置的每个隐藏层尺寸
        for h_size in hidden_sizes:  # 向 ModuleList 追加单层 GRU
            self.gru_layers_list.append(  # 定义单层 GRU，输入维度随上一层输出变化
                nn.GRU(input_size=in_size, hidden_size=h_size, num_layers=1,  # 单层 GRU 不使用 dropout，批次维度在前，单向
                       dropout=0.0, batch_first=True, bidirectional=False)  # 关闭双向，保持与训练结构一致
            )  # 把下一层的输入维度更新为当前隐藏层尺寸
            in_size = h_size
  # 定义 Dropout 层用于正则化
        self.dropout = nn.Dropout(dropout)  # 定义全连接层，输入维度为最后一层隐藏尺寸
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：依次经过 ECA 与多个 GRU 子层，最后取末时间步输出预测。
        参数 x 为输入序列张量；返回形状 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 先应用 ECA 通道注意力
        out = x  # 初始化循环输入为注意力输出
        for gru in self.gru_layers_list:  # 依次通过每一个 GRU 子层
            out, _ = gru(out)  # 运行当前 GRU 层并更新输出
            out = self.dropout(out)  # 每层 GRU 之后施加 Dropout
        return self.fc(out[:, -1, :])  # 取最后时间步输出接全连接层得到预测

class LSTMModel(nn.Module):  # 定义基础 LSTM 回归模型
    """
    基础 LSTM 回归模型：LSTM 层加全连接层，取最后时间步输出预测。
    """
    def __init__(self, input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.3):  # 定义构造函数，接收输入维度、时间步与网络超参
        """
        初始化模型。
        参数 input_dim 为输入特征维度，time_step 为窗口长度，
        lstm_hidden 为隐藏尺寸，lstm_layers 为层数，dropout 为丢弃率。
        """
        super(LSTMModel, self).__init__()  # 调用父类初始化方法
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,  # 定义 LSTM 层，多层时启用 dropout 且批次维度在前
                            dropout=dropout if lstm_layers > 1 else 0.0, batch_first=True)  # 仅当层数大于 1 时才设置 dropout
        self.fc = nn.Linear(lstm_hidden, 1)  # 定义全连接层输出单个预测值

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播。
        参数 x 为形状 (batch, time_step, input_dim) 的序列；
        返回形状 (batch, 1) 的预测值。
        """
        out, _ = self.lstm(x)  # 运行 LSTM 取得输出序列，忽略最终隐状态
        return self.fc(out[:, -1, :])  # 取最后时间步输出接全连接层得到预测

class Chomp1d(nn.Module):  # 定义裁剪模块，用于 TCN 中去掉因果卷积产生的多余时间步
    """
    裁剪模块：去掉因果卷积在序列末尾引入的多余时间步。
    """
    def __init__(self, chomp_size):  # 定义构造函数，接收要裁剪的长度
        """
        初始化裁剪模块。
        参数 chomp_size 为要裁掉的末尾时间步数量。无返回值。
        """
        super(Chomp1d, self).__init__()  # 调用父类初始化方法
        self.chomp_size = chomp_size  # 保存需要裁剪的时间步数

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：裁掉输入末尾 chomp_size 个时间步。
        参数 x 为一维卷积输出张量；返回裁剪后且内存连续的张量。
        """
        return x[:, :, :-self.chomp_size].contiguous()  # 裁掉末尾多余的 chomp_size 个时间步并保证内存连续

class TemporalBlock(nn.Module):  # 定义 TCN 的时间卷积块，含残差连接
    """
    TCN 时间卷积块：两层空洞因果卷积加裁剪、激活与丢弃，并带残差连接。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):  # 定义构造函数，接收输入输出通道数、卷积核、步长、膨胀率与 padding
        """
        初始化时间卷积块。
        参数 n_inputs/n_outputs 为输入输出通道数，kernel_size 为卷积核大小，
        stride 为步长，dilation 为膨胀率，padding 为填充长度，dropout 为丢弃率。
        """
        super(TemporalBlock, self).__init__()  # 调用父类初始化方法
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)  # 第一层空洞因果卷积
        self.chomp1 = Chomp1d(padding)  # 对应的裁剪层，去掉卷积引入的尾部冗余
        self.relu1 = nn.ReLU()  # 第一层 ReLU 激活
        self.dropout1 = nn.Dropout(dropout)  # 第一层 Dropout
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)  # 第二层空洞因果卷积
        self.chomp2 = Chomp1d(padding)  # 第二层对应的裁剪层
        self.relu2 = nn.ReLU()  # 第二层 ReLU 激活
        self.dropout2 = nn.Dropout(dropout)  # 第二层 Dropout
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,  # 把卷积、裁剪、激活与丢弃按顺序组成串行网络
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 串行模块的第二组层
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 当输入输出通道数不同时用 1x1 卷积做下采样对齐维度，否则不做处理
        self.relu = nn.ReLU()  # 定义残差相加后的 ReLU 激活

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：主分支输出与残差相加后经 ReLU 返回。
        参数 x 为形状 (batch, channels, seq_len) 的张量；返回同形状的张量。
        """
        out = self.net(x)  # 通过主分支网络得到输出
        res = x if self.downsample is None else self.downsample(x)  # 计算残差分支：通道数不同时先做下采样
        return self.relu(out + res)  # 把主分支与残差相加并经过 ReLU 后返回

class TCNModel(nn.Module):  # 定义 TCN 回归模型
    """
    TCN 回归模型：由多个 TemporalBlock 堆叠而成，取末时间步输出预测。
    """
    def __init__(self, input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.3):  # 定义构造函数，接收通道数配置列表等超参
        """
        初始化模型。
        参数 input_dim 为输入特征维度，num_channels 为各层通道数列表，
        kernel_size 为卷积核大小，dropout 为丢弃率。
        """
        super(TCNModel, self).__init__()  # 调用父类初始化方法
        layers = []  # 初始化用于存放各时间块的列表
        for i in range(len(num_channels)):  # 按通道配置列表逐层构建时间卷积块
            dilation_size = 2 ** i  # 膨胀系数按 2 的幂次递增
            in_channels = input_dim if i == 0 else num_channels[i-1]  # 第一层的输入通道数为输入特征维度
            out_channels = num_channels[i]  # 否则输入通道数为上一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,  # 向列表追加一个 TemporalBlock，设置步长与膨胀率
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]  # padding 由卷积核大小与膨胀率共同决定
        self.network = nn.Sequential(*layers)  # 把所有时间块按顺序组成网络
        self.fc = nn.Linear(num_channels[-1], 1)  # 定义全连接层把最后通道映射为单个预测值

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播：把输入转为通道在前后送入 TCN 网络并预测。
        参数 x 为形状 (batch, time_step, input_dim) 的序列；
        返回形状 (batch, 1) 的预测值。
        """
        x = x.transpose(1, 2)  # 把输入维度置换为 (batch, channels, seq_len)
        out = self.network(x)  # 通过由多个时间块组成的 TCN 网络
        return self.fc(out[:, :, -1])  # 取最后一个时间步的输出接全连接层得到预测

# ===================== 小波分解函数 =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):  # 定义滑动窗口小波分解函数
    """滑动窗口小波分解 - 不保留原始TD列"""  # 补充：逐列处理数据，对目标列用滑动窗口提取近似与细节系数特征，其余列原样保留；返回小波特征 DataFrame
    if window_size is None:  # 未指定窗口大小时按分解层数推算，且不超过数据总长
        window_size = min(2 ** (level + 6), len(data))  # 根据 level 自动计算默认窗口长度
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空 DataFrame 存放小波特征
    
    for col in data.columns:  # 逐列遍历输入数据
        if col == 'timestamp':  # 跳过时间戳列，不对其做小波分解
            continue  # 跳过当前列
        
        signal = data[col].values  # 取出当前列的一维信号数组
        
        if col == target_col:  # 仅对目标列进行小波分解
            approx_list = []  # 初始化近似系数列表
            detail_lists = [[] for _ in range(level)]  # 为每个分解层初始化细节系数列表
            
            for i in range(len(signal)):  # 逐个时间步构造滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证不越过数据开头
                window_signal = signal[start_idx:i+1]  # 截取从窗口起点到当前位置的信号片段
                
                if len(window_signal) < window_size:  # 窗口长度不足时用边缘值在左侧补齐
                    pad_width = window_size - len(window_signal)  # 计算需要补齐的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 使用 edge 模式补齐窗口
                
                try:  # 捕获小波分解过程中可能出现的异常
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对补齐后的窗口做多层离散小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为当前步特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层细节系数最后一个值，空则补 0
                except:  # 分解失败时的兜底处理
                    approx_list.append(window_signal[-1])  # 近似特征退化为窗口最后一个原始值
                    for j in range(level):  # 各层细节特征统一填 0
                        detail_lists[j].append(0.0)  # 填充当前层的细节特征
            
            # 不保留原始TD列，只添加小波特征  # 说明：不保留原始 TD 列，只保留小波派生特征
            wavelet_features[f'{col}_approx'] = approx_list  # 把近似系数写入一列特征
            for j in range(level):  # 逐层把细节系数写入对应列
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入第 j+1 层细节特征列
        else:  # 非目标列直接保留原信号
            wavelet_features[col] = signal  # 把原始列整体复制到特征表
    
    return wavelet_features  # 返回构造好的小波特征表

# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印加载测试数据提示
data = pd.read_csv('部分温湿度数据.csv', encoding='gbk')  # 以 GBK 编码读取测试 CSV
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m', origin=pd.Timestamp('2025/6/21 0:04'))  # 按采样间隔与起始时间生成时间戳列

features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 定义候选特征列名
target = '温度偏差'  # 指定目标列为温度偏差
features = [f for f in features if f in data.columns]  # 只保留数据中实际存在的特征列

# 缺失值处理
for col in ['温度偏差', '外界温度', 'CO2', '目标温度','相对湿度','TD']:  # 遍历需要处理的数值列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅对存在缺失值的列进行处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')  # 线性插值后以前向、后向填充补齐缺失值

for col in ['日龄']:  # 单独处理日龄这类离散列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅对存在缺失的日龄列处理
        data[col] = data[col].fillna(method='ffill').fillna(method='bfill')  # 用前向与后向填充补齐离散列缺失

data = data.dropna(subset=features + [target]).reset_index(drop=True)  # 删除特征或目标仍含缺失的行并重置索引
print(f"数据形状: {data.shape}")  # 打印清洗后的数据规模

# ===================== 路径解析 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 取得当前脚本所在目录作为基准路径

def resolve_model_path(rel_path):  # 定义模型权重路径解析函数
    """在常见目录中解析模型路径"""  # 补充：依次在脚本目录及其常见子目录中查找目标文件，返回第一个存在的路径；都找不到时返回首个候选路径
    candidates = [  # 列出所有可能的相对路径候选
        os.path.join(BASE_DIR, rel_path),  # 脚本目录下的候选路径
        os.path.join(BASE_DIR, '温度偏差预测', rel_path),  # 温度偏差预测子目录下的候选路径
        os.path.join(BASE_DIR, 'THI', rel_path),  # THI 子目录下的候选路径
    ]
    for p in candidates:  # 逐个候选路径检查是否存在
        if os.path.exists(p):  # 找到存在的路径立即返回
            return p  # 全部不存在时返回首个候选路径以便后续报错
    return candidates[0]

# ===================== 模型配置 =====================
models_config = [  # 定义待对比的模型配置列表
    {'name': 'LSTM', 'path': 'pth/Model_LSTM_pytorch.pth', 'color': '#FFA500', 'linestyle': '--'},  # LSTM 模型配置：名称、权重路径与绘图样式
    {'name': 'TCN', 'path': 'pth/Model_TCN_pytorch.pth', 'color': '#00CED1', 'linestyle': '--'},  # TCN 模型配置
    {'name': 'GRU', 'path': 'pth/Model_GRU_pytorch.pth', 'color': '#4169E1', 'linestyle': '--'},  # GRU 模型配置
    {'name': 'GRU+ECA', 'path': 'pth/Model_GRUECA_pytorch.pth', 'color': '#32CD32', 'linestyle': '--'},  # GRU+ECA 模型配置
    {'name': 'GRU+Wavelet', 'path': 'pth/Model_GRUWavelet_pytorch.pth', 'color': '#8B008B', 'linestyle': '--'},  # GRU+Wavelet 模型配置
    {'name': 'GRU+Wavelet+ECA', 'path': 'pth/Model_GRUWavelet_no_leakage_pytorch.pth', 'color': '#DC143C', 'linestyle': '--'},  # GRU+Wavelet+ECA 模型配置
]
# ===================== 加载模型并预测 =====================
predictions = {}  # 初始化保存各模型预测结果的字典
metrics = {}  # 初始化保存各模型评估指标的字典

for model_cfg in models_config:  # 遍历模型配置，逐个加载并预测
    print(f"\n{'='*60}")  # 打印 60 个等号的分隔线
    print(f"加载模型: {model_cfg['name']}")  # 打印当前正在加载的模型名称
    print(f"{'='*60}")  # 再次打印分隔线
    
    try:  # 捕获单个模型加载或预测过程中的异常
        model_path = resolve_model_path(model_cfg['path'])  # 解析出模型权重文件实际路径
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # 加载 checkpoint，映射到指定设备且允许加载完整对象
        
        time_step = checkpoint['time_step']  # 从检查点取出输入时间窗口长度
        pred_offset = checkpoint.get('pred_offset', 6)  # 默认6，但优先使用保存的值  # 补充：默认预测偏移为 6，但优先使用检查点保存的值
        feature_scaler = checkpoint.get('feature_scaler')  # 取出检查点中的特征归一化器
        target_scaler = checkpoint.get('target_scaler')  # 取出检查点中的目标归一化器
        
        print(f"  time_step: {time_step}, pred_offset: {pred_offset}")  # 打印时间窗口与预测偏移配置
        
        # 判断是否为小波模型
        is_wavelet = 'wavelet' in checkpoint or 'level' in checkpoint  # 若检查点含 wavelet 或 level 键则判定为小波模型
        has_eca = any('eca' in key for key in checkpoint['model_state_dict'].keys())  # 检查权重名中是否含 eca 判断是否带注意力层
        
        print(f"  is_wavelet: {is_wavelet}, has_eca: {has_eca}")  # 打印模型结构判定结果
        
        # 准备特征
        if is_wavelet:  # 如果是小波模型
            print(f"  小波模型")  # 打印小波模型提示
            wavelet_target_col = checkpoint.get('wavelet_target_col', target)  # 取检查点中保存的小波目标列名
            wavelet_features = wavelet_decompose_sliding(  # 对测试特征做滑动窗口小波分解
                data[features],  # 传入原始测试特征
                target_col=wavelet_target_col,  # 指定小波分解的目标列
                wavelet='db4',  # 指定小波基函数
                level=3,  # 指定分解层数
                window_size=time_step  # 窗口长度与训练时时间步保持一致
            )
            
            saved_feature_names = checkpoint.get('feature_names', list(wavelet_features.columns))  # 取检查点保存的特征名，缺省用全部小波特征列名
            test_features_df = wavelet_features[saved_feature_names]  # 按保存的特征顺序重排列，保证与训练一致
            print(f"  特征: {saved_feature_names}")  # 打印本次使用的特征名
        else:  # 非小波模型分支
            print(f"  非小波模型")  # 打印非小波模型提示
            test_features_df = data[features].copy()  # 直接复制原始测试特征
        
        # 归一化特征
        if feature_scaler is None:  # 若检查点未保存特征归一化器
            print(f"  警告：feature_scaler 为 None，需要重新创建")  # 打印需要重建归一化器的警告
            train_data = pd.read_csv('温湿度数据111.csv')  # 重新读取训练数据用于拟合归一化器
            feature_scaler = MinMaxScaler()  # 新建特征归一化器
            target_scaler = MinMaxScaler()  # 新建目标归一化器
            
            # 如果是小波模型，需要先对训练数据做小波分解
            if is_wavelet:  # 若为小波模型
                train_wavelet = wavelet_decompose_sliding(train_data[features], target_col='温度偏差', wavelet='db4', level=3, window_size=time_step)  # 对训练数据做同样的小波分解后拟合归一化器
                feature_scaler.fit(train_wavelet[saved_feature_names])  # 用小波特征拟合特征归一化器
            else:  # 非小波模型分支
                feature_scaler.fit(train_data[features])  # 用训练数据原始特征拟合归一化器
            target_scaler.fit(train_data[[target]])  # 用训练目标拟合目标归一化器
        else:  # 检查点已保存归一化器的分支
            print(f"  使用模型保存的 feature_scaler")  # 打印使用已保存归一化器的提示
        
        test_features_normalized = pd.DataFrame(  # 用归一化器转换测试特征并重建 DataFrame
            feature_scaler.transform(test_features_df),  # 对测试特征执行 transform
            columns=test_features_df.columns  # 指定转换后 DataFrame 的列名
        )
        
        # 创建序列
        def create_sequences(data, time_step, pred_offset):  # 定义把归一化后的数据切分成滑动窗口序列的函数
            """
            把归一化后的数据切分为滑动窗口序列。
            参数 data 为归一化后的特征表，time_step 为窗口长度，
            pred_offset 为预测偏移；返回形状 (样本数, time_step, 特征数) 的三维数组。
            """
            sequences = []  # 初始化序列列表
            for i in range(len(data) - time_step - pred_offset):  # 遍历可构造序列的起点
                sequences.append(data.iloc[i:i + time_step].values)  # 取时间步长度的特征窗口并转为数组
            return np.array(sequences)  # 返回由所有序列堆叠成的三维数组
        
        X_test = create_sequences(test_features_normalized, time_step, pred_offset)  # 构造测试集输入序列 X_test
        
        # 关键：y_true 对齐，与训练脚本保持一致（time_step + pred_offset - 1）
        y_true_normalized = target_scaler.transform(data[[target]]).flatten()  # 把真实目标归一化并展平为一维数组
        y_true_normalized_seq = y_true_normalized[time_step + pred_offset - 1:time_step + pred_offset - 1 + len(X_test)]  # 按 time_step + pred_offset - 1 的偏移对齐真实值，与训练脚本保持一致
        
        print(f"  X_test: {X_test.shape}, y_true: {y_true_normalized_seq.shape}")  # 打印输入与真实值的形状
        
        # 构建模型
        input_dim = X_test.shape[2]  # 由输入序列第三维得到模型输入特征数
        
        if is_wavelet and has_eca:  # 同时是小波模型且带 ECA
            model = GRUWaveletECAModel(input_dim, time_step).to(device)  # 实例化 GRU+Wavelet+ECA 模型并移到设备
        elif is_wavelet and not has_eca:  # 是小波模型但不带 ECA
            model = GRUWaveletModel(input_dim, time_step).to(device)  # 实例化 GRU+Wavelet 模型并移到设备
        elif has_eca:  # 带 ECA 但非小波
            model = GRUECAModel(input_dim, time_step).to(device)  # 实例化 GRU+ECA 模型并移到设备
        elif 'lstm' in model_cfg['path'].lower():  # 路径名含 lstm 时选择 LSTM 模型
            model = LSTMModel(input_dim, time_step).to(device)  # 实例化 LSTM 模型并移到设备
        elif 'tcn' in model_cfg['path'].lower():  # 路径名含 tcn 时选择 TCN 模型
            model = TCNModel(input_dim, time_step).to(device)  # 实例化 TCN 模型并移到设备
        else:  # 其余情况使用基础 GRU
            model = GRUModel(input_dim, time_step).to(device)  # 实例化 GRU 模型并移到设备
        
        # 调试：检查模型权重形状
        if model_cfg['name'] == 'GRU+Wavelet':  # 对 GRU+Wavelet 模型打印权重与结构形状用于调试
            print(f"  模型权重形状:")  # 打印权重形状提示
            for name, param in checkpoint['model_state_dict'].items():  # 遍历检查点权重并打印形状
                print(f"    {name}: {param.shape}")  # 打印单个权重名与形状
            print(f"  模型结构:")  # 打印模型结构提示
            for name, param in model.named_parameters():  # 遍历模型参数并打印形状
                print(f"    {name}: {param.shape}")  # 打印单个参数名与形状
        
        # 加载模型权重，忽略不匹配的键
        model_state_dict = checkpoint['model_state_dict']  # 取出检查点中的模型权重字典
        model_dict = model.state_dict()  # 取得当前模型的参数字典
        
        # 过滤掉不匹配的键
        filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}  # 过滤掉模型中不存在或形状不匹配的键
        
        # 检查是否有未加载的键
        missing_keys = set(model_dict.keys()) - set(filtered_state_dict.keys())  # 计算模型中有但过滤后未加载的键
        unexpected_keys = set(model_state_dict.keys()) - set(model_dict.keys())  # 计算检查点中模型不存在的多余键
        
        if missing_keys:  # 若存在未加载的键则告警
            print(f"  警告: 以下键未从checkpoint加载: {missing_keys}")  # 打印未加载键集合
        if unexpected_keys:  # 若存在多余键则告警
            print(f"  警告: 以下键在checkpoint中但模型中不存在(已忽略): {unexpected_keys}")  # 打印被忽略的多余键集合
        
        model_dict.update(filtered_state_dict)  # 用过滤后的权重更新当前模型参数字典
        model.load_state_dict(model_dict)  # 把更新后的参数加载进模型
        model.eval()  # 将模型切换为评估模式，关闭 Dropout 等训练专属层
        
        # 预测（归一化空间）
        X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试序列转换为浮点张量并移到设备
        with torch.no_grad():  # 关闭梯度计算以节省显存并加速推理
            y_pred_normalized = model(X_test_tensor).cpu().numpy().flatten()  # 调用模型预测并把结果取回 CPU 转为 numpy 一维数组
        
        # 反归一化到原始空间
        y_pred = target_scaler.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()  # 把预测结果反归一化回原始温度偏差量纲
        y_true = target_scaler.inverse_transform(y_true_normalized_seq.reshape(-1, 1)).flatten()  # 把对齐后的真实值反归一化回原始量纲
        
        # 调试输出
        if model_cfg['name'] in ['GRU+Wavelet', 'GRU+Wavelet+ECA']:  # 对小波相关模型额外打印调试信息
            print(f"  调试信息:")  # 打印调试信息标题
            print(f"    y_pred 前5个值: {y_pred[:5]}")  # 打印预测值前五个
            print(f"    y_true 前5个值: {y_true[:5]}")  # 打印真实值前五个
            print(f"    y_pred 范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")  # 打印预测值取值范围
            print(f"    y_true 范围: [{y_true.min():.4f}, {y_true.max():.4f}]")  # 打印真实值取值范围
        
        # 保存结果
        predictions[model_cfg['name']] = {'y_pred': y_pred, 'y_true': y_true}  # 保存当前模型的预测与真实值
        
        # 计算指标
        r2 = r2_score(y_true, y_pred)  # 计算决定系数 R2
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算均方误差并开方得到 RMSE
        mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差 MAE
        
        metrics[model_cfg['name']] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}  # 保存当前模型的评估指标
        
        print(f"  R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")  # 打印当前模型的三项指标
        
    except Exception as e:  # 捕获模型加载或预测失败的异常
        print(f"  [失败] {e}")  # 打印失败原因
        print(f"  模型路径: {model_cfg['path']}")  # 打印配置的权重路径
        print(f"  解析路径: {resolve_model_path(model_cfg['path'])}")  # 打印实际解析出的权重路径
        import traceback  # 导入 traceback 模块以打印详细堆栈
        traceback.print_exc()  # 打印异常完整堆栈便于定位问题
        continue  # 跳过当前模型继续处理下一个

# ===================== 统一长度 =====================
if len(predictions) > 0:  # 只有成功加载了至少一个模型才进行绘图
    min_len = min(len(pred['y_pred']) for pred in predictions.values())  # 找出所有模型预测结果中的最短长度
    for name in predictions:  # 遍历每个已成功预测的模型
        predictions[name]['y_pred'] = predictions[name]['y_pred'][:min_len]  # 把预测结果截断到统一长度
        predictions[name]['y_true'] = predictions[name]['y_true'][:min_len]  # 把真实值截断到统一长度
    
    # 使用第一个模型的y_true
    y_true_unified = list(predictions.values())[0]['y_true']  # 使用第一个模型的真实值作为统一基准
    
    # ===================== 绘图 =====================
    print(f"\n{'='*60}")  # 打印 60 个等号的分隔线
    print("绘制对比图")  # 打印正在绘制对比图的提示
    print(f"{'='*60}")  # 再次打印分隔线
    
    timesteps = np.arange(len(y_true_unified))  # 生成与真实值等长的横轴步数序列
    
    fig, ax = plt.subplots(figsize=(14, 6))  # 创建 14x6 英寸的对比图画布
    
    # 先绘制 Actual - 使用橙色实线
    ax.plot(timesteps, y_true_unified, label='Actual', linewidth=2.0,   # 先用橙色实线绘制真实值曲线，置于最上层
            color='#D2691E', alpha=1.0, zorder=10)  # 设置真实值线宽、颜色、透明度与绘制层级
    
    # 按照配置顺序绘制模型预测 - 使用较细的实线
    for model_cfg in models_config:  # 按配置顺序绘制各模型的预测曲线
        if model_cfg['name'] in predictions:  # 只绘制成功加载的模型
            y_pred = predictions[model_cfg['name']]['y_pred']  # 取出当前模型的预测值
            ax.plot(timesteps, y_pred,   # 绘制单条预测曲线
                   label=model_cfg['name'],  # 图例标签使用模型名称
                   linewidth=1.5,  # 设置预测曲线线宽
                   color=model_cfg['color'],  # 使用配置中指定的颜色
                   linestyle=model_cfg['linestyle'],  # 使用配置中指定的线型
                   alpha=0.8)  # 设置预测曲线透明度

    ax.set_xlabel('Timestep', fontsize=16)  # 设置横轴标签
    ax.set_ylabel('TD (℃)', fontsize=16)  # 设置纵轴标签（温度偏差，摄氏度）
    ax.set_xlim(-15, 800)  # 左侧留白，避免0刻度贴边  # 补充：限制横轴显示范围，左侧留白避免 0 刻度贴边
    ax.tick_params(axis='both', which='major', labelsize=16)  # 设置坐标轴刻度数字字号
    ax.legend(fontsize=16, loc='upper right', frameon=False)  # 在图右上角显示图例且不加边框
    ax.grid(False)  # 不显示网格，保持简洁  # 补充：关闭网格线，保持图面简洁
    
    # 只保留左边和底部的边框
    ax.spines['top'].set_visible(False)  # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    ax.spines['left'].set_linewidth(1.2)  # 设置左边框线宽
    ax.spines['left'].set_color('black')  # 设置左边框颜色为黑色
    ax.spines['bottom'].set_linewidth(1.2)  # 设置底边框线宽
    ax.spines['bottom'].set_color('black')  # 设置底边框颜色为黑色
    
    plt.tight_layout()  # 自动调整布局避免标签被裁切
    plt.savefig('multi_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')  # 以 300 dpi 保存对比图并应用紧凑裁切
    print(f"\n对比图已保存: multi_model_comparison.png")  # 打印对比图保存提示
    
    # 打印指标
    print(f"\n{'='*60}")  # 打印 60 个等号的分隔线
    print("模型性能对比")  # 打印模型性能对比标题
    print(f"{'='*60}")  # 再次打印分隔线
    print(f"{'模型':<20} {'R2':<10} {'RMSE':<10} {'MAE':<10}")  # 打印表头：模型名、R2、RMSE、MAE
    print(f"{'-'*60}")  # 打印表头下方的分隔线
    for model_name, metric in metrics.items():  # 遍历指标字典逐行打印
        print(f"{model_name:<20} {metric['R2']:<10.4f} {metric['RMSE']:<10.4f} {metric['MAE']:<10.4f}")  # 按列对齐输出每个模型的三项指标
    print(f"{'='*60}")  # 打印最后一条分隔线

print("\n完成！")  # 打印全部流程完成的提示
