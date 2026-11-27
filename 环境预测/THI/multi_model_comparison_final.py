"""
多模型对比脚本 - 完全按照训练逻辑

本脚本在独立的测试集上横向对比多个已训练好的时间序列预测模型，包括：
LSTM、TCN、GRU、GRU–ECA、GRU–Wavelet（小波分解）、GRU–Wavelet–ECA
以及多层堆叠的 GRU–Wavelet–ECA。脚本严格复现训练阶段的数据预处理逻辑
（小波分解、归一化、滑动窗口序列构造），从 checkpoint 自动推断模型结构
（TCN 通道/卷积核、堆叠 GRU 的隐藏维度），加载权重后完成预测，
统一各模型输出长度后计算 R2 / RMSE / MAE，并绘制多模型预测对比图。
"""
import pandas as pd  # 导入 pandas 库并重命名为 pd，用于表格数据的读取与处理
import numpy as np  # 导入 numpy 库并重命名为 np，用于数组与矩阵运算
import matplotlib  # 导入 matplotlib 基础模块，用于指定绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 模块并重命名为 plt，用于绘制对比图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标的归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、MSE、MAE 评估指标
import torch  # 导入 PyTorch 主库，用于张量计算与 GPU 加速
import torch.nn as nn  # 导入神经网络子库并重命名为 nn，用于搭建各类网络层
import pywt  # 导入 PyWavelets 库，用于小波分解（小波模型特征构造）
import warnings  # 导入 warnings 模块，用于过滤运行时的警告信息
warnings.filterwarnings('ignore')  # 忽略所有警告，保持输出简洁

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据 GPU 是否可用选择计算设备
print(f"使用设备: {device}")  # 打印本次运行所使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体，以便正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，避免负号显示为方块

# ===================== 模型定义 =====================
class ECALayer(nn.Module):  # 定义 ECA（高效通道注意力）层，通过一维卷积实现通道权重学习
    """ECA 高效通道注意力层：用自适应池化 + 1D 卷积生成通道注意力权重。"""
    def __init__(self, channels, gamma=2, b=1):  # 构造函数，根据通道数自适应计算卷积核大小
        """
        初始化 ECA 注意力层。

        主要参数:
            channels (int): 输入特征的通道数。
            gamma (int): 用于计算卷积核大小的缩放系数，默认 2。
            b (int): 用于计算卷积核大小的偏置项，默认 1。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的构造函数
        t = int(abs((np.log2(channels) + b) / gamma))  # 根据通道数按公式计算卷积核尺寸基数 t
        k_size = t if t % 2 else t + 1  # 保证卷积核大小为奇数（偶数则加 1）
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化：把每个通道压缩为 1 个全局描述值
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道间交互，不使用偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 把卷积输出映射为 0~1 的注意力权重

    def forward(self, x):  # 定义前向传播：对输入特征按通道施加注意力权重
        """
        前向传播：计算通道注意力并对输入特征加权。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, channels) 的输入序列。

        返回:
            Tensor: 与输入同形状的加权后序列。
        """
        x_permuted = x.permute(0, 2, 1)  # 将维度换为 (batch, channels, time_step) 以适配 1D 卷积
        y = self.avg_pool(x_permuted)  # 全局平均池化得到 (batch, channels, 1) 的通道描述
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 通过 1D 卷积学习通道权重（转置以满足卷积输入格式）
        y = self.sigmoid(y)  # 归一化得到 0~1 的注意力权重
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将注意力权重广播并按通道逐元素加权
        return x_permuted.permute(0, 2, 1)  # 把维度换回 (batch, time_step, channels) 后返回

class GRUModel(nn.Module):  # 定义基础 GRU 预测模型：GRU + 全连接输出层
    """基础 GRU 模型：取序列最后时刻隐藏状态输出单值预测。"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 构造函数，定义 GRU 与输出层
        """
        初始化 GRUModel。

        主要参数:
            input_dim (int): 每个时间步的输入特征维度。
            time_step (int): 输入序列的时间步长度。
            gru_hidden (int): GRU 隐藏层维度，默认 128。
            gru_layers (int): GRU 堆叠层数，默认 1。
            dropout (float): 多层 GRU 之间的 dropout 概率，默认 0.1。
        """
        super(GRUModel, self).__init__()  # 调用父类构造函数
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅多层启用 dropout，batch 在第一维
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层：把最后时刻隐藏状态映射为 1 个预测值

    def forward(self, x):  # 定义前向传播
        """
        前向传播：GRU 编码后取最后时刻输出做预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        out, _ = self.gru(x)  # 前向运行 GRU，得到全部时刻输出（忽略最终隐藏状态）
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出经全连接层得到预测

class GRUECAModel(nn.Module):  # 定义 GRU + ECA 注意力模型：先用 ECA 加权通道再送入 GRU
    """GRU + ECA 通道注意力模型。"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 构造函数
        """
        初始化 GRUECAModel。

        主要参数:
            input_dim (int): 输入特征维度，同时作为 ECA 的通道数。
            time_step (int): 时间步长度。
            gru_hidden (int): GRU 隐藏层维度，默认 128。
            gru_layers (int): GRU 层数，默认 1。
            dropout (float): dropout 概率，默认 0.1。
        """
        super(GRUECAModel, self).__init__()  # 调用父类构造函数
        self.eca = ECALayer(channels=input_dim)  # 定义 ECA 注意力层，通道数等于输入特征数
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅多层启用 dropout
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接输出层

    def forward(self, x):  # 定义前向传播
        """
        前向传播：先经 ECA 通道加权，再由 GRU 编码并输出预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 对输入施加 ECA 通道注意力
        out, _ = self.gru(x)  # 将加权后的序列送入 GRU
        return self.fc(out[:, -1, :])  # 取最后时刻输出并预测

class GRUWaveletModel(nn.Module):  # 定义 GRU + 小波分解模型（不带 ECA）
    """GRU + 小波分解（不带ECA）"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 构造函数
        """
        初始化 GRUWaveletModel（网络结构与基础 GRU 相同，区别在输入为小波特征）。

        主要参数:
            input_dim (int): 小波特征（含近似/细节分量）的输入维度。
            time_step (int): 时间步长度。
            gru_hidden (int): GRU 隐藏层维度，默认 128。
            gru_layers (int): GRU 层数，默认 1。
            dropout (float): dropout 概率，默认 0.1。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类构造函数
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅多层启用 dropout
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接输出层

    def forward(self, x):  # 定义前向传播
        """
        前向传播：对小波特征序列用 GRU 编码并预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的小波特征序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        out, _ = self.gru(x)  # GRU 前向
        return self.fc(out[:, -1, :])  # 取最后时刻输出并预测

class GRUWaveletECAModel(nn.Module):  # 定义 GRU + 小波分解 + ECA 模型（单层 GRU 版本）
    """GRU + 小波分解 + ECA（单层GRU版本）"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 构造函数
        """
        初始化 GRUWaveletECAModel（单层 GRU + ECA）。

        主要参数:
            input_dim (int): 小波特征输入维度。
            time_step (int): 时间步长度。
            gru_hidden (int): GRU 隐藏层维度，默认 128。
            gru_layers (int): GRU 层数，默认 1。
            dropout (float): dropout 概率，默认 0.1。
        """
        super(GRUWaveletECAModel, self).__init__()  # 调用父类构造函数
        self.eca = ECALayer(channels=input_dim)  # 定义 ECA 注意力层
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 仅多层启用 dropout
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接输出层

    def forward(self, x):  # 定义前向传播
        """
        前向传播：ECA 加权 -> GRU 编码 -> 输出预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的小波特征序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 对小波特征施加 ECA 通道注意力
        out, _ = self.gru(x)  # 将加权序列送入 GRU
        return self.fc(out[:, -1, :])  # 取最后时刻输出并预测

class GRUWaveletECAStackedModel(nn.Module):  # 定义多层独立 GRU 堆叠的 GRU–Wavelet–ECA 模型
    """GRU + 小波分解 + ECA（多层独立GRU堆叠版本）"""
    def __init__(self, input_dim, hidden_sizes, dropout=0.2):  # 构造函数，hidden_sizes 指定每层隐藏维度
        """
        初始化堆叠 GRU 模型：ECA 之后按 hidden_sizes 依次堆叠多个单层 GRU。

        主要参数:
            input_dim (int): 输入特征维度。
            hidden_sizes (list[int]): 每一层 GRU 的隐藏维度列表。
            dropout (float): 每层 GRU 之后施加的 dropout 概率，默认 0.2。
        """
        super(GRUWaveletECAStackedModel, self).__init__()  # 调用父类构造函数
        self.eca = ECALayer(channels=input_dim)  # 定义 ECA 注意力层

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多个 GRU 层，保证参数被正确注册
        in_size = input_dim  # 记录当前层的输入维度，初始为特征维度
        for h_size in hidden_sizes:  # 遍历每一层的隐藏维度
            self.gru_layers_list.append(  # 向 ModuleList 追加一个单层 GRU
                nn.GRU(  # 实例化 GRU 层
                    input_size=in_size,  # 本层输入维度等于上一层输出维度
                    hidden_size=h_size,  # 本层隐藏维度
                    num_layers=1,  # 每层只含一个 GRU
                    dropout=0.0,  # 单层内部不使用 dropout（dropout 在层间单独施加）
                    batch_first=True,  # batch 位于第一维
                    bidirectional=False  # 使用单向 GRU
                )
            )  # 追加完成
            in_size = h_size  # 更新下一层的输入维度为本层隐藏维度

        self.dropout = nn.Dropout(dropout)  # 定义层间 dropout
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 全连接层：输入维度为最后一层 GRU 的隐藏维度

    def forward(self, x):  # 定义前向传播
        """
        前向传播：ECA 加权后依次经过所有堆叠 GRU 层，取最后时刻输出预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 对输入施加 ECA 通道注意力
        out = x  # 初始化循环变量为加权后的输入
        for gru in self.gru_layers_list:  # 依次经过每一个 GRU 层
            out, _ = gru(out)  # 当前 GRU 层前向
            out = self.dropout(out)  # 层后施加 dropout
        return self.fc(out[:, -1, :])  # 取最后时刻输出并预测

class LSTMModel(nn.Module):  # 定义 LSTM 预测模型：LSTM + 全连接输出层
    """基础 LSTM 模型：取序列最后时刻隐藏状态输出单值预测。"""
    def __init__(self, input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.1):  # 构造函数
        """
        初始化 LSTMModel。

        主要参数:
            input_dim (int): 输入特征维度。
            time_step (int): 时间步长度。
            lstm_hidden (int): LSTM 隐藏层维度，默认 128。
            lstm_layers (int): LSTM 层数，默认 1。
            dropout (float): 多层之间的 dropout 概率，默认 0.1。
        """
        super(LSTMModel, self).__init__()  # 调用父类构造函数
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,  # 定义 LSTM 层
                            dropout=dropout if lstm_layers > 1 else 0.0, batch_first=True)  # 仅多层启用 dropout
        self.fc = nn.Linear(lstm_hidden, 1)  # 全连接输出层

    def forward(self, x):  # 定义前向传播
        """
        前向传播：LSTM 编码后取最后时刻输出做预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        out, _ = self.lstm(x)  # LSTM 前向，得到全部时刻输出
        return self.fc(out[:, -1, :])  # 取最后时刻输出并预测

class Chomp1d(nn.Module):  # 定义裁剪模块，用于实现因果卷积（去掉卷积尾部多出的时间步）
    """裁剪模块：截去因果卷积在序列尾部产生的 padding，保证输出不泄露未来信息。"""
    def __init__(self, chomp_size):  # 构造函数
        """
        初始化裁剪模块。

        主要参数:
            chomp_size (int): 需要从尾部裁剪掉的时间步数（通常等于 padding 大小）。
        """
        super(Chomp1d, self).__init__()  # 调用父类构造函数
        self.chomp_size = chomp_size  # 保存裁剪长度

    def forward(self, x):  # 定义前向传播
        """
        前向传播：裁掉输入序列尾部的 chomp_size 个时间步并保证内存连续。

        主要参数:
            x (Tensor): 形状为 (batch, channels, time_steps) 的卷积输出。

        返回:
            Tensor: 裁剪后的连续张量。
        """
        return x[:, :, :-self.chomp_size].contiguous()  # 去掉最后 chomp_size 个时间步并转为连续内存布局

class TemporalBlock(nn.Module):  # 定义 TCN 的基本时间块：两层膨胀因果卷积 + 残差连接
    """TCN 的基本时间块：膨胀因果卷积 + 裁剪 + ReLU + Dropout，并带残差连接。"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):  # 构造函数
        """
        初始化 TCN 时间块。

        主要参数:
            n_inputs (int): 输入通道数。
            n_outputs (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步长。
            dilation (int): 膨胀系数，控制感受野。
            padding (int): 卷积 padding 大小。
            dropout (float): dropout 概率，默认 0.2。
        """
        super(TemporalBlock, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)  # 第一层膨胀因果卷积
        self.chomp1 = Chomp1d(padding)  # 裁掉第一层卷积尾部多余的 padding
        self.relu1 = nn.ReLU()  # 第一层激活函数
        self.dropout1 = nn.Dropout(dropout)  # 第一层 dropout
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)  # 第二层膨胀因果卷积
        self.chomp2 = Chomp1d(padding)  # 裁掉第二层卷积尾部多余的 padding
        self.relu2 = nn.ReLU()  # 第二层激活函数
        self.dropout2 = nn.Dropout(dropout)  # 第二层 dropout
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,  # 把主路径各层按顺序组合成序列
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 主路径结束
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 通道数不一致时用 1×1 卷积匹配残差维度
        self.relu = nn.ReLU()  # 残差相加后的最终激活函数

    def forward(self, x):  # 定义前向传播
        """
        前向传播：主路径输出与残差（下采样）相加后再激活。

        主要参数:
            x (Tensor): 形状为 (batch, n_inputs, time_steps) 的输入。

        返回:
            Tensor: 形状为 (batch, n_outputs, time_steps) 的输出。
        """
        out = self.net(x)  # 计算主路径输出
        res = x if self.downsample is None else self.downsample(x)  # 通道数一致则直接取输入作为残差，否则下采样
        return self.relu(out + res)  # 残差相加并激活后返回

class TCNModel(nn.Module):  # 定义 TCN 预测模型：由多个 TemporalBlock 堆叠后接全连接层
    """TCN（时间卷积网络）模型：膨胀因果卷积堆叠，取最后时刻特征输出单值预测。"""
    def __init__(self, input_dim, time_step, num_channels=[32, 128], kernel_size=3, dropout=0.2):  # 构造函数
        """
        初始化 TCNModel，num_channels 与 kernel_size 通常由 checkpoint 自动推断。

        主要参数:
            input_dim (int): 输入特征维度。
            time_step (int): 时间步长度。
            num_channels (list[int]): 每个时间块的输出通道数，默认 [32, 128]。
            kernel_size (int): 卷积核大小，默认 3。
            dropout (float): dropout 概率，默认 0.2。
        """
        super(TCNModel, self).__init__()  # 调用父类构造函数
        layers = []  # 初始化列表，收集各个时间块
        for i in range(len(num_channels)):  # 按通道配置逐层构建时间块
            dilation_size = 2 ** i  # 膨胀系数按 2 的幂递增，扩大感受野
            in_channels = input_dim if i == 0 else num_channels[i-1]  # 首层输入为特征维度，后续为上一层通道数
            out_channels = num_channels[i]  # 本层输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,  # 构造时间块并加入列表
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]  # padding 保证因果卷积输入输出等长
        self.network = nn.Sequential(*layers)  # 把所有时间块组合成顺序网络
        self.fc = nn.Linear(num_channels[-1], 1)  # 全连接层：输入为最后一层通道数

    def forward(self, x):  # 定义前向传播
        """
        前向传播：把输入换维后送入 TCN，取最后时刻特征做预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        x = x.transpose(1, 2)  # 换维为 (batch, input_dim, time_step) 以适配 Conv1d
        out = self.network(x)  # 经过 TCN 网络得到 (batch, num_channels[-1], time_step)
        return self.fc(out[:, :, -1])  # 取最后一个时间步的特征经全连接层输出预测

def infer_tcn_config_from_state_dict(state_dict):  # 从 checkpoint 的权重张量形状推断 TCN 的通道数与卷积核大小
    """从checkpoint权重推断TCN结构参数"""
    block_indices = set()  # 用集合记录出现的时间块编号（去重）
    for k in state_dict.keys():  # 遍历 checkpoint 中的所有权重键
        if k.startswith('network.') and '.conv1.weight' in k:  # 仅关注各时间块第一层卷积的权重键
            try:
                idx = int(k.split('.')[1])  # 从键名中解析出时间块序号
                block_indices.add(idx)  # 将该序号加入集合
            except:  # 解析失败则忽略该键
                pass  # 静默跳过

    if not block_indices:  # 若没有识别到任何时间块
        return [128], 3  # 返回默认配置：单块 128 通道、卷积核 3

    sorted_indices = sorted(block_indices)  # 将块序号排序，保证通道顺序与训练时一致
    num_channels = []  # 初始化通道数列表
    kernel_size = None  # 初始化卷积核大小为 None

    for idx in sorted_indices:  # 按序遍历每个时间块
        w = state_dict[f'network.{idx}.conv1.weight']  # 取出对应块第一层卷积的权重张量
        num_channels.append(int(w.shape[0]))  # 权重的第 0 维即输出通道数，记录到列表
        if kernel_size is None:  # 只在第一个块上读取卷积核大小
            kernel_size = int(w.shape[-1])  # 权重的最后一维即卷积核宽度

    return num_channels, (kernel_size if kernel_size is not None else 3)  # 返回推断出的通道列表与卷积核大小（缺省为 3）

def infer_stacked_gru_hidden_sizes(state_dict):  # 从 checkpoint 权重推断堆叠 GRU 每层的隐藏维度
    """从checkpoint权重推断gru_layers_list每层hidden size"""
    layer_map = {}  # 用字典记录 {层序号: 隐藏维度}
    for k, v in state_dict.items():  # 遍历 checkpoint 中的所有权重键值对
        if k.startswith('gru_layers_list.') and k.endswith('weight_ih_l0'):  # 仅关注各 GRU 的输入-隐藏权重键
            try:
                idx = int(k.split('.')[1])  # 从键名中解析出 GRU 层序号
                layer_map[idx] = int(v.shape[0] // 3)  # GRU 门控有 3 倍隐藏维度，故隐藏维度 = 张量第 0 维 // 3
            except:  # 解析失败则忽略该键
                pass  # 静默跳过

    if not layer_map:  # 若没有识别到任何 GRU 层
        return [128]  # 返回默认隐藏维度配置

    return [layer_map[i] for i in sorted(layer_map.keys())]  # 按层序号排序后返回各层隐藏维度列表

# ===================== 小波分解函数 =====================
def wavelet_decompose_sliding(data, target_col='THI', wavelet='db4', level=3, window_size=None):  # 对目标列做滑动窗口小波分解生成新特征
    """
    滑动窗口小波分解 - 不保留原始TD列

    对每个时间步取一个历史窗口，对目标列做 level 层小波分解，提取最后一个
    近似系数与各层细节系数作为新特征；其余列原样保留。

    主要参数:
        data (DataFrame): 原始特征表。
        target_col (str): 需要小波分解的目标列名，默认 'THI'。
        wavelet (str): 小波基名称，默认 'db4'。
        level (int): 分解层数，默认 3。
        window_size (int, 可选): 滑动窗口长度，默认按 2^(level+6) 与数据长度取较小者。

    返回:
        DataFrame: 含小波近似/细节特征及其余原始列的DataFrame。
    """
    if window_size is None:  # 若未指定窗口长度
        window_size = min(2 ** (level + 6), len(data))  # 取 2^(level+6) 与数据长度中的较小值作为窗口长度
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建空 DataFrame 存放小波特征，索引与原数据一致
    
    for col in data.columns:  # 遍历原数据的每一列
        if col == 'timestamp':  # 跳过时间戳列（不参与分解）
            continue  # 处理下一列
        
        signal = data[col].values  # 取出当前列的数值序列
        
        if col == target_col:  # 若当前列是需要分解的目标列
            approx_list = []  # 初始化列表，保存每个时间步的近似系数
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数初始化一个空列表
            
            for i in range(len(signal)):  # 逐时间步构造窗口并分解
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证不越界
                window_signal = signal[start_idx:i+1]  # 截取从起点到当前时刻的历史窗口
                
                if len(window_signal) < window_size:  # 若窗口长度不足（序列开头）
                    pad_width = window_size - len(window_signal)  # 计算需要补点的数量
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 在窗口前端用边缘值填充至规定长度
                
                try:
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层小波分解
                    approx_list.append(coeffs[0][-1])  # 取最后 1 个近似系数作为近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层最后一个细节系数，空则用 0 代替
                except:  # 若小波分解出现异常
                    approx_list.append(window_signal[-1])  # 退化为使用窗口最后一个原始值作为近似特征
                    for j in range(level):  # 遍历每一层
                        detail_lists[j].append(0.0)  # 细节系数用 0 填充
            
            # 不保留原始TD列，只添加小波特征
            wavelet_features[f'{col}_approx'] = approx_list  # 把近似系数序列加入结果表
            for j in range(level):  # 遍历每一层细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 把第 j+1 层细节系数加入结果表
        else:  # 若当前列不是目标列
            wavelet_features[col] = signal  # 原样保留该列
    
    return wavelet_features  # 返回构造好的小波特征表

# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始读取测试数据
data = pd.read_csv('部分温湿度数据.csv', encoding='gbk')  # 以 gbk 编码读取测试数据 CSV
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m', origin=pd.Timestamp('2024/4/27 16:09'))  # 以索引×5分钟生成时间戳

features = ['日龄', 'THI', '当前温度', '目标温度']  # 定义候选特征列名列表
target = 'THI'  # 定义预测目标列为 THI
features = [f for f in features if f in data.columns]  # 过滤掉数据中不存在的候选特征

# 缺失值处理
for col in ['THI', '当前温度', '目标温度']:  # 遍历连续型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当列存在且有缺失值时处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')  # 线性插值后再前后向填充

for col in ['日龄']:  # 遍历离散型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当列存在且有缺失值时处理
        data[col] = data[col].fillna(method='ffill').fillna(method='bfill')  # 用前后向填充处理离散变量缺失

data = data.dropna(subset=features + [target]).reset_index(drop=True)  # 删除仍含缺失值的行并重置索引
print(f"数据形状: {data.shape}")  # 打印清洗后数据的行列数

# ===================== 模型配置 =====================
models_config = [  # 定义待对比模型的配置列表（名称、权重路径、绘图颜色与线型）
    {'name': 'LSTM', 'path': 'pth新/Model_LSTM_pytorch.pth', 'color': '#FFA500', 'linestyle': '--'},  # LSTM 模型配置
    {'name': 'TCN', 'path': 'pth新/Model_TCN_pytorch.pth', 'color': '#00CED1', 'linestyle': '--'},  # TCN 模型配置
    {'name': 'GRU', 'path': 'pth新/Model_GRU_pytorch.pth', 'color': '#4169E1', 'linestyle': '--'},  # GRU 模型配置
    {'name': 'GRU–ECA', 'path': 'pth新/Model_GRUECA_pytorch.pth', 'color': '#32CD32', 'linestyle': '--'},  # GRU–ECA 模型配置
    {'name': 'GRU–Wavelet', 'path': 'pth新/Model_GRUWavelet_pytorch.pth', 'color': '#8B008B', 'linestyle': '--'},  # GRU–Wavelet 模型配置
    {'name': 'GRU–Wavelet–ECA', 'path': 'pth新/Model_GRUWavelet_no_leakage_pytorch.pth', 'color': '#DC143C', 'linestyle': '--'},  # GRU–Wavelet–ECA 模型配置
]
# ===================== 加载模型并预测 =====================
predictions = {}  # 初始化字典，保存每个模型的预测结果与真实值
metrics = {}  # 初始化字典，保存每个模型的评估指标

for model_cfg in models_config:  # 遍历模型配置，逐个加载并预测
    print(f"\n{'='*60}")  # 打印分隔线
    print(f"加载模型: {model_cfg['name']}")  # 打印当前加载的模型名称
    print(f"{'='*60}")  # 打印分隔线
    
    try:
        checkpoint = torch.load(model_cfg['path'], map_location=device, weights_only=False)  # 加载当前模型的 checkpoint 并映射到设备
        
        time_step = checkpoint['time_step']  # 从 checkpoint 读取时间窗口长度
        pred_offset = checkpoint.get('pred_offset', 6)  # 默认6，但优先使用保存的值
        feature_scaler = checkpoint.get('feature_scaler')  # 读取保存的特征归一化器（可能不存在）
        target_scaler = checkpoint.get('target_scaler')  # 读取保存的目标归一化器（可能不存在）
        
        print(f"  time_step: {time_step}, pred_offset: {pred_offset}")  # 打印窗口与偏移配置
        
        # 判断是否为小波模型
        is_wavelet = 'wavelet' in checkpoint or 'level' in checkpoint  # 依据 checkpoint 是否含 wavelet/level 键判断是否为小波模型
        has_eca = any('eca' in key for key in checkpoint['model_state_dict'].keys())  # 依据权重键名是否含 eca 判断是否带注意力
        
        print(f"  is_wavelet: {is_wavelet}, has_eca: {has_eca}")  # 打印模型类型判断结果
        
        # 准备特征
        if is_wavelet:  # 若为小波模型
            print(f"  小波模型")  # 打印提示
            wavelet_features = wavelet_decompose_sliding(  # 对测试数据做滑动窗口小波分解生成特征
                data[features],  # 输入原始特征列
                target_col='THI',  # 指定对 THI 列做分解
                wavelet='db4',  # 使用 db4 小波基
                level=3,  # 分解 3 层
                window_size=time_step  # 窗口长度使用模型的时间步长
            )
            
            saved_feature_names = checkpoint.get('feature_names', list(wavelet_features.columns))  # 读取保存的特征名，若无则用当前全部列名
            test_features_df = wavelet_features[saved_feature_names]  # 按训练时的特征顺序选取列
            print(f"  特征: {saved_feature_names}")  # 打印所用特征名
        else:  # 若为非小波模型
            print(f"  非小波模型")  # 打印提示
            test_features_df = data[features].copy()  # 直接复制原始特征列
        
        # 归一化特征
        if feature_scaler is None:  # 若 checkpoint 未保存特征归一化器
            print(f"  警告：feature_scaler 为 None，需要重新创建")  # 打印警告
            train_data = pd.read_csv('温湿度数据.csv', encoding='gbk')  # 重新读取训练数据以拟合归一化器
            feature_scaler = MinMaxScaler()  # 新建特征归一化器
            target_scaler = MinMaxScaler()  # 新建目标归一化器
            
            # 如果是小波模型，需要先对训练数据做小波分解
            if is_wavelet:  # 若为小波模型
                train_wavelet = wavelet_decompose_sliding(train_data[features], target_col='THI', wavelet='db4', level=3, window_size=time_step)  # 对训练数据做相同的小波分解
                feature_scaler.fit(train_wavelet[saved_feature_names])  # 用小波特征拟合特征归一化器
            else:  # 若为非小波模型
                feature_scaler.fit(train_data[features])  # 用原始特征拟合特征归一化器
            target_scaler.fit(train_data[[target]])  # 用训练数据目标拟合目标归一化器
        else:  # 若 checkpoint 已保存特征归一化器
            print(f"  使用模型保存的 feature_scaler")  # 打印提示
        
        test_features_normalized = pd.DataFrame(  # 构造归一化后的测试特征 DataFrame
            feature_scaler.transform(test_features_df),  # 用特征归一化器变换测试特征
            columns=test_features_df.columns  # 保持列名一致
        )
        
        # 创建序列
        def create_sequences(data, time_step, pred_offset):  # 定义函数：把特征表切分为滑动窗口序列
            """
            将二维特征表切分为滑动窗口三维序列样本。

            主要参数:
                data (DataFrame): 归一化后的特征表。
                time_step (int): 窗口长度。
                pred_offset (int): 预测偏移步数。

            返回:
                np.ndarray: 形状为 (样本数, time_step, 特征数) 的数组。
            """
            sequences = []  # 初始化样本列表
            for i in range(len(data) - time_step - pred_offset):  # 窗口起点遍历，预留窗口与偏移长度
                sequences.append(data.iloc[i:i + time_step].values)  # 截取一个窗口并转为数组
            return np.array(sequences)  # 堆叠为三维数组返回
        
        X_test = create_sequences(test_features_normalized, time_step, pred_offset)  # 为归一化测试特征构造序列
        
        # 关键：y_true 也要归一化后再取对应位置！
        y_true_normalized = target_scaler.transform(data[[target]]).flatten()  # 先对真实目标做归一化并展平
        y_true_normalized_seq = y_true_normalized[time_step + pred_offset - 1:time_step + pred_offset - 1 + len(X_test)]  # 取与每个样本预测目标对应的归一化标签
        
        print(f"  X_test: {X_test.shape}, y_true: {y_true_normalized_seq.shape}")  # 打印测试序列与标签形状
        
        # 构建模型
        input_dim = X_test.shape[2]  # 输入维度等于序列第三维（特征数）
        model_state_dict = checkpoint['model_state_dict']  # 取出 checkpoint 中的权重状态字典

        if is_wavelet and has_eca:  # 若为小波 + ECA 模型
            # 兼容两种结构：单层GRU 或 gru_layers_list堆叠GRU
            if any(k.startswith('gru_layers_list.') for k in model_state_dict.keys()):  # 若权重键含 gru_layers_list 前缀
                hidden_sizes = infer_stacked_gru_hidden_sizes(model_state_dict)  # 推断堆叠 GRU 每层隐藏维度
                print(f"  检测到堆叠GRU结构, hidden_sizes={hidden_sizes}")  # 打印推断结果
                model = GRUWaveletECAStackedModel(input_dim, hidden_sizes=hidden_sizes, dropout=0.2).to(device)  # 构建堆叠 GRU 模型
            else:  # 否则为单层 GRU
                model = GRUWaveletECAModel(input_dim, time_step).to(device)  # 构建单层 GRU + ECA 模型
        elif is_wavelet and not has_eca:  # 若为小波但不带 ECA
            model = GRUWaveletModel(input_dim, time_step).to(device)  # 构建 GRU–Wavelet 模型
        elif has_eca:  # 若带 ECA 但非小波
            model = GRUECAModel(input_dim, time_step).to(device)  # 构建 GRU–ECA 模型
        elif 'lstm' in model_cfg['path'].lower():  # 若路径名含 lstm
            model = LSTMModel(input_dim, time_step).to(device)  # 构建 LSTM 模型
        elif 'tcn' in model_cfg['path'].lower():  # 若路径名含 tcn
            # 从权重自动推断TCN层通道和卷积核
            num_channels, kernel_size = infer_tcn_config_from_state_dict(model_state_dict)  # 从权重推断 TCN 结构
            print(f"  检测到TCN结构, num_channels={num_channels}, kernel_size={kernel_size}")  # 打印推断结果
            model = TCNModel(input_dim, time_step, num_channels=num_channels, kernel_size=kernel_size, dropout=0.2).to(device)  # 构建 TCN 模型
        else:  # 其余情况按基础 GRU 处理
            model = GRUModel(input_dim, time_step).to(device)  # 构建基础 GRU 模型
        
        # 调试：检查模型权重形状
        if model_cfg['name'] == 'GRU+Wavelet':  # 仅对该名称的模型打印调试信息（注意配置中名称含 en dash，通常不触发）
            print(f"  模型权重形状:")  # 打印标题
            for name, param in checkpoint['model_state_dict'].items():  # 遍历 checkpoint 中的权重
                print(f"    {name}: {param.shape}")  # 打印权重名与形状
            print(f"  模型结构:")  # 打印标题
            for name, param in model.named_parameters():  # 遍历模型参数
                print(f"    {name}: {param.shape}")  # 打印参数名与形状
        
        # 加载模型权重，忽略不匹配的键
        model_state_dict = checkpoint['model_state_dict']  # 再次取出权重字典
        model_dict = model.state_dict()  # 取出当前模型的状态字典
        
        # 过滤掉不匹配的键
        filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}  # 仅保留键名匹配且形状一致的权重
        
        # 检查是否有未加载的键
        missing_keys = set(model_dict.keys()) - set(filtered_state_dict.keys())  # 计算模型中未被加载的键
        unexpected_keys = set(model_state_dict.keys()) - set(model_dict.keys())  # 计算 checkpoint 中多余的键
        
        if missing_keys:  # 若存在未加载的键
            print(f"  警告: 以下键未从checkpoint加载: {missing_keys}")  # 打印警告
        if unexpected_keys:  # 若存在多余键
            print(f"  警告: 以下键在checkpoint中但模型中不存在(已忽略): {unexpected_keys}")  # 打印警告
        
        model_dict.update(filtered_state_dict)  # 用过滤后的权重更新模型状态字典
        model.load_state_dict(model_dict)  # 将合并后的权重加载到模型
        model.eval()  # 将模型切换为评估模式
        
        # 预测（归一化空间）
        X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试序列转为张量并移至设备
        with torch.no_grad():  # 关闭梯度计算
            y_pred_normalized = model(X_test_tensor).cpu().numpy().flatten()  # 预测并转为展平的一维 NumPy 数组
        
        # 反归一化到原始空间
        y_pred = target_scaler.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()  # 将预测结果反归一化并展平
        y_true = target_scaler.inverse_transform(y_true_normalized_seq.reshape(-1, 1)).flatten()  # 将真实标签反归一化并展平
        
        # 调试输出
        if model_cfg['name'] in ['GRU+Wavelet', 'GRU+Wavelet+ECA']:  # 对这两个名称打印调试信息（同样通常不触发）
            print(f"  调试信息:")  # 打印标题
            print(f"    y_pred 前5个值: {y_pred[:5]}")  # 打印预测前 5 个值
            print(f"    y_true 前5个值: {y_true[:5]}")  # 打印真实前 5 个值
            print(f"    y_pred 范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")  # 打印预测值范围
            print(f"    y_true 范围: [{y_true.min():.4f}, {y_true.max():.4f}]")  # 打印真实值范围
        
        # 保存结果
        predictions[model_cfg['name']] = {'y_pred': y_pred, 'y_true': y_true}  # 将该模型的预测与真实值存入结果字典
        
        # 计算指标
        r2 = r2_score(y_true, y_pred)  # 计算 R2
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算 RMSE
        mae = mean_absolute_error(y_true, y_pred)  # 计算 MAE
        
        metrics[model_cfg['name']] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}  # 保存该模型的三项指标
        
        print(f"  R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")  # 打印该模型的评估指标
        
    except Exception as e:  # 捕获单个模型加载/预测过程中的所有异常
        print(f"  [失败] {e}")  # 打印失败原因
        print(f"  模型路径: {model_cfg['path']}")  # 打印失败模型的路径
        import traceback  # 导入 traceback 模块
        traceback.print_exc()  # 打印完整异常堆栈便于排查
        continue  # 继续处理下一个模型

# ===================== 统一长度 =====================
if len(predictions) > 0:  # 仅当至少成功加载了一个模型时
    min_len = min(len(pred['y_pred']) for pred in predictions.values())  # 求所有模型预测结果中的最小长度
    for name in predictions:  # 遍历每个模型
        predictions[name]['y_pred'] = predictions[name]['y_pred'][:min_len]  # 截断预测结果到统一长度
        predictions[name]['y_true'] = predictions[name]['y_true'][:min_len]  # 截断真实值到统一长度
    
    # 使用第一个模型的y_true
    y_true_unified = list(predictions.values())[0]['y_true']  # 取第一个模型的真实值作为统一横轴基准
    
    # ===================== 绘图 =====================
    print(f"\n{'='*60}")  # 打印分隔线
    print("绘制对比图")  # 打印绘图提示
    print(f"{'='*60}")  # 打印分隔线
    
    timesteps = np.arange(len(y_true_unified))  # 生成横轴步数索引
    
    fig, ax = plt.subplots(figsize=(14, 6))  # 创建 14×6 英寸的画布与坐标轴
    
    # 先绘制 Actual - 使用橙色实线
    ax.plot(timesteps, y_true_unified, label='Actual', linewidth=2.0,  # 绘制真实值曲线（橙色实线，置于顶层）
            color='#D2691E', alpha=1.0, zorder=10)  # 设置颜色、不透明度与图层顺序
    
    # 按照配置顺序绘制模型预测 - 使用较细的实线
    for model_cfg in models_config:  # 按配置顺序遍历模型
        if model_cfg['name'] in predictions:  # 仅绘制成功加载的模型
            y_pred = predictions[model_cfg['name']]['y_pred']  # 取出该模型的预测结果
            ax.plot(timesteps, y_pred,  # 绘制该模型的预测曲线
                   label=model_cfg['name'],  # 图例标签为模型名
                   linewidth=1.5,  # 设置线宽
                   color=model_cfg['color'],  # 使用配置中指定的颜色
                   linestyle=model_cfg['linestyle'],  # 使用配置中指定的线型
                   alpha=0.8)  # 设置半透明度以便叠加观察
    
    ax.set_xlabel('Timestep', fontsize=16)  # 设置横轴标签
    ax.set_ylabel('THI', fontsize=16)  # 设置纵轴标签
    ax.set_ylabel('THI', fontsize=16)  # （重复设置）再次设置纵轴标签，与原脚本一致
    ax.set_xlim(0, 600)  # 只显示前300个数据点
    ax.tick_params(axis='both', which='major', labelsize=16)  # 设置主刻度字体大小
    ax.legend(fontsize=16, loc='upper right', frameon=False)  # 去掉图例边框
    ax.grid(False)  # 不显示网格，保持简洁
    
    # 只保留左边和底部的边框
    ax.spines['top'].set_visible(False)  # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右侧边框
    ax.spines['left'].set_linewidth(1.2)  # 设置左边框线宽
    ax.spines['left'].set_color('black')  # 设置左边框颜色为黑色
    ax.spines['bottom'].set_linewidth(1.2)  # 设置底边框线宽
    ax.spines['bottom'].set_color('black')  # 设置底边框颜色为黑色
    
    plt.tight_layout()  # 自动调整布局
    plt.savefig('multi_model_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')  # 以 600 dpi 保存对比图
    print(f"\n对比图已保存: multi_model_comparison.png")  # 打印保存提示
    
    # 打印指标
    print(f"\n{'='*60}")  # 打印分隔线
    print("模型性能对比")  # 打印指标对比标题
    print(f"{'='*60}")  # 打印分隔线
    print(f"{'模型':<20} {'R2':<10} {'RMSE':<10} {'MAE':<10}")  # 打印表头
    print(f"{'-'*60}")  # 打印表头分隔线
    for model_name, metric in metrics.items():  # 遍历各模型的指标
        print(f"{model_name:<20} {metric['R2']:<10.4f} {metric['RMSE']:<10.4f} {metric['MAE']:<10.4f}")  # 按列对齐打印各指标
    print(f"{'='*60}")  # 打印分隔线

print("\n完成！")  # 打印全部完成提示
