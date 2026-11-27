"""
多模型对比脚本 - 完全按照训练逻辑

本脚本在同一测试集上横向对比 LSTM、TCN、GRU、GRU–ECA、GRU–Wavelet、
GRU–Wavelet–ECA 六种模型的温度偏差预测效果。脚本按各模型保存的检查点
自动重建对应网络结构、加载权重、完成推理并计算 R²/RMSE/MAE，
最终把多条预测曲线与真实曲线绘制在一张对比图中，并打印性能表格。
"""
import pandas as pd  # 导入 pandas 数据分析库并起别名 pd，用于 CSV 读取与表格处理
import numpy as np  # 导入 numpy 数值计算库并起别名 np，用于数组构造与指标计算
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端直接保存图片  # Agg 后端无需图形界面，适合离线出图
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口并起别名 plt，用于绘制多模型对比图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，作为 scaler 缺失时重新拟合的兜底方案
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R²、MSE、MAE 评估指标
import torch  # 导入 PyTorch 主库，用于张量计算与模型加载
import torch.nn as nn  # 导入 PyTorch 神经网络模块并起别名 nn，用于搭建各类网络
import pywt  # 导入 PyWavelets 小波分析库，用于小波分解特征构造
import warnings  # 导入 warnings 模块，用于屏蔽无关警告
warnings.filterwarnings('ignore')  # 忽略运行时警告，保持控制台输出整洁

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有 GPU 则用 cuda，否则回退到 cpu
print(f"使用设备: {device}")  # 打印推理使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，避免图中中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 关闭负号的 Unicode 显示

# ===================== 模型定义 =====================
class ECALayer(nn.Module):
    """ECA（Efficient Channel Attention）高效通道注意力层。

    通过全局平均池化与一维卷积学习各通道权重，对输入特征做通道级重标定，
    核大小通道数自适应确定，参数量极小。
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 注意力层。

        参数:
            channels: 输入特征通道数。
            gamma: 核大小自适应公式中的缩放系数。
            b: 核大小自适应公式中的偏置项。
        """
        super(ECALayer, self).__init__()  # 调用父类初始化
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应计算卷积核基数 t
        k_size = t if t % 2 else t + 1  # 保证卷积核为奇数，便于对称 padding
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应全局平均池化，把序列压缩为每通道一个值
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道间交互，无偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 把卷积输出压缩为 0~1 的通道权重

    def forward(self, x):
        """对输入施加通道注意力并重标定。

        参数:
            x: (batch, seq, channels) 的输入序列。
        返回:
            (batch, seq, channels) 加权后的序列。
        """
        x_permuted = x.permute(0, 2, 1)  # 转置为 (batch, channels, seq) 以适配 Conv1d
        y = self.avg_pool(x_permuted)  # 全局平均池化得到 (batch, channels, 1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 转置后做通道卷积再转回，得到通道权重
        y = self.sigmoid(y)  # 归一化权重到 0~1
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将通道权重广播乘回每个时间步
        return x_permuted.permute(0, 2, 1)  # 转回 (batch, seq, channels) 形状

class GRUModel(nn.Module):
    """基础 GRU 回归模型：GRU 编码序列，取末时刻输出做单值预测。"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):
        """初始化 GRU 模型。

        参数:
            input_dim: 每步输入特征数；time_step: 时间步数（仅记录）。
            gru_hidden: 隐藏单元数；gru_layers: 层数；dropout: 多层时层间丢弃率。
        """
        super(GRUModel, self).__init__()  # 调用父类初始化
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 定义 GRU 层，batch 在首维
        self.fc = nn.Linear(gru_hidden, 1)  # 末时刻隐藏状态映射到预测值
    def forward(self, x):
        """前向传播：序列 -> 预测值。"""
        out, _ = self.gru(x)  # 运行 GRU，取每步输出
        return self.fc(out[:, -1, :])  # 用最后一个时间步输出做预测
class GRUECAModel(nn.Module):
    """GRU + ECA 注意力模型：先做通道注意力再送入 GRU 预测。"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):
        """初始化 GRU–ECA 模型，参数含义同 GRUModel。"""
        super(GRUECAModel, self).__init__()  # 调用父类初始化
        self.eca = ECALayer(channels=input_dim)  # 在 GRU 之前加入通道注意力
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 定义 GRU 层
        self.fc = nn.Linear(gru_hidden, 1)  # 输出层

    def forward(self, x):
        """前向传播：先通道加权再 GRU 回归。"""
        x = self.eca(x)  # 对输入施加 ECA 通道注意力
        out, _ = self.gru(x)  # 运行 GRU
        return self.fc(out[:, -1, :])  # 末时刻输出做预测

class GRUWaveletModel(nn.Module):
    """GRU + 小波分解（不带ECA）"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):
        """初始化 GRU–小波模型，参数含义同 GRUModel。"""
        super(GRUWaveletModel, self).__init__()  # 调用父类初始化
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,
                          dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 定义 GRU 层
        self.fc = nn.Linear(gru_hidden, 1)  # 输出层

    def forward(self, x):
        """前向传播：输入已含小波特征，直接 GRU 回归。"""
        out, _ = self.gru(x)  # 运行 GRU
        return self.fc(out[:, -1, :])  # 末时刻输出做预测

class GRUWaveletECAModel(nn.Module):
    """GRU + 小波分解 + ECA，使用 ModuleList 与训练时结构一致"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):
        """初始化多层 GRU–小波–ECA 模型。

        参数:
            input_dim: 输入特征数；time_step: 时间步数（仅记录）。
            gru_hidden: 隐藏单元数，可为整数或列表；gru_layers: 层数。
            dropout: 每层 GRU 之后的丢弃率。
        """
        super(GRUWaveletECAModel, self).__init__()  # 调用父类初始化
        self.eca = ECALayer(channels=input_dim)  # 前置通道注意力

        if isinstance(gru_hidden, list):  # 若隐藏维度以列表给出则直接采用
            hidden_sizes = gru_hidden  # 使用逐层指定的隐藏维度
        else:  # 否则按层数复制同一隐藏维度
            hidden_sizes = [gru_hidden] * gru_layers  # 各层使用相同隐藏维度

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 容纳逐层堆叠的 GRU
        in_size = input_dim  # 记录当前层输入维度
        for h_size in hidden_sizes:  # 逐层构建 GRU
            self.gru_layers_list.append(  # 向 ModuleList 追加一层 GRU
                nn.GRU(input_size=in_size, hidden_size=h_size, num_layers=1,
                       dropout=0.0, batch_first=True, bidirectional=False)  # 单层单向 GRU，层间 dropout 单独施加
            )
            in_size = h_size  # 下一层的输入维度等于本层隐藏维度

        self.dropout = nn.Dropout(dropout)  # 每层 GRU 输出后施加的 dropout
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 最后一层隐藏维度映射到预测值

    def forward(self, x):
        """前向传播：ECA -> 多层 GRU -> 全连接预测。"""
        x = self.eca(x)  # 施加通道注意力
        out = x  # 初始化序列变量
        for gru in self.gru_layers_list:  # 依次通过每个 GRU 层
            out, _ = gru(out)  # 当前 GRU 层前向
            out = self.dropout(out)  # 施加 dropout
        return self.fc(out[:, -1, :])  # 用末时刻输出回归预测

class LSTMModel(nn.Module):
    """LSTM 回归模型：单向 LSTM 编码序列，取末时刻输出做预测。"""
    def __init__(self, input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.3):
        """初始化 LSTM 模型，参数含义同 GRUModel。"""
        super(LSTMModel, self).__init__()  # 调用父类初始化
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0, batch_first=True)  # 定义 LSTM 层
        self.fc = nn.Linear(lstm_hidden, 1)  # 输出层

    def forward(self, x):
        """前向传播：序列 -> 预测值。"""
        out, _ = self.lstm(x)  # 运行 LSTM
        return self.fc(out[:, -1, :])  # 末时刻输出做预测

class Chomp1d(nn.Module):
    """裁剪模块：去掉因果卷积在序列尾部多出的 padding 部分。"""
    def __init__(self, chomp_size):
        """初始化裁剪模块。

        参数:
            chomp_size: 需要从尾部裁剪掉的时间步数。
        """
        super(Chomp1d, self).__init__()  # 调用父类初始化
        self.chomp_size = chomp_size  # 保存裁剪长度

    def forward(self, x):
        """裁掉张量最后一维的尾部若干步并保证内存连续。"""
        return x[:, :, :-self.chomp_size].contiguous()  # 去除尾部 padding 并转为连续张量

class TemporalBlock(nn.Module):
    """TCN 的基本时间块：两膨胀卷积 + 裁剪 + ReLU + Dropout，带残差连接。"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        """初始化时间块。

        参数:
            n_inputs/n_outputs: 输入/输出通道数；kernel_size: 卷积核大小。
            stride: 卷积步长；dilation: 膨胀系数；padding: 填充步数。
            dropout: 丢弃率，默认 0.3。
        """
        super(TemporalBlock, self).__init__()  # 调用父类初始化
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)  # 第一层膨胀卷积
        self.chomp1 = Chomp1d(padding)  # 裁掉第一层卷积的尾部 padding
        self.relu1 = nn.ReLU()  # 第一层激活
        self.dropout1 = nn.Dropout(dropout)  # 第一层 dropout
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)  # 第二层膨胀卷积
        self.chomp2 = Chomp1d(padding)  # 裁掉第二层卷积的尾部 padding
        self.relu2 = nn.ReLU()  # 第二层激活
        self.dropout2 = nn.Dropout(dropout)  # 第二层 dropout
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 将卷积支路串成序列
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 通道数不一致时用 1x1 卷积做残差升维
        self.relu = nn.ReLU()  # 残差相加后的最终激活

    def forward(self, x):
        """前向传播：主支路输出与残差相加后激活。"""
        out = self.net(x)  # 通过卷积主支路
        res = x if self.downsample is None else self.downsample(x)  # 通道一致直接用输入，否则升维
        return self.relu(out + res)  # 残差相加并激活

class TCNModel(nn.Module):
    """时序卷积网络（TCN）回归模型：多层 TemporalBlock 后取末时刻做预测。"""
    def __init__(self, input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.3):
        """初始化 TCN 模型。

        参数:
            input_dim: 输入特征数；time_step: 时间步数（仅记录）。
            num_channels: 每层通道数列表；kernel_size: 卷积核大小。
            dropout: 丢弃率，默认 0.3。
        """
        super(TCNModel, self).__init__()  # 调用父类初始化
        layers = []  # 初始化层列表
        for i in range(len(num_channels)):  # 按通道配置逐层构建时间块
            dilation_size = 2 ** i  # 膨胀系数按 2 的幂递增
            in_channels = input_dim if i == 0 else num_channels[i-1]  # 首层输入为特征数，其余为上一层通道数
            out_channels = num_channels[i]  # 本层输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]  # 添加一个时间块
        self.network = nn.Sequential(*layers)  # 将各时间块串成网络
        self.fc = nn.Linear(num_channels[-1], 1)  # 最后一层通道映射到预测值

    def forward(self, x):
        """前向传播：把 (batch, seq, feat) 转置后送入 TCN 并取末时刻预测。"""
        x = x.transpose(1, 2)  # 转为 (batch, feat, seq) 以适配 Conv1d
        out = self.network(x)  # 通过 TCN 主体
        return self.fc(out[:, :, -1])  # 取最后一个时间步的通道输出做预测

# ===================== 小波分解函数 =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):
    """滑动窗口小波分解 - 不保留原始TD列

    参数:
        data: 原始特征 DataFrame（行=时间，列=变量）。
        target_col: 需要做小波分解的目标列名。
        wavelet: 使用的小波基函数，默认 db4。
        level: 分解层数，默认 3。
        window_size: 滑动窗口长度，默认按层数自适应。
    返回:
        DataFrame，目标列被其近似系数与各层细节系数替代，其余列原样保留。
    """
    if window_size is None:  # 未指定窗口大小时
        window_size = min(2 ** (level + 6), len(data))  # 按分解层数自适应，且不超过数据长度
    
    wavelet_features = pd.DataFrame(index=data.index)  # 构造与原数据同索引的空结果表
    
    for col in data.columns:  # 遍历每个原始列
        if col == 'timestamp':  # 时间戳列不参与分解
            continue  # 跳过
        
        signal = data[col].values  # 取出当前列的数值序列
        
        if col == target_col:  # 仅对目标列做小波分解
            approx_list = []  # 收集每步的近似系数
            detail_lists = [[] for _ in range(level)]  # 为每层细节系数各建一个列表
            
            for i in range(len(signal)):  # 逐时间步滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证不越界
                window_signal = signal[start_idx:i+1]  # 截取截至当前点的窗口信号
                
                if len(window_signal) < window_size:  # 窗口不足长度时
                    pad_width = window_size - len(window_signal)  # 计算需补齐的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值在左侧补齐
                
                try:  # 尝试对当前窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取该层细节系数末值，空则用 0 补
                except:  # 分解异常时采用原始值兜底
                    approx_list.append(window_signal[-1])  # 分解失败时退化为窗口最后一个原始值
                    for j in range(level):  # 为每层细节系数
                        detail_lists[j].append(0.0)  # 填充 0
            
            # 不保留原始TD列，只添加小波特征
            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数列
            for j in range(level):  # 逐层
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入第 j+1 层细节系数列
        else:  # 非目标列无需分解
            wavelet_features[col] = signal  # 非目标列原样复制，不做分解
    
    return wavelet_features  # 返回构造好的小波特征表

# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始加载测试数据
data = pd.read_csv('部分温湿度数据.csv', encoding='gbk')  # 以 GBK 编码读取测试数据 CSV
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m', origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟步长和指定起点生成时间戳

features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 预定义候选特征列表
target = '温度偏差'  # 指定目标列
features = [f for f in features if f in data.columns]  # 仅保留实际存在的特征列

# 缺失值处理
for col in ['温度偏差', '外界温度', 'CO2', '目标温度','相对湿度','TD']:  # 遍历需要插值的连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 列存在且有缺失时才处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')  # 线性插值后再前后填充

for col in ['日龄']:  # 遍历需填充的离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 列存在且有缺失时才处理
        data[col] = data[col].fillna(method='ffill').fillna(method='bfill')  # 前后向填充离散缺失

data = data.dropna(subset=features + [target]).reset_index(drop=True)  # 删除仍含缺失的行并重置索引
print(f"数据形状: {data.shape}")  # 打印处理后数据形状

# ===================== 模型配置 =====================
models_config = [  # 定义待对比模型列表：名称、权重路径、绘图颜色与线型
    {'name': 'LSTM', 'path': 'pth/Model_LSTM_pytorch.pth', 'color': '#FFA500', 'linestyle': '--'},
    {'name': 'TCN', 'path': 'pth/Model_TCN_pytorch.pth', 'color': '#00CED1', 'linestyle': '--'},
    {'name': 'GRU', 'path': 'pth/Model_GRU_pytorch.pth', 'color': '#4169E1', 'linestyle': '--'},
    {'name': 'GRU–ECA', 'path': 'pth/Model_GRUECA_pytorch.pth', 'color': '#32CD32', 'linestyle': '--'},
    {'name': 'GRU–Wavelet', 'path': 'pth/Model_GRUWavelet_pytorch.pth', 'color': '#8B008B', 'linestyle': '--'},
    {'name': 'GRU–Wavelet–ECA', 'path': 'pth/Model_GRUWavelet_no_leakage_pytorch.pth', 'color': '#DC143C', 'linestyle': '--'},
]
# ===================== 加载模型并预测 =====================
predictions = {}  # 存放每个模型的预测与真实值
metrics = {}  # 存放每个模型的评估指标

for model_cfg in models_config:  # 逐个模型执行加载与推理
    print(f"\n{'='*60}")  # 打印分隔线
    print(f"加载模型: {model_cfg['name']}")  # 打印当前模型名称
    print(f"{'='*60}")  # 打印分隔线
    
    try:  # 尝试加载当前模型并检查点推理，失败时跳过该模型
        checkpoint = torch.load(model_cfg['path'], map_location=device, weights_only=False)  # 加载当前模型的检查点
        
        time_step = checkpoint['time_step']  # 从检查点读取时间窗口
        pred_offset = checkpoint.get('pred_offset', 6)  # 默认6，但优先使用保存的值  # 读取预测偏移，缺省为 6
        feature_scaler = checkpoint.get('feature_scaler')  # 读取特征归一化器，可能为空
        target_scaler = checkpoint.get('target_scaler')  # 读取目标归一化器，可能为空
        
        print(f"  time_step: {time_step}, pred_offset: {pred_offset}")  # 打印窗口与偏移配置
        
        # 判断是否为小波模型
        is_wavelet = 'wavelet' in checkpoint or 'level' in checkpoint  # 检查点含小波标记则视为小波模型
        has_eca = any('eca' in key for key in checkpoint['model_state_dict'].keys())  # 权重名含 eca 则含注意力层
        
        print(f"  is_wavelet: {is_wavelet}, has_eca: {has_eca}")  # 打印模型类型判定结果
        
        # 准备特征
        if is_wavelet:  # 小波模型需构造小波特征
            print(f"  小波模型")  # 提示当前为小波模型
            wavelet_target_col = checkpoint.get('wavelet_target_col', target)  # 读取分解目标列，缺省为温度偏差
            wavelet_features = wavelet_decompose_sliding(  # 对测试数据做滑动窗口小波分解
                data[features],  # 输入特征表
                target_col=wavelet_target_col,  # 指定目标列
                wavelet='db4',  # 小波基
                level=3,  # 分解层数
                window_size=time_step  # 窗口长度取时间步数
            )
            
            saved_feature_names = checkpoint.get('feature_names', list(wavelet_features.columns))  # 读取训练时保存的特征名，缺省为当前全部列
            test_features_df = wavelet_features[saved_feature_names]  # 按保存顺序挑选特征列
            print(f"  特征: {saved_feature_names}")  # 打印使用的特征名
        else:  # 非小波模型
            print(f"  非小波模型")  # 提示当前为普通模型
            test_features_df = data[features].copy()  # 直接取原始特征列副本
        
        # 归一化特征
        if feature_scaler is None:  # 检查点未保存特征归一化器
            print(f"  警告：feature_scaler 为 None，需要重新创建")  # 提示重新拟合
            train_data = pd.read_csv('温湿度数据.csv')  # 读取训练数据
            feature_scaler = MinMaxScaler()  # 新建特征归一化器
            target_scaler = MinMaxScaler()  # 新建目标归一化器
            
            # 如果是小波模型，需要先对训练数据做小波分解
            if is_wavelet:  # 小波模型需先分解训练数据
                train_wavelet = wavelet_decompose_sliding(train_data[features], target_col='温度偏差', wavelet='db4', level=3, window_size=time_step)  # 对训练特征做小波分解
                feature_scaler.fit(train_wavelet[saved_feature_names])  # 用分解后的特征拟合缩放器
            else:  # 普通模型
                feature_scaler.fit(train_data[features])  # 用原始训练特征拟合缩放器
            target_scaler.fit(train_data[[target]])  # 拟合目标缩放器
        else:  # 已有归一化器
            print(f"  使用模型保存的 feature_scaler")  # 提示复用保存的归一化器
        
        test_features_normalized = pd.DataFrame(  # 构造归一化后的测试特征表
            feature_scaler.transform(test_features_df),  # 用归一化器变换测试特征
            columns=test_features_df.columns  # 保持列名
        )
        
        # 创建序列
        def create_sequences(data, time_step, pred_offset):
            """把二维特征表滑窗切成 (样本数, time_step, 特征数) 的三维序列。"""
            sequences = []  # 初始化窗口列表
            for i in range(len(data) - time_step - pred_offset):  # 滑动起点，预留窗口与偏移
                sequences.append(data.iloc[i:i + time_step].values)  # 截取窗口转数组
            return np.array(sequences)  # 堆叠返回
        
        X_test = create_sequences(test_features_normalized, time_step, pred_offset)  # 构造测试输入序列
        
        # 关键：y_true 对齐，与训练脚本保持一致（time_step + pred_offset - 1）
        y_true_normalized = target_scaler.transform(data[[target]]).flatten()  # 归一化目标列并展平
        y_true_normalized_seq = y_true_normalized[time_step + pred_offset - 1:time_step + pred_offset - 1 + len(X_test)]  # 与训练一致地偏移截取真实值
        
        print(f"  X_test: {X_test.shape}, y_true: {y_true_normalized_seq.shape}")  # 打印输入与真实值形状
        
        # 构建模型
        input_dim = X_test.shape[2]  # 输入维度由测试序列最后一维决定
        
        if is_wavelet and has_eca:  # 小波 + 注意力
            model = GRUWaveletECAModel(input_dim, time_step).to(device)  # 构建 GRU–小波–ECA 模型
        elif is_wavelet and not has_eca:  # 小波无注意力
            model = GRUWaveletModel(input_dim, time_step).to(device)  # 构建 GRU–小波模型
        elif has_eca:  # 仅注意力
            model = GRUECAModel(input_dim, time_step).to(device)  # 构建 GRU–ECA 模型
        elif 'lstm' in model_cfg['path'].lower():  # 路径含 lstm
            model = LSTMModel(input_dim, time_step).to(device)  # 构建 LSTM 模型
        elif 'tcn' in model_cfg['path'].lower():  # 路径含 tcn
            model = TCNModel(input_dim, time_step).to(device)  # 构建 TCN 模型
        else:  # 其余按 GRU 处理
            model = GRUModel(input_dim, time_step).to(device)  # 构建基础 GRU 模型
        # 调试：检查模型权重形状
        if model_cfg['name'] == 'GRU+Wavelet':  # 仅对该名称打印形状调试信息
            print(f"  模型权重形状:")  # 提示输出检查点权重形状
            for name, param in checkpoint['model_state_dict'].items():  # 遍历检查点权重
                print(f"    {name}: {param.shape}")  # 打印名称与形状
            print(f"  模型结构:")  # 提示输出模型参数形状
            for name, param in model.named_parameters():  # 遍历模型参数
                print(f"    {name}: {param.shape}")  # 打印名称与形状
        # 加载模型权重，忽略不匹配的键
        model_state_dict = checkpoint['model_state_dict']  # 取出检查点权重字典
        model_dict = model.state_dict()  # 取出当前模型参数字典
        # 过滤掉不匹配的键
        filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}  # 只保留名称与形状都匹配的权重
        # 检查是否有未加载的键
        missing_keys = set(model_dict.keys()) - set(filtered_state_dict.keys())  # 找出未成功加载的模型键
        unexpected_keys = set(model_state_dict.keys()) - set(model_dict.keys())  # 找出检查点中多余的键
        if missing_keys:  # 存在未加载键
            print(f"  警告: 以下键未从checkpoint加载: {missing_keys}")  # 打印警告
        if unexpected_keys:  # 存在多余键
            print(f"  警告: 以下键在checkpoint中但模型中不存在(已忽略): {unexpected_keys}")  # 打印警告
        model_dict.update(filtered_state_dict)  # 用过滤后的权重更新模型字典
        model.load_state_dict(model_dict)  # 加载权重到模型
        model.eval()  # 切换评估模式
        # 预测（归一化空间）
        X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试序列转张量并搬到设备
        with torch.no_grad():  # 关闭梯度计算
            y_pred_normalized = model(X_test_tensor).cpu().numpy().flatten()  # 推理并展平预测
        # 反归一化到原始空间
        y_pred = target_scaler.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()  # 预测反归一化
        y_true = target_scaler.inverse_transform(y_true_normalized_seq.reshape(-1, 1)).flatten()  # 真实值反归一化
        # 调试输出
        if model_cfg['name'] in ['GRU+Wavelet', 'GRU+Wavelet+ECA']:  # 对小波模型打印调试信息
            print(f"  调试信息:")  # 提示输出调试
            print(f"    y_pred 前5个值: {y_pred[:5]}")  # 打印预测前五个值
            print(f"    y_true 前5个值: {y_true[:5]}")  # 打印真实前五个值
            print(f"    y_pred 范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")  # 打印预测值范围
            print(f"    y_true 范围: [{y_true.min():.4f}, {y_true.max():.4f}]")  # 打印真实值范围
        # 保存结果
        predictions[model_cfg['name']] = {'y_pred': y_pred, 'y_true': y_true}  # 按模型名保存预测与真实
        # 计算指标
        r2 = r2_score(y_true, y_pred)  # 计算 R²
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算 RMSE
        mae = mean_absolute_error(y_true, y_pred)  # 计算 MAE
        metrics[model_cfg['name']] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}  # 保存指标
        print(f"  R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")  # 打印当前模型指标
    except Exception as e:  # 单个模型失败不影响其余模型
        print(f"  [失败] {e}")  # 打印失败原因
        print(f"  模型路径: {model_cfg['path']}")  # 打印失败模型路径
        import traceback  # 导入 traceback 用于打印调用栈
        traceback.print_exc()  # 输出详细异常栈
        continue  # 继续下一个模型
# ===================== 统一长度 =====================
if len(predictions) > 0:  # 仅当至少有一个模型预测成功时绘图
    min_len = min(len(pred['y_pred']) for pred in predictions.values())  # 取所有模型中最短的预测长度作为统一长度
    for name in predictions:  # 遍历每个模型
        predictions[name]['y_pred'] = predictions[name]['y_pred'][:min_len]  # 截断预测到统一长度
        predictions[name]['y_true'] = predictions[name]['y_true'][:min_len]  # 截断真实值到统一长度
    # 使用第一个模型的y_true
    y_true_unified = list(predictions.values())[0]['y_true']  # 取第一个模型的真实值作为绘图基准
    # ===================== 绘图 =====================
    print(f"\n{'='*60}")  # 打印分隔线
    print("绘制对比图")  # 打印绘图阶段标题
    print(f"{'='*60}")  # 打印分隔线
    timesteps = np.arange(len(y_true_unified))  # 生成横轴步数索引
    fig, ax = plt.subplots(figsize=(14, 6))  # 创建 14x6 英寸画布
    # 先绘制 Actual - 使用橙色实线
    ax.plot(timesteps, y_true_unified, label='Actual', linewidth=2.0,  # 以真实值为基准绘制 Actual 曲线
            color='#D2691E', alpha=1.0, zorder=10)  # 绘制真实值曲线并置于最上层
    # 按照配置顺序绘制模型预测 - 使用较细的实线
    for model_cfg in models_config:  # 按配置顺序逐个绘制模型
        if model_cfg['name'] in predictions:  # 仅绘制预测成功的模型
            y_pred = predictions[model_cfg['name']]['y_pred']  # 取出该模型预测
            ax.plot(timesteps, y_pred,  # 绘制该模型预测曲线
                   label=model_cfg['name'],  # 图例名称
                   linewidth=1.5,  # 线宽较细
                   color=model_cfg['color'],  # 使用配置颜色
                   linestyle=model_cfg['linestyle'],  # 使用配置线型
                   alpha=0.8)  # 半透明便于叠加
    ax.set_xlabel('Timestep', fontsize=16)  # 设置横轴标签
    ax.set_ylabel('TD', fontsize=16)  # 设置纵轴标签
    ax.set_xlim(-15, 600)  # 左侧留白，避免0刻度贴边  # 限制横轴显示范围
    ax.tick_params(axis='both', which='major', labelsize=16)  # 放大主刻度字号
    ax.legend(fontsize=16, loc='upper right', frameon=False)  # 去掉图例边框  # 右上角显示无框图例
    ax.grid(False)  # 不显示网格，保持简洁
    # 只保留左边和底部的边框
    ax.spines['top'].set_visible(False)  # 隐藏上边框
    ax.spines['right'].set_visible(False)  # 隐藏右边框
    ax.spines['left'].set_linewidth(1.2)  # 加粗左边框
    ax.spines['left'].set_color('black')  # 设置左边框为黑色
    ax.spines['bottom'].set_linewidth(1.2)  # 加粗底边框
    ax.spines['bottom'].set_color('black')  # 设置底边框为黑色
    plt.tight_layout()  # 自动紧凑布局
    plt.savefig('multi_model_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')  # 保存多模型对比图
    print(f"\n对比图已保存: multi_model_comparison.png")  # 打印保存提示
    # 打印指标
    print(f"\n{'='*60}")  # 打印分隔线
    print("模型性能对比")  # 打印性能表标题
    print(f"{'='*60}")  # 打印分隔线
    print(f"{'模型':<20} {'R2':<10} {'RMSE':<10} {'MAE':<10}")  # 打印表头
    print(f"{'-'*60}")  # 打印表头分隔线
    for model_name, metric in metrics.items():  # 遍历各模型指标
        print(f"{model_name:<20} {metric['R2']:<10.4f} {metric['RMSE']:<10.4f} {metric['MAE']:<10.4f}")  # 按行打印模型指标
    print(f"{'='*60}")  # 打印分隔线
print("\n完成！")  # 打印全部流程结束标志
