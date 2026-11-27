"""
GRU 模型测试脚本（支持小波变换）。

用于加载训练好的 GRU 模型并对新采集的温湿度数据逐点预测，
支持 GRU-Only 模型与 GRU+小波变换+ECA 模型两类权重文件。
脚本包含缺失值处理、滑动窗口小波分解（避免未来信息泄漏）、
序列构造、模型加载与预测、评估指标计算，以及预测结果与误差分布的可视化。
"""
import pandas as pd  # 导入 pandas，用于 CSV 数据读取与表格化统计
import numpy as np  # 导入 numpy，用于数组运算与数值计算
import matplotlib  # 导入 matplotlib 基础模块，为设置后端做准备
matplotlib.use('Agg')  # 设置 Agg 非交互后端，便于无图形界面下保存图像
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制预测曲线与误差分析图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、MSE、MAE 三个回归评估指标函数
import torch  # 导入 PyTorch 主包，用于张量计算与设备选择
import torch.nn as nn  # 导入神经网络模块，用于构建模型各层
import pywt  # 小波变换库，提供 wavedec/upcoef 等离散小波分解与重构函数
import os  # 导入 os，用于检查模型权重文件是否存在
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  # 导入刻度定位与格式化工具（本脚本中未直接使用）
import numpy as np  # 再次导入 numpy（与原文件保持一致，保持原样）
# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选用 CUDA（若可用）否则使用 CPU 作为计算设备
print(f"使用设备: {device}")  # 打印当前使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号


# ===================== 模型定义 =====================
class ECALayer(nn.Module):  # 定义 ECA 高效通道注意力层类
    """
    ECA-Net（Efficient Channel Attention）高效通道注意力模块。
    通过一维卷积在无降维的情况下实现跨通道信息交互，为各特征通道生成注意力权重，
    从而突出对温度偏差预测更重要的特征通道。
    参数: channels 为输入特征通道数；gamma、b 用于自适应推导一维卷积核大小。
    """
    def __init__(self, channels, gamma=2, b=1):  # 初始化 ECA 模块，channels 为输入通道数
        """
        根据输入通道数自适应计算卷积核大小，并构建池化、卷积与 Sigmoid 层。
        参数: channels 输入通道数；gamma、b 用于计算卷积核尺度基数。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应计算卷积核尺寸基数
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数，使 padding 保持对称
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 一维卷积跨通道学习注意力权重
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 将权重压缩到 0 到 1 之间
    
    def forward(self, x):  # 前向传播，对输入施加通道注意力加权
        """对形状 (batch, time, channels) 的输入施加通道注意力，返回同形状张量。"""
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step)
        x_permuted = x.permute(0, 2, 1)  # 将输入转置为 (batch, channels, time) 以适配 Conv1d
        
        # 全局平均池化
        y = self.avg_pool(x_permuted)  # 全局平均池化得到通道描述向量
        
        # 1D卷积
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 一维卷积后再转回原维度顺序
        
        # Sigmoid激活
        y = self.sigmoid(y)  # 用 Sigmoid 归一化为通道注意力权重
        
        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将通道权重逐通道乘回原特征
        
        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 转回 (batch, time, channels) 并返回


class GRUOnlyModel(nn.Module):  # 定义 GRU-Only 模型类（含 ECA 注意力）
    """
    GRU-Only 模型 + ECA-Net 通道注意力机制。
    输入形状为 (batch, time_step, input_dim)，输出形状为 (batch, 1)。
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 初始化模型结构参数
        """
        构建 ECA 注意力、单向 GRU 编码层与输出全连接层。
        参数: input_dim 输入特征数；time_step 时间步长；gru_hidden 隐藏维度；
              gru_layers GRU 层数；dropout 丢弃率（仅多层时生效）。
        """
        super(GRUOnlyModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间步长
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏维度

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块，通道数等于输入维度
        
        self.gru = nn.GRU(  # 构建单向单层 GRU 编码层，batch_first 表示批次维在前
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(gru_hidden, 1)  # 定义输出全连接层，将隐藏状态映射为单个预测值

    def forward(self, x):  # 前向传播
        """前向传播：ECA 增强输入后经 GRU 编码，取末时刻输出做单点预测。"""
        # ECA注意力增强输入特征
        x = self.eca(x)  # 用 ECA 对输入特征做通道注意力增强
        
        # GRU编码
        out, h = self.gru(x)  # GRU 编码得到各时刻输出与最终隐藏状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # 取最后一个时间步的输出
        
        return self.fc(out_last)  # 经全连接层得到单点预测


class GRUWaveletModel(nn.Module):  # 定义 GRU + 小波变换 + ECA 模型类
    """
    GRU + 小波变换 + ECA 注意力机制预测模型。
    支持以列表形式为每层指定不同的 GRU 隐藏维度，层间使用 Dropout；
    输入形状为 (batch, time_step, input_dim)，输出形状为 (batch, 1)。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 初始化模型结构参数
        """
        构建 ECA 注意力与由 ModuleList 管理的一层或多层 GRU 编码栈。
        参数: input_dim 输入特征数；time_step 时间步长；gru_hidden 隐藏维度或维度列表；
              gru_layers GRU 层数；dropout 为层间丢弃率。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间步长
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏维度

        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块

        if isinstance(gru_hidden, list):  # 若隐藏维度以列表形式给出（支持每层不同维度）
            hidden_sizes = gru_hidden  # 直接采用给定的隐藏维度列表
        else:  # 否则
            hidden_sizes = [gru_hidden] * gru_layers  # 将同一隐藏维度复制 gru_layers 次

        self.gru_layers_list = nn.ModuleList()  # 用 ModuleList 管理多个 GRU 子层
        in_size = input_dim  # 记录当前层的输入维度
        for i, h_size in enumerate(hidden_sizes):  # 遍历每个隐藏维度
            self.gru_layers_list.append(  # 向 ModuleList 追加一个单层 GRU
                nn.GRU(
                    input_size=in_size,
                    hidden_size=h_size,
                    num_layers=1,
                    dropout=0.0,
                    batch_first=True,
                    bidirectional=False
                )
            )
            in_size = h_size  # 将下一层的输入维度更新为当前层的隐藏维度

        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 输出全连接层，输入维度为最后一层的隐藏维度

    def forward(self, x):  # 前向传播
        """前向传播：ECA 增强后逐层 GRU 编码并 Dropout，取末时刻输出预测。"""
        x = self.eca(x)  # 用 ECA 增强输入特征
        out = x  # 将增强后的输入作为首层输入
        for gru in self.gru_layers_list:  # 逐个 GRU 子层前向传播
            out, _ = gru(out)  # 当前层 GRU 编码
            out = self.dropout(out)  # 施加 Dropout 以防过拟合
        out_last = out[:, -1, :]  # 取最后时刻的输出
        return self.fc(out_last)  # 全连接层输出预测值
# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始加载测试数据
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，依次尝试读取
data = None  # 初始化测试数据变量为空
for enc in encodings:  # 遍历候选编码
    try:  # 尝试用当前编码读取
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 以当前编码读取测试温湿度 CSV
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码异常
        continue  # 尝试下一种编码

if data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常终止脚本
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 按索引乘以 5 分钟并给定起始时间构造 timestamp 列
                                   origin=pd.Timestamp('2025/6/21 0:04'))
# 特征配置
features = ['日龄','温度偏差','CO2','目标温度','外界温度']  # 定义候选特征列表
target = '温度偏差'  # 设定目标列为温度偏差

# 过滤特征列（只保留数据中存在的列）
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列

print(f"使用特征: {features}")  # 打印使用的特征
print(f"数据形状: {data.shape}")  # 打印测试数据形状

# 检查目标列是否存在
if target not in data.columns:  # 检查目标列是否存在
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带温度偏差的CSV文件。")  # 若缺少目标列则抛出 KeyError

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印缺失值处理标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据形状
print(f"\n各列缺失值统计:")  # 打印缺失值统计标题
for col in features + [target]:  # 遍历特征与目标列
    if col in data.columns:  # 仅处理数据中存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值数量
        nan_pct = nan_count / len(data) * 100  # 计算该列缺失值所占百分比
        if nan_count > 0:  # 仅当存在缺失值时打印
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失值数量与比例

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续变量，采用线性插值与前后填充处理缺失值
discrete_vars = ['日龄']  # 离散变量，采用前向与后向填充处理缺失值

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印当前处理提示
        before_nan = data[col].isna().sum()  # 记录处理前的缺失值数量
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 对连续变量做线性插值，适合时间序列连续变化
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].ffill()  # 前向填充开头的缺失值
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].bfill()  # 后向填充结尾的缺失值
        
        after_nan = data[col].isna().sum()  # 记录处理后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前的缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后的缺失值数量
        
# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印当前处理提示
        before_nan = data[col].isna().sum()  # 记录处理前的缺失值数量
        
        # 前向填充
        data[col] = data[col].ffill()  # 对离散变量做前向填充
        # 后向填充
        data[col] = data[col].bfill()  # 对离散变量做后向填充
        
        after_nan = data[col].isna().sum()  # 记录处理后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前的缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后的缺失值数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征与目标列剩余的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除仍含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后的数据形状
print(f"{'='*60}\n")  # 打印分隔线

# 重置索引
data = data.reset_index(drop=True)  # 重置索引，保证索引连续

# ===================== 小波变换函数（无数据泄漏版本，仅分解温度偏差） =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None, plot_decomposition=False, plot_feature='温度偏差'):  # 定义无数据泄漏的滑动窗口小波分解函数，仅分解目标列
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
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印开始小波分解的提示
    print(f"小波基函数: {wavelet}")  # 打印所使用的小波基函数
    print(f"分解层数: {level}")  # 打印分解层数
    # 自动计算窗口大小：确保足够大以进行有效的小波分解
    if window_size is None:  # 当未指定窗口大小时
        window_size = min(36, len(data))  # 使用 36 作为默认窗口大小（与原文件一致）
    
    print(f"滑动窗口大小: {window_size}")  # 打印自动计算出的滑动窗口大小
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空结果表
    
    # 用于绘图的数据
    plot_data = {}  # 初始化用于绘图的数据字典
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 跳过时间列
            continue  # 不参与分解
        
        signal = data[col].values  # 取出当前列的 numpy 数组信号
        
        # 只对目标列进行小波分解
        if col == target_col:  # 仅对目标列执行小波分解
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 初始化近似系数序列列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数各建一个空列表
            
            for i in range(len(signal)):  # 逐时间点滑动窗口
                # 只使用当前时间点之前的数据（包括当前点）
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，只使用当前及历史数据
                window_signal = signal[start_idx:i+1]  # 截取从起点到当前点的窗口信号
                
                # 如果窗口太小，用边界值填充到window_size
                if len(window_signal) < window_size:  # 当窗口长度不足时
                    pad_width = window_size - len(window_signal)  # 计算需要填充的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边界值在前端填充至指定窗口长度
                
                # 小波分解
                try:  # 尝试对窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 wavedec 做 level 层离散小波分解
                    
                    # 取最后一个值作为当前时间点的特征
                    approx_list.append(coeffs[0][-1])  # 取近似系数末值作为当前点近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取该层细节系数末值，空则记为 0
                except Exception as e:  # 捕获分解异常（异常对象 e 未使用，与原文件一致）
                    # 如果分解失败，使用原始值
                    approx_list.append(window_signal[-1])  # 用窗口末值作为近似特征回退
                    for j in range(level):  # 遍历各层细节特征
                        detail_lists[j].append(0.0)  # 将细节特征回退为 0.0
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 将近似系数序列写入结果表
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 将每层细节系数写入对应特征列
            
            # 保存指定特征的数据用于绘图
            if plot_decomposition and col == plot_feature:  # 若需要绘图且当前列为指定绘图特征
                plot_data['col_name'] = col  # 记录列名
                plot_data['original'] = signal  # 记录原始信号
                plot_data['approx'] = np.array(approx_list)  # 记录近似系数数组
                plot_data['details'] = [np.array(detail_lists[j]) for j in range(level)]  # 记录各层细节系数数组
                print(f"已保存 '{col}' 的小波分解数据用于绘图")  # 打印已保存绘图数据的提示
        else:  # 对于非目标列
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列保持原样的提示
            wavelet_features[col] = signal  # 将该列原样保留到结果表
    
    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印分解完成提示与生成特征数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印小波分解特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印其他原始特征数量
    
    # 绘制小波分解效果图
    if plot_decomposition and plot_data:  # 若需要绘图且已采集到绘图数据
        plot_wavelet_decomposition(plot_data, wavelet, level)  # 调用绘图函数绘制分解效果图
    elif plot_decomposition and not plot_data:  # 否则若需要绘图但没有数据
        print(f"⚠️  警告：未找到特征 '{plot_feature}'，无法绘制分解图")  # 打印未找到特征的警告
    
    return wavelet_features  # 返回小波分解后的特征表


def plot_wavelet_decomposition(plot_data, wavelet, level):  # 定义 SCI 论文风格的小波分解绘图函数
    """
    绘制小波分解效果图，版式按 SCI 期刊论文标准设计。
    Plot wavelet decomposition for SCI publication.
    参数:
        plot_data: 含原始信号与分解结果的字典（Dictionary containing original signal and decomposition results）。
        wavelet: 小波基函数名称（Wavelet basis function name）。
        level: 分解层数（Decomposition level）。
    返回: 无返回值；图像保存为 PNG 与 PDF 两种格式。
    """
    # Set publication-quality parameters
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体族为 Times New Roman
    plt.rcParams['font.size'] = 14  # 设置全局字号
    plt.rcParams['axes.linewidth'] = 1.2  # 设置坐标轴线条宽度
    
    col_name = plot_data['col_name']  # 从绘图数据字典中取出列名
    original = plot_data['original']  # 取出原始信号
    approx = plot_data['approx']  # 取出近似系数
    details = plot_data['details']  # 取出各层细节系数
    
    # Plot first 1000 points for better visualization
    plot_length = min(800, len(original))  # 取前 800 个点绘图以获得更清晰的可视化
    x_axis = np.arange(plot_length)  # 生成横轴索引数组
    
    # Create subplots
    fig, axes = plt.subplots(level + 2, 1, figsize=(12, 10))  # 创建 level+2 行的子图画布
    
    # Plot original signal
    axes[0].plot(x_axis, original[:plot_length], linewidth=1.2, color='#2C3E50')  # 在第一个子图绘制原始信号
    axes[0].set_title('(a) Original Signal: Temperature Deviation',  # 设置原始信号子图标题
                      fontsize=16, fontweight='bold', loc='left')
    axes[0].set_ylabel('Amplitude (°C)', fontsize=14)  # 设置纵轴标签（幅值，摄氏度）
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 添加虚线网格
    axes[0].set_xlim(0, plot_length)  # 设置横轴范围
    axes[0].tick_params(labelsize=13)  # 设置刻度标签字号
    
    # Plot approximation coefficients (low frequency)
    axes[1].plot(x_axis, approx[:plot_length], linewidth=1.2, color='#27AE60')  # 在第二个子图绘制近似（低频）系数
    axes[1].set_title('(b) Approximation Coefficients (Low Frequency)',   # 设置近似系数子图标题
                      fontsize=16, fontweight='bold', loc='left')
    axes[1].set_ylabel('Amplitude (°C)', fontsize=14)  # 设置纵轴标签
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 添加虚线网格
    axes[1].set_xlim(0, plot_length)  # 设置横轴范围
    axes[1].tick_params(labelsize=13)  # 设置刻度标签字号
    
    # Plot detail coefficients (high frequency)
    colors = ['#E74C3C', '#F39C12', '#9B59B6', '#34495E', '#E91E63']  # 定义各层细节系数的绘图颜色
    detail_labels = ['D₁', 'D₂', 'D₃', 'D₄', 'D₅']  # 定义各层细节系数的符号标签
    titles = [  # 定义各层细节系数的子图标题列表
        '(c) Detail Coefficients D₁ (High Frequency)',
        '(d) Detail Coefficients D₂ (Medium Frequency)',
        '(e) Detail Coefficients D₃ (Highest Frequency)',
        '(f) Detail Coefficients D₄',
        '(g) Detail Coefficients D₅'
    ]
    
    for i in range(level):  # 遍历每一层细节系数
        axes[i + 2].plot(x_axis, details[i][:plot_length],   # 在对应子图绘制该层细节系数
                        linewidth=1.2, color=colors[i % len(colors)])
        axes[i + 2].set_title(titles[i], fontsize=16, fontweight='bold', loc='left')  # 设置细节系数子图标题
        axes[i + 2].set_ylabel('Amplitude (°C)', fontsize=14)  # 设置纵轴标签
        axes[i + 2].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)  # 添加虚线网格
        axes[i + 2].set_xlim(0, plot_length)  # 设置横轴范围
        axes[i + 2].tick_params(labelsize=13)  # 设置刻度标签字号
    
    axes[-1].set_xlabel('Timestep', fontsize=14)  # 为最后一个子图设置横轴标签（时间步）
    axes[-1].tick_params(labelsize=13)  # 设置最后一个子图的刻度字号
    
    plt.tight_layout()  # 调整子图布局使之紧凑
    
    # Save in both PNG and PDF formats for publication
    filename_png = f'wavelet_decomposition_{col_name}_sci.png'  # 生成 PNG 格式的文件名
    filename_pdf = f'wavelet_decomposition_{col_name}_sci.pdf'  # 生成 PDF 格式的文件名
    
    plt.savefig(filename_png, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存 PNG 图像
    plt.savefig(filename_pdf, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存 PDF 图像
    
    print(f"\n[OK] Wavelet decomposition figure saved (SCI format):")  # 打印图像已保存的提示
    print(f"  - {filename_png}")  # 打印 PNG 文件路径
    print(f"  - {filename_pdf}")  # 打印 PDF 文件路径
    plt.close()  # 关闭当前图形释放内存
    
    # Restore default font settings
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 恢复中文字体设置
    plt.rcParams['font.family'] = 'sans-serif'  # 恢复字体族为 sans-serif

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 提示开始加载训练好的模型
# 优先加载无数据泄漏版本的模型
USE_WAVELET = False  # 标志位：是否使用小波变换模型
NO_LEAKAGE = False  # 标志位：是否为无数据泄漏版本
# 尝试多个可能的模型路径
model_paths = [  # 候选模型权重路径列表
    'Model_GRUWavelet_no_leakage_pytorch.pth',
]

checkpoint = None  # 初始化 checkpoint 为空
MODEL_PATH = None  # 初始化最终模型路径为空

for path in model_paths:  # 遍历候选模型路径
    if os.path.exists(path):  # 仅当文件存在时尝试加载
        try:  # 尝试加载权重文件
            checkpoint = torch.load(path, map_location=device, weights_only=False)  # 加载模型权重（weights_only=False 以允许保存 scaler 对象）
            MODEL_PATH = path  # 记录实际加载的模型路径
            
            # 判断模型类型
            if 'no_leakage' in path or checkpoint.get('no_data_leakage', False):  # 根据路径名或 checkpoint 标记判断是否为无泄漏小波模型
                USE_WAVELET = True  # 启用小波变换标志
                NO_LEAKAGE = True  # 置位无数据泄漏标志
                print(f"成功加载小波变换模型（无数据泄漏版本）: {MODEL_PATH}")  # 打印无泄漏小波模型加载成功信息
            elif 'Wavelet' in path:  # 否则若路径名含 Wavelet
                USE_WAVELET = True  # 启用小波变换标志
                NO_LEAKAGE = False  # 清除无泄漏标志（旧版本）
                print(f"成功加载小波变换模型（旧版本）: {MODEL_PATH}")  # 打印旧版小波模型加载成功信息
                print("⚠️  警告：该模型可能存在数据泄漏问题")  # 打印可能存在数据泄漏的警告
            else:  # 否则视为普通 GRU 模型
                USE_WAVELET = False  # 关闭小波变换标志
                NO_LEAKAGE = False  # 清除无泄漏标志
                print(f"成功加载普通GRU模型: {MODEL_PATH}")  # 打印普通 GRU 模型加载成功信息
            break  # 加载成功后跳出路径循环
        except Exception as e:  # 捕获加载过程中的异常
            print(f"尝试加载 {path} 失败: {e}")  # 打印当前路径加载失败信息
            continue  # 继续尝试下一个路径

if checkpoint is None:  # 若没有任何模型被成功加载
    raise FileNotFoundError(f"无法找到模型文件。尝试的路径: {model_paths}")  # 抛出文件未找到异常

input_dim = checkpoint['input_dim']  # 从 checkpoint 读取输入特征维度
time_step = checkpoint['time_step']  # 从 checkpoint 读取时间步长

# ⚠️ 重要：从模型中获取训练时使用的原始特征列表
if 'original_features' in checkpoint:  # 若 checkpoint 保存了原始特征列表
    # 如果模型保存了原始特征列表，使用它
    model_features = checkpoint['original_features']  # 直接采用保存的原始特征列表
    print(f"\n从模型加载原始特征列表: {model_features}")  # 打印从模型加载的特征列表
else:  # 否则
    # 如果没有保存，尝试从feature_names反推（去掉小波后缀）
    feature_names_saved = checkpoint.get('feature_names', None)  # 尝试读取保存的特征名列表
    if feature_names_saved:  # 若存在特征名列表
        # 从小波特征名中提取原始特征名
        model_features = []  # 初始化反推的原始特征列表
        for fname in feature_names_saved:  # 遍历保存的每个特征名
            if '_approx' in fname:  # 若特征名含 _approx 后缀说明来自小波分解
                original_name = fname.replace('_approx', '')  # 去除 _approx 后缀得到原始特征名
                if original_name not in model_features:  # 避免重复添加
                    model_features.append(original_name)  # 将原始特征名加入列表
        print(f"\n从小波特征名反推原始特征: {model_features}")  # 打印反推得到的原始特征
    else:  # 若特征名列表也不存在
        # 如果都没有，使用默认特征（但会给出警告）
        model_features = features  # 退而使用测试数据中的默认特征
        print(f"\n⚠️  警告：模型未保存特征列表，使用测试数据的特征: {model_features}")  # 打印模型未保存特征列表的警告

# 检查测试数据是否包含所需特征
missing_features = [f for f in model_features if f not in data.columns]  # 检查测试数据是否缺少模型所需特征
if missing_features:  # 若有缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常终止脚本

# 使用模型训练时的特征
features = model_features  # 将特征更新为模型训练时使用的特征
print(f"\n最终使用特征: {features}")  # 打印最终使用的特征列表

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
train_data = None  # 初始化训练数据变量为空
for enc in encodings:  # 遍历候选编码
    try:  # 尝试用当前编码读取训练数据
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 读取训练数据 CSV 以获取归一化参数
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码异常
        continue  # 继续尝试下一种编码
if train_data is None:  # 若所有编码均失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常

# 检查训练数据中是否包含所需特征
print(f"\n训练数据列名: {list(train_data.columns)}")  # 打印训练数据的列名
missing_in_train = [f for f in features if f not in train_data.columns]  # 找出训练数据中缺失的特征
if missing_in_train:  # 若有缺失特征
    print(f"\n⚠️ 警告：训练数据缺少以下特征: {missing_in_train}")  # 打印训练数据缺失特征的警告
    print(f"可用的特征: {[col for col in train_data.columns if col != 'timestamp']}")  # 打印训练数据中可用的特征
    # 只使用训练数据中存在的特征
    features = [f for f in features if f in train_data.columns]  # 只保留训练数据中实际存在的特征
    print(f"调整后使用的特征: {features}")  # 打印调整后使用的特征
    
    if len(features) == 0:  # 若调整后没有任何可用特征
        raise ValueError("训练数据中没有可用的特征！")  # 抛出异常

if USE_WAVELET:  # 若使用小波变换模型
    print("\n使用小波变换特征...")  # 打印使用小波变换特征的提示
    
    # 获取小波参数
    wavelet_type = checkpoint.get('wavelet', 'db4')  # 从 checkpoint 读取小波基函数
    wavelet_level = checkpoint.get('level', 3)  # 从 checkpoint 读取分解层数
    wavelet_target_col = checkpoint.get('wavelet_target_col', '温度偏差')  # 获取小波分解的目标列（默认温度偏差）
    feature_names = checkpoint.get('feature_names', None)  # 读取保存的特征名列表
    
    print(f"  小波基函数: {wavelet_type}")  # 打印小波基函数
    print(f"  分解层数: {wavelet_level}")  # 打印分解层数
    print(f"  小波分解目标列: {wavelet_target_col}")  # 打印小波分解目标列
    print(f"  数据泄漏防护: {'是' if NO_LEAKAGE else '否'}")  # 打印是否启用数据泄漏防护
    
    # 从checkpoint中获取小波窗口大小，与训练时保持一致（训练时为time_step=48）
    wavelet_window_size = checkpoint.get('wavelet_window', time_step)  # 从 checkpoint 读取小波窗口大小，与训练时保持一致
    print(f"  小波窗口大小: {wavelet_window_size}（与训练时一致）")  # 打印小波窗口大小
    
    # 根据模型版本选择小波分解方法
    if NO_LEAKAGE:  # 根据模型版本选择对应的小波分解方法
        # 使用滑动窗口小波分解（无数据泄漏，仅分解温度偏差）
        print("\n对训练数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 打印对训练数据进行滑动窗口小波分解的提示
        train_wavelet_features = wavelet_decompose_sliding(train_data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window_size, plot_decomposition=False)  # 对训练数据做无泄漏小波分解（仅分解目标列）
        
        print("\n对测试数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 打印对测试数据进行滑动窗口小波分解的提示
        test_wavelet_features = wavelet_decompose_sliding(data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window_size, plot_decomposition=True, plot_feature=wavelet_target_col)  # 对测试数据做无泄漏小波分解并绘制分解图
    else:  # 否则（旧版小波模型）
        # 使用传统小波分解（可能有数据泄漏）
        print("\n⚠️  使用传统小波分解（可能存在数据泄漏）")  # 打印使用传统小波分解的警告
        
        def wavelet_decompose_legacy(data, wavelet='db4', level=3, plot_decomposition=False, plot_feature='温度偏差'):  # 定义传统小波分解函数（向后兼容，可能存在数据泄漏）
            """
            传统小波分解方法（向后兼容）。
            对全部特征列直接做整段 wavedec 分解并重构，可能引入未来信息泄漏，
            仅在加载旧版模型时使用。
            """
            wavelet_features = pd.DataFrame()  # 创建空的结果特征表
            plot_data = {}  # 初始化绘图数据字典
            for col in data.columns:  # 遍历输入数据的每一列
                if col == 'timestamp':  # 跳过时间列
                    continue  # 不参与分解
                signal = data[col].values  # 取出当前列的信号
                coeffs = pywt.wavedec(signal, wavelet, level=level)  # 对整段信号做离散小波分解
                approx = coeffs[0]  # 取出近似系数
                approx_recon = pywt.upcoef('a', approx, wavelet, level=level, take=len(signal))  # 用 upcoef 将近似系数重构为原长度信号
                wavelet_features[f'{col}_approx'] = approx_recon[:len(signal)]  # 将重构的近似信号写入特征表
                
                detail_recons = []  # 初始化细节系数重构结果列表
                for i, detail in enumerate(coeffs[1:], 1):  # 遍历各层细节系数
                    detail_recon = pywt.upcoef('d', detail, wavelet, level=level-i+1, take=len(signal))  # 用 upcoef 将细节系数重构为原长度信号
                    wavelet_features[f'{col}_detail{i}'] = detail_recon[:len(signal)]  # 将重构的细节信号写入特征表
                    detail_recons.append(detail_recon[:len(signal)])  # 记录该层细节重构结果供绘图使用
                
                # 保存指定特征的数据用于绘图
                if plot_decomposition and col == plot_feature:  # 若需要绘图且当前列为指定特征
                    plot_data['col_name'] = col  # 记录列名
                    plot_data['original'] = signal  # 记录原始信号
                    plot_data['approx'] = approx_recon[:len(signal)]  # 记录重构的近似信号
                    plot_data['details'] = detail_recons  # 记录各层重构的细节信号
                    print(f"已保存 '{col}' 的小波分解数据用于绘图")  # 打印已保存绘图数据的提示
            
            if plot_decomposition and plot_data:  # 若需要绘图且已采集到数据
                plot_wavelet_decomposition(plot_data, wavelet, level)  # 调用绘图函数
            elif plot_decomposition and not plot_data:  # 否则若需要绘图但无数据
                print(f"⚠️  警告：未找到特征 '{plot_feature}'，无法绘制分解图")  # 打印未找到特征的警告
            
            return wavelet_features  # 返回传统小波分解后的特征表
        
        train_wavelet_features = wavelet_decompose_legacy(train_data[features], wavelet=wavelet_type, level=wavelet_level, plot_decomposition=False)  # 用传统方法分解训练数据
        test_wavelet_features = wavelet_decompose_legacy(data[features], wavelet=wavelet_type, level=wavelet_level, plot_decomposition=True, plot_feature='温度偏差')  # 用传统方法分解测试数据并绘制分解图
    
    # 直接使用小波分解后的特征（已包含原始特征和小波特征）
    train_combined = train_wavelet_features  # 训练合并特征直接采用小波分解结果
    test_combined = test_wavelet_features  # 测试合并特征直接采用小波分解结果
    
    # 归一化（如果模型保存了scaler，直接使用）
    if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若 checkpoint 中保存了归一化器
        print("\n使用模型保存的归一化器...")  # 打印使用模型保存的归一化器
        feature_scaler = checkpoint['feature_scaler']  # 取出特征归一化器
        target_scaler = checkpoint['target_scaler']  # 取出目标归一化器
    else:  # 否则
        print("\n使用训练数据重新拟合归一化器...")  # 打印使用训练数据重新拟合归一化器的提示
        feature_scaler = MinMaxScaler()  # 创建新的特征归一化器
        target_scaler = MinMaxScaler()  # 创建新的目标归一化器
        feature_scaler.fit(train_combined)  # 在训练合并特征上拟合
        target_scaler.fit(train_data[[target]])  # 在训练目标上拟合
    
    # 保存原始列名
    original_columns = test_combined.columns  # 保存测试合并特征的原始列名
    
    test_combined = pd.DataFrame(  # 对测试合并特征归一化并重建为 DataFrame
        feature_scaler.transform(test_combined),
        columns=original_columns
    )
    
    print(f"  特征数量: {len(test_combined.columns)}")  # 打印特征数量
    print(f"  特征列表: {list(test_combined.columns)}")  # 打印特征列表
    
else:  # 否则（普通 GRU 模型）
    print("\n使用原始特征...")  # 打印使用原始特征的提示
    
    # 普通模型：仅使用原始特征
    feature_scaler = MinMaxScaler()  # 创建特征归一化器
    target_scaler = MinMaxScaler()  # 创建目标归一化器
    
    feature_scaler.fit(train_data[features])  # 在训练数据原始特征上拟合
    target_scaler.fit(train_data[[target]])  # 在训练目标上拟合
    
    test_combined = pd.DataFrame(  # 对测试原始特征归一化并重建为 DataFrame
        feature_scaler.transform(data[features]),
        columns=features
    )
    
    print(f"  特征数量: {len(features)}")  # 打印特征数量

# ===================== 序列构造 =====================
pred_offset = 6   # 与训练脚本 gru.py 保持一致（6 步 × 5 分钟 = 30 分钟超前预测）

def create_sequences(data, time_step):  # 定义序列构造函数
    """
    创建时间序列样本，将二维特征表滑窗切分为三维序列。
    注意：这里是对单个数据集（机器人新数据）进行序列构造，不涉及训练/测试划分。
    参数: data 为归一化后的特征表；time_step 为输入窗口长度。
    返回: 形状为 (样本数, time_step, 特征数) 的 numpy 数组。
    """
    sequences = []  # 初始化序列样本列表
    for i in range(len(data) - time_step - pred_offset + 1):  # 滑动窗口起点范围，尾部留出预测偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 取窗口内数据作为一个样本
    return np.array(sequences)  # 将样本列表转为 numpy 数组返回

print("\n创建序列数据...")  # 提示开始创建序列数据
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 对目标列做归一化处理

# 创建序列
X_test = create_sequences(test_combined, time_step)  # 由归一化后的测试合并特征构造序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 按预测偏移对齐测试目标值
X_test = X_test[:len(y_test)]  # 确保长度匹配  # 截断序列使其与目标长度一致

print(f"\n测试数据形状:")  # 打印测试数据形状的提示
print(f"  X_test: {X_test.shape}")  # 打印 X_test 的形状
print(f"  y_test: {y_test.shape}")  # 打印 y_test 的形状

# ===================== 构建模型 =====================
if USE_WAVELET:  # 若使用小波变换模型
    model = GRUWaveletModel(  # 构建 GRU + 小波 + ECA 模型并部署到设备
        input_dim=input_dim,
        time_step=time_step,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.3
    ).to(device)
else:  # 否则构建 GRU-Only 模型
    model = GRUOnlyModel(  # 构建 GRU-Only 模型并部署到设备
        input_dim=input_dim,
        time_step=time_step,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.3
    ).to(device)

model.load_state_dict(checkpoint['model_state_dict'])  # 载入训练好的模型权重
model.eval()  # 切换模型为评估模式

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型参数量

# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始预测
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试特征转为张量并部署到设备
with torch.no_grad():  # 关闭梯度计算
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并将结果取回为 numpy 数组
# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将归一化真实值反归一化还原量纲
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将归一化预测值反归一化还原量纲
# 计算评估指标
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 由 MSE 开方得到 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE
if USE_WAVELET:  # 若使用小波变换模型
    model_name = "GRU+Wavelet+ECA (No Leakage)" if NO_LEAKAGE else "GRU+Wavelet+ECA"  # 根据是否无泄漏选择模型名称
else:  # 否则
    model_name = "GRU-Only"  # 模型名称设为 GRU-Only
print("\n" + "=" * 60)  # 打印分隔线
print(f"{model_name} Model Test Results")  # 打印模型名称与测试结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R2 指标
print(f"MSE:       {mse:.4f}")  # 打印 MSE 指标
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE 指标
print(f"MAE:       {mae:.4f}")  # 打印 MAE 指标
print("=" * 60)  # 打印分隔线
# ===================== 结果可视化 =====================
# 只绘制前120个时间步长
plot_length = min(1600, len(y_test_rescaled))  # 设置绘图点数为前 1600 个时间步
y_test_plot = y_test_rescaled[:plot_length]  # 截取真实值绘图片段
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取预测值绘图片段

# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 用步数索引作为横轴

plt.figure(figsize=(15, 8))  # 创建 15×8 的预测结果画布
y_test_flat = y_test_plot.flatten()  # 将真实值片段展平
y_pred_flat = y_pred_plot.flatten()  # 将预测值片段展平
plt.tick_params(axis='both', labelsize=16)  # 设置刻度标签字号
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值曲线
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签
plt.ylabel('TD (℃)', fontsize=16)  # 设置纵轴标签（温度偏差，摄氏度）
# plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.tight_layout()  # 调整布局使之紧凑
plt.text(0.5, 0.95, '(f)', transform=plt.gca().transAxes,  # 在图上添加 (f) 子图编号文本
         fontsize=16, fontweight='bold', va='top', ha='center')
plt.tight_layout()  # 再次调整布局
if USE_WAVELET:  # 若使用小波变换模型
    output_filename = 'test_GRUWavelet_no_leakage_prediction.png' if NO_LEAKAGE else 'test_GRUWavelet_prediction.png'  # 根据是否无泄漏选择预测图文件名
else:  # 否则
    output_filename = 'test_GRUOnly_prediction.png'  # 使用 GRU-Only 预测图文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存预测结果图
print(f"\n预测结果图已保存: {output_filename}")  # 打印预测图保存路径
plt.close()  # 关闭图形

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算预测误差（预测值减真实值）

plt.figure(figsize=(15, 10))  # 创建 15×10 的误差分析画布

# 子图1: 预测值 vs 真实值散点图
# 子图1: 预测值 vs 真实值散点图
N = 1600  # 设定散点图使用的样本数 N 为 1600
y_true_n = y_test_rescaled[:N]  # 取前 N 个真实值
y_pred_n = y_pred_rescaled[:N]  # 取前 N 个预测值
r2_n = r2_score(y_true_n, y_pred_n)  # 计算前 N 个样本的 R2

plt.subplot(2, 2, 1)  # 选择第 1 个子图
plt.scatter(y_true_n, y_pred_n, alpha=0.5, s=20)  # 绘制真实值与预测值的散点图
plt.plot([y_true_n.min(), y_true_n.max()],  # 绘制对角线作为理想预测线
         [y_true_n.min(), y_true_n.max()],
         'r--', lw=2, label='理想预测线')
plt.xlabel('真实温度偏差 (℃)', fontsize=11)  # 设置散点图横轴标签
plt.ylabel('预测温度偏差 (℃)', fontsize=11)  # 设置散点图纵轴标签
plt.title(f'前{N}个样本：预测值 vs 真实值 (R²={r2_n:.4f})', fontsize=12)  # 设置散点图标题并显示 R2
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 选择第 2 个子图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差分布直方图
plt.xlabel('预测误差 (℃)', fontsize=11)  # 设置直方图横轴标签
plt.ylabel('频数', fontsize=11)  # 设置直方图纵轴标签
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 设置直方图标题并显示 MAE
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差红色虚线参考线
plt.grid(True, alpha=0.3, axis='y')  # 仅为纵轴添加网格
# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 选择第 3 个子图
error_steps = np.arange(min(120, len(errors)))  # 生成误差时序图的横轴索引（前 120 步）
plt.plot(error_steps, errors[:120], alpha=0.7, linewidth=1)  # 绘制误差时序曲线
plt.xlabel('Timestep', fontsize=11)  # 设置横轴标签
plt.ylabel('Prediction Error (℃)', fontsize=11)  # 设置纵轴标签
plt.title('Error Time Series', fontsize=12)  # 设置误差时序图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考线
plt.grid(True, alpha=0.3)  # 添加半透明网格
# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 选择第 4 个子图
box_data = [errors]  # 准备箱线图用的误差数据
plt.boxplot(box_data, tick_labels=[model_name])  # 绘制预测误差箱线图
plt.ylabel('预测误差 (℃)', fontsize=11)  # 设置箱线图纵轴标签
plt.title('误差统计分布', fontsize=12)  # 设置箱线图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考线
plt.grid(True, alpha=0.3, axis='y')  # 仅为纵轴添加网格
plt.tight_layout()  # 调整布局使之紧凑
if USE_WAVELET:  # 若使用小波变换模型
    error_filename = 'test_GRUWavelet_no_leakage_error_analysis.png' if NO_LEAKAGE else 'test_GRUWavelet_error_analysis.png'  # 根据是否无泄漏选择误差图文件名
else:  # 否则
    error_filename = 'test_GRUOnly_error_analysis.png'  # 使用 GRU-Only 误差图文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印误差分析图保存路径
import numpy as np  # 再次导入 numpy（与原文件保持一致，保持原样）
import matplotlib.pyplot as plt  # 再次导入 pyplot（与原文件保持一致，保持原样）
# 计算误差
errors = y_pred_rescaled - y_test_rescaled  # 重新计算预测误差（预测值减真实值）
# 如果是二维数组，先压平
errors = np.array(errors).flatten()  # 若为二维数组则先展平为一维
plt.figure(figsize=(8, 5))  # 创建 8×5 的误差分布画布
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)  # 绘制误差分布直方图（30 个区间）
# 画误差为 0 的参考线
plt.axvline(0, linestyle='--', linewidth=2, label='零误差线')  # 绘制零误差虚线参考线并标注图例

plt.xlabel('预测误差 (预测值 - 真实值)', fontsize=11)  # 设置横轴标签（预测误差）
plt.ylabel('频数', fontsize=11)  # 设置纵轴标签（频数）
plt.title('误差分布图', fontsize=12)  # 设置误差分布图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.show()  # 显示图像
plt.close()  # 关闭图形
# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("Error Statistics")  # 打印误差统计标题
print("=" * 60)  # 打印分隔线
print(f"Mean Error:     {np.mean(errors):.4f} C")  # 打印平均误差
print(f"Std Deviation:  {np.std(errors):.4f} C")  # 打印误差标准差
print(f"Max Error:      {np.max(errors):.4f} C")  # 打印最大误差
print(f"Min Error:      {np.min(errors):.4f} C")  # 打印最小误差
print(f"Median Error:   {np.median(errors):.4f} C")  # 打印误差中位数
print("=" * 60)  # 打印分隔线
print("\nTest Completed!")  # 打印测试完成提示






