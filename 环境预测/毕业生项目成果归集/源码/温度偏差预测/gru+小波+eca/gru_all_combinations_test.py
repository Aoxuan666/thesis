"""
GRU + 小波分解 + ECA 注意力机制：全特征组合穷举测试脚本。

本脚本以"温度偏差"为预测目标，读取温湿度时序 CSV 数据并完成缺失值处理；
利用滑动窗口小波分解（db4 基函数、3 层）提取温度偏差的近似系数与各层细节系数；
随后穷举"原始特征 + 小波特征"的全部 16 种组合，对每种组合执行 5 折交叉验证并训练最终模型，
在测试集上计算 R2、RMSE、MAE 等指标，将结果汇总到 CSV 文件并绘制多幅统计对比图。
脚本自动选用 CUDA 或 CPU 设备，使用 Agg 后端以便无图形界面下保存图像。
"""

import pandas as pd  # 导入 pandas，用于 CSV 读取与 DataFrame 表格化统计
import numpy as np  # 导入 numpy，用于数组运算、统计指标与数值处理
import time  # 导入 time，用于统计模型训练耗时
from itertools import combinations  # 导入组合生成器，用于枚举小波特征的所有子集
import matplotlib  # 导入 matplotlib 基础模块，为设置后端做准备
matplotlib.use('Agg')  # 在导入 pyplot 之前设置 Agg 非交互后端，便于无窗口保存图片
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制各类统计图表
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标的归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、MSE、MAE 三个回归评估指标函数
from sklearn.model_selection import KFold  # 导入 KFold，用于 5 折交叉验证的数据划分
import torch  # 导入 PyTorch 主包，用于张量计算与设备选择
import torch.nn as nn  # 导入神经网络模块，用于构建 ECA 与 GRU 层
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import DataLoader, TensorDataset  # 导入 DataLoader 与 TensorDataset，用于批量加载训练数据
import pywt  # 小波变换库（PyWavelets），提供 wavedec 等离散小波分解函数

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选用 CUDA（若可用）否则使用 CPU 作为计算设备
print(f"使用设备: {device}")  # 打印当前使用的计算设备
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文标题正常显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号，避免坐标轴负号显示为方块

# ===================== 数据加载 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，依次尝试直到成功读取文件
data = None  # 初始化数据变量为空，用于记录读取结果
for enc in encodings:  # 遍历候选编码列表
    try:  # 尝试用当前编码读取 CSV
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 以当前编码读取温湿度数据文件
        print(f"成功使用 {enc} 编码加载数据")  # 打印成功使用的编码方式
        break  # 读取成功后跳出编码尝试循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码相关的异常
        continue  # 解码失败则尝试下一种编码
if data is None:  # 若所有编码均读取失败
    raise ValueError("无法读取数据文件")  # 抛出无法读取数据的异常，终止脚本
data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按起始时间与 5 分钟频率构造 timestamp 时间列
features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 定义候选特征名列表
target = '温度偏差'  # 设定预测目标列为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列
# 缺失值处理
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 定义需做线性插值的连续变量列表
discrete_vars = ['日龄']  # 定义需做前后向填充的离散变量列表
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且存在缺失值时进行处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').ffill().bfill()  # 对连续变量依次做线性插值、前向填充与后向填充

for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且存在缺失值时进行处理
        data[col] = data[col].ffill().bfill()  # 对离散变量做前向填充与后向填充

data = data.dropna(subset=features + [target])  # 删除特征列或目标列仍含缺失值的行
print(f"数据形状: {data.shape}")  # 打印清洗后的数据形状

# ===================== 小波分解函数 =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):  # 定义滑动窗口小波分解函数，仅对目标列分解以避免未来信息泄漏
    """
    使用滑动窗口对目标列做小波分解，其他列原样保留，从而避免未来信息泄漏。
    参数:
        data: 输入的 DataFrame；
        target_col: 需要进行分解的列名（默认'温度偏差'）；
        wavelet: 小波基函数（默认 'db4'）；
        level: 分解层数（默认 3）；
        window_size: 滑动窗口长度，None 时按 2^(level+6) 自动计算。
    返回:
        包含原始列与小波近似/细节特征的新 DataFrame。
    """
    if window_size is None:  # 未指定窗口大小时
        window_size = min(2 ** (level + 6), len(data))  # 窗口取 2 的 (level+6) 次方与数据长度的较小值
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建一个与原数据同索引的空结果表
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 若是时间列则跳过
            continue  # 不参与小波分解
        
        signal = data[col].values  # 取出当前列的 numpy 数组信号
        
        if col == target_col:  # 仅对目标列执行小波分解
            approx_list = []  # 初始化近似系数序列列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数各建一个空列表
            
            for i in range(len(signal)):  # 逐个时间点滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证只使用当前及历史数据
                window_signal = signal[start_idx:i+1]  # 截取从起点到当前点的窗口信号
                
                if len(window_signal) < window_size:  # 当窗口长度不足时
                    pad_width = window_size - len(window_signal)  # 计算需要填充的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边界值在前端填充至指定窗口长度
                
                try:  # 尝试对窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 wavedec 对窗口信号做 level 层离散小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为当前点的近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取该层细节系数末值作为特征，若该层为空则记为 0
                except:  # 分解失败时采用回退策略（保留原样）
                    approx_list.append(window_signal[-1])  # 用窗口最后一个原始值作为近似特征回退
                    for j in range(level):  # 遍历各层细节特征
                        detail_lists[j].append(0.0)  # 将细节特征回退为 0.0
            
            wavelet_features[f'{col}_approx'] = approx_list  # 将近似系数序列写入结果表的近似特征列
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 将每层细节系数写入对应的特征列
        else:  # 对于非目标列
            wavelet_features[col] = signal  # 将该列数据原样保留到结果表
    
    return wavelet_features  # 返回小波分解后的特征表

# ===================== ECA注意力机制 =====================
class ECALayer(nn.Module):  # 定义 ECA 高效通道注意力层，经一维卷积生成通道权重
    """ECA 高效通道注意力模块：对通道维度自适应加权，突出重要特征通道。"""
    def __init__(self, channels, gamma=2, b=1):  # 初始化 ECA 模块，channels 为输入通道数
        """
        根据输入通道数自适应推导一维卷积核大小，并构建池化、卷积与 Sigmoid。
        参数: channels 为输入通道数；gamma、b 用于计算卷积核尺度。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应计算卷积核尺寸基数
        k_size = t if t % 2 else t + 1  # 保证卷积核大小为奇数
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 一维卷积跨通道学习注意力权重，padding 保持长度不变
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 将权重压缩到 0 到 1 之间

    def forward(self, x):  # 前向传播，对输入施加通道注意力加权
        """对形状 (batch, time, channels) 的输入施加通道注意力，返回同形状张量。"""
        x_permuted = x.permute(0, 2, 1)  # 将输入转置为 (batch, channels, time) 以适配 Conv1d
        y = self.avg_pool(x_permuted)  # 全局平均池化得到通道描述向量
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 一维卷积后再转回原维度顺序
        y = self.sigmoid(y)  # 用 Sigmoid 归一化为通道注意力权重
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将通道权重逐通道乘回原特征
        return x_permuted.permute(0, 2, 1)  # 转回 (batch, time, channels) 并返回

# ===================== GRU+ECA模型 =====================
class GRUECAModel(nn.Module):  # 定义 GRU + ECA 组合预测模型
    """GRU + ECA 模型：先用 ECA 增强输入，再由 GRU 编码，最后全连接输出单点预测。"""
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 初始化模型结构参数
        """
        构建 ECA 注意力、GRU 编码层与输出全连接层。
        参数: input_dim 输入特征数；time_step 时间步长；gru_hidden 隐藏维度；
              gru_layers GRU 层数；dropout 丢弃率。
        """
        super(GRUECAModel, self).__init__()  # 调用父类初始化
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块，通道数等于输入特征数
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 构建 GRU 编码层，多层时启用 dropout 并设置 batch_first
                         dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # GRU 层的 dropout 仅在多层时生效，batch_first 表示批次维在首维
        self.fc = nn.Linear(gru_hidden, 1)  # 定义输出全连接层，将隐藏状态映射为单个预测值

    def forward(self, x):  # 前向传播：依次经过 ECA、GRU 并输出预测
        """前向传播：依次经过 ECA 与 GRU，取最后时刻输出经全连接得到预测。"""
        x = self.eca(x)  # 用 ECA 对输入特征做通道注意力增强
        out, _ = self.gru(x)  # GRU 编码得到各时间步的输出序列
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出经全连接得到预测值

class EarlyStopping:  # 定义早停机制，验证损失长期不降时终止并保留最佳权重
    """早停器：记录最佳验证损失与模型权重，patience 轮无改善即置位停止标志。"""
    def __init__(self, patience=10):  # 初始化早停参数
        """设置耐心值，并初始化计数器、最佳损失、停止标志与最佳模型权重。"""
        self.patience = patience  # 记录容忍无改善的轮数 patience
        self.counter = 0  # 无改善计数器清零
        self.best_loss = None  # 最佳损失初始为空
        self.early_stop = False  # 早停标志初始为假
        self.best_model = None  # 最佳模型权重初始为空

    def __call__(self, val_loss, model):  # 每轮验证后调用，更新最佳权重或累计无改善次数
        """根据本次验证损失更新最佳模型；连续 patience 轮未改善则触发早停。"""
        if self.best_loss is None:  # 首次调用时
            self.best_loss = val_loss  # 将当前验证损失记为最佳损失
            self.best_model = model.state_dict().copy()  # 保存当前模型权重副本
        elif val_loss > self.best_loss:  # 本次验证损失变差时
            self.counter += 1  # 无改善计数加一
            if self.counter >= self.patience:  # 无改善轮数达到耐心阈值时
                self.early_stop = True  # 置位早停标志
        else:  # 本次验证损失改善时
            self.best_loss = val_loss  # 更新最佳损失
            self.best_model = model.state_dict().copy()  # 更新最佳模型权重副本
            self.counter = 0  # 重置无改善计数器
def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 定义训练函数，含早停并返回训练与验证损失曲线
    """
    在给定加载器上训练模型，使用 MSE 损失与 Adam 优化器，并配合早停机制。
    参数: model 待训练模型；train_loader/val_loader 训练与验证数据加载器；
          epochs 最大轮数；lr 学习率；patience 早停耐心值。
    返回: (train_losses, val_losses) 每轮平均训练与验证损失列表。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新全部可训练参数
    early_stopping = EarlyStopping(patience=patience)  # 创建早停器实例
    train_losses, val_losses = [], []  # 初始化训练与验证损失记录列表
    for epoch in range(epochs):  # 遍历训练轮次
        model.train()  # 切换为训练模式，启用 dropout 等训练专属行为
        train_loss = 0.0  # 累计本轮训练损失
        for batch_x, batch_y in train_loader:  # 按批次读取训练数据
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将批次数据搬运到计算设备
            optimizer.zero_grad()  # 清空上一轮累积的梯度
            outputs = model(batch_x)  # 前向计算得到预测输出
            loss = criterion(outputs, batch_y)  # 计算预测值与真值的均方误差损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新模型参数
            train_loss += loss.item()  # 累加当前批次的损失值
        train_loss /= len(train_loader)  # 求本轮平均训练损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        model.eval()  # 切换为评估模式，关闭 dropout
        val_loss = 0.0  # 累计本轮验证损失
        with torch.no_grad():  # 关闭梯度计算以节省显存与算力
            for batch_x, batch_y in val_loader:  # 按批次读取验证数据
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将验证批次数据搬运到计算设备
                outputs = model(batch_x)  # 前向计算得到验证预测
                val_loss += criterion(outputs, batch_y).item()  # 累加当前批次的验证损失
        val_loss /= len(val_loader)  # 求本轮平均验证损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        early_stopping(val_loss, model)  # 调用早停器更新最佳模型与停止状态
        if early_stopping.early_stop:  # 若触发早停
            model.load_state_dict(early_stopping.best_model)  # 载入训练过程中保存的最佳权重
            break  # 跳出训练轮次循环

    if early_stopping.best_model is not None:  # 训练结束后若曾保存最佳权重
        model.load_state_dict(early_stopping.best_model)  # 确保模型最终保留最佳状态
    return train_losses, val_losses  # 返回训练与验证损失曲线

def create_sequences(data, time_step, pred_offset):  # 定义序列构造函数，按时间步与预测偏移切分样本
    """
    将二维特征表滑窗切分为三维序列样本 (样本数, time_step, 特征数)。
    参数: data 特征表；time_step 输入窗口长度；pred_offset 预测偏移。
    返回: numpy 三维数组形式的序列样本。
    """
    sequences = []  # 初始化序列样本列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点范围，尾部留出预测偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 取窗口内数据作为一个样本
    return np.array(sequences)  # 将样本列表转为 numpy 数组返回

# ===================== 主测试流程 =====================
time_step = 36  # 设定 GRU 输入时间步长为 36 步，即 180 分钟
pred_offset = 6  # 设定预测偏移为 6 步，即提前 30 分钟预测

print(f"\n{'='*80}")  # 打印分隔线
print("GRU + 小波分解 + ECA - 所有特征组合测试")  # 打印脚本标题
print(f"{'='*80}")  # 打印分隔线

# 划分数据集
split_index = int(len(data) * 0.8)  # 按 8 比 2 计算训练集与测试集的划分下标
train_data_raw = data.iloc[:split_index].copy()  # 取前 80% 数据作为训练集
test_data_raw = data.iloc[split_index:].copy()  # 取后 20% 数据作为测试集

print(f"训练集: {len(train_data_raw)}, 测试集: {len(test_data_raw)}")  # 打印训练集与测试集样本数

# 小波分解
print("\n进行小波分解...")  # 提示开始小波分解
train_wavelet = wavelet_decompose_sliding(train_data_raw[features], target_col='温度偏差',  # 对训练集特征执行滑动窗口小波分解
                                         wavelet='db4', level=3, window_size=time_step)  # 对训练集分解时使用 db4 基、3 层、窗口大小等于时间步长
test_wavelet = wavelet_decompose_sliding(test_data_raw[features], target_col='温度偏差',  # 对测试集特征执行滑动窗口小波分解
                                        wavelet='db4', level=3, window_size=time_step)  # 对测试集分解时使用相同的小波参数，保证特征口径一致

# 定义特征组
other_features = [f for f in features if f != '温度偏差']  # 收集除温度偏差以外的原始特征
wavelet_features = ['温度偏差_approx', '温度偏差_detail1', '温度偏差_detail2', '温度偏差_detail3']  # 定义温度偏差的四个小波特征名

print(f"\n其他特征: {other_features}")  # 打印其他原始特征列表
print(f"小波特征: {wavelet_features}")  # 打印小波特征列表

# 生成所有可能的小波特征组合（从0个到4个）
all_combinations = []  # 初始化组合列表

# 组合0: 仅原始特征
all_combinations.append({  # 加入基线组合：仅使用原始特征、无小波特征
    'name': '仅原始特征',  # 组合名称：仅使用原始特征的基线
    'wavelet_subset': [],  # 该组合所选的小波特征子集，空列表表示无小波特征
    'description': '基线模型（无小波）'  # 该组合的文字说明
})  # 结束基线组合字典的构造

# 组合1-15: 小波特征的所有子集
for r in range(1, len(wavelet_features) + 1):  # 枚举选取 1 到 4 个小波特征
    for combo in combinations(wavelet_features, r):  # 生成指定数量下的所有小波特征子集
        all_combinations.append({  # 将该子集合为一种特征组合记录下来
            'name': f"原始+{list(combo)}",  # 组合名称：原始特征加选定的小波特征
            'wavelet_subset': list(combo),  # 该组合实际包含的小波特征子集
            'description': f"添加{len(combo)}个小波特征"  # 组合说明：新增的小波特征数量
        })  # 结束该组合字典的构造

print(f"\n总共测试 {len(all_combinations)} 种组合")  # 打印需要测试的组合总数
print(f"  - 0个小波特征: 1种")  # 打印含 0 个小波特征的组合数
print(f"  - 1个小波特征: 4种")  # 打印含 1 个小波特征的组合数
print(f"  - 2个小波特征: 6种")  # 打印含 2 个小波特征的组合数
print(f"  - 3个小波特征: 4种")  # 打印含 3 个小波特征的组合数
print(f"  - 4个小波特征: 1种")  # 打印含 4 个小波特征的组合数

# 存储所有结果
all_results = []  # 初始化所有实验结果的列表

print(f"\n{'='*80}")  # 打印分隔线
print("开始测试所有组合（含5折交叉验证）...")  # 提示开始测试所有组合（含交叉验证）
print(f"{'='*80}\n")  # 打印分隔线

for idx, combo in enumerate(all_combinations, 1):  # 遍历每一种特征组合
    print(f"\n[{idx}/{len(all_combinations)}] 测试: {combo['name']}")  # 打印当前组合的测试进度
    
    # 选择特征
    if len(combo['wavelet_subset']) == 0:  # 若组合不含任何小波特征
        # 仅原始特征
        selected_features = other_features + ['温度偏差']  # 基线组合使用全部原始特征
        train_features_selected = train_data_raw[selected_features]  # 从原始训练数据取出所选特征
        test_features_selected = test_data_raw[selected_features]  # 从原始测试数据取出所选特征
    else:  # 组合含有小波特征时
        # 原始特征 + 选定的小波特征
        selected_features = other_features + combo['wavelet_subset']  # 所选特征为其他原始特征加选定的小波特征
        train_features_selected = train_wavelet[selected_features]  # 从小波分解结果中取出训练集所选特征
        test_features_selected = test_wavelet[selected_features]  # 从小波分解结果中取出测试集所选特征
    
    print(f"  特征数: {len(selected_features)}")  # 打印当前组合的特征数量
    
    try:  # 异常保护：单个组合失败不影响整体测试
        # 归一化
        feature_scaler = MinMaxScaler()  # 创建特征归一化器
        target_scaler = MinMaxScaler()  # 创建目标归一化器
        
        feature_scaler.fit(train_features_selected)  # 只在训练特征上拟合特征归一化器
        target_scaler.fit(train_data_raw[[target]])  # 只在训练目标上拟合目标归一化器
        
        train_features_norm = pd.DataFrame(  # 对训练特征执行归一化并重建为 DataFrame
            feature_scaler.transform(train_features_selected),  # 用特征归一化器变换训练特征
            columns=selected_features  # 为归一化结果指定与原特征一致的列名
        )  # 结束训练特征的归一化
        test_features_norm = pd.DataFrame(  # 对测试特征执行归一化并重建为 DataFrame
            feature_scaler.transform(test_features_selected),  # 用特征归一化器变换测试特征
            columns=selected_features  # 为归一化结果指定与原特征一致的列名
        )  # 结束测试特征的归一化
        
        train_target_norm = target_scaler.transform(train_data_raw[[target]]).flatten()  # 归一化训练目标并展平为一维数组
        test_target_norm = target_scaler.transform(test_data_raw[[target]]).flatten()  # 归一化测试目标并展平为一维数组
        
        # 创建序列
        X_train = create_sequences(train_features_norm, time_step, pred_offset)  # 由归一化训练特征构造时序样本
        y_train = train_target_norm[time_step + pred_offset:][:len(X_train)]  # 按预测偏移对齐训练目标并截断到样本数
        
        X_test = create_sequences(test_features_norm, time_step, pred_offset)  # 由归一化测试特征构造时序样本
        y_test = test_target_norm[time_step + pred_offset:][:len(X_test)]  # 按预测偏移对齐测试目标并截断到样本数
        
        # ===================== 5折交叉验证 =====================
        print(f"  开始5折交叉验证...")  # 提示开始 5 折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 定义 5 折打乱划分，固定随机种子保证可复现
        cv_val_losses = []  # 记录每一折的最佳验证损失
        cv_training_times = []  # 记录每一折的训练耗时
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):  # 遍历交叉验证的每一折
            X_train_fold = X_train[train_idx]  # 取出本折的训练特征
            y_train_fold = y_train[train_idx]  # 取出本折的训练目标
            X_val_fold = X_train[val_idx]  # 取出本折的验证特征
            y_val_fold = y_train[val_idx]  # 取出本折的验证目标
            
            # 转换为Tensor
            X_train_fold_tensor = torch.FloatTensor(X_train_fold)  # 将本折训练特征转换为浮点张量
            y_train_fold_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)  # 将本折训练目标转换为张量并增加特征维度
            X_val_fold_tensor = torch.FloatTensor(X_val_fold)  # 将本折验证特征转换为浮点张量
            y_val_fold_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)  # 将本折验证目标转换为张量并增加特征维度
            
            train_fold_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)  # 封装本折训练集为 TensorDataset
            val_fold_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)  # 封装本折验证集为 TensorDataset
            train_fold_loader = DataLoader(train_fold_dataset, batch_size=32, shuffle=True)  # 构建本折训练数据加载器并打乱
            val_fold_loader = DataLoader(val_fold_dataset, batch_size=32, shuffle=False)  # 构建本折验证数据加载器不打乱
            
            # 构建模型
            model_fold = GRUECAModel(len(selected_features), time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)  # 构建本折 GRU+ECA 模型并部署到设备
            
            # 训练
            fold_start_time = time.time()  # 记录本折训练开始时间
            _, fold_val_losses = train_model(model_fold, train_fold_loader, val_fold_loader,  # 训练本折模型并获取验证损失曲线
                                            epochs=200, lr=0.001, patience=10)  # 训练参数：最大轮数 200、学习率 0.001、早停耐心 10
            fold_training_time = time.time() - fold_start_time  # 计算本折训练耗时
            
            best_val_loss = min(fold_val_losses)  # 取本折最小的验证损失作为该折最优结果
            cv_val_losses.append(best_val_loss)  # 记录本折最佳验证损失
            cv_training_times.append(fold_training_time)  # 记录本折训练耗时
        
        # 计算交叉验证统计
        cv_mean_loss = np.mean(cv_val_losses)  # 计算交叉验证平均验证损失
        cv_std_loss = np.std(cv_val_losses)  # 计算交叉验证验证损失标准差
        cv_total_time = np.sum(cv_training_times)  # 计算交叉验证累计训练时长
        
        print(f"  5折交叉验证完成 - 平均验证损失: {cv_mean_loss:.6f} ± {cv_std_loss:.6f}")  # 打印交叉验证平均损失与标准差
        
        # ===================== 训练最终模型（使用全部训练数据） =====================
        print(f"  训练最终模型...")  # 提示开始训练最终模型
        X_train_tensor = torch.FloatTensor(X_train)  # 将全量训练特征转换为浮点张量
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 将全量训练目标转换为张量并增加维度
        X_test_tensor = torch.FloatTensor(X_test)  # 将测试特征转换为浮点张量
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 将测试目标转换为张量并增加维度
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 封装最终训练集为 TensorDataset
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 封装测试集为 TensorDataset
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构建最终训练数据加载器
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 构建测试数据加载器
        
        # 构建并训练最终模型
        final_model = GRUECAModel(len(selected_features), time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)  # 构建最终 GRU+ECA 模型并部署到设备
        
        final_start_time = time.time()  # 记录最终模型训练开始时间
        train_losses, val_losses = train_model(final_model, train_loader, test_loader,  # 在全部训练数据上训练最终模型（以测试集作早停验证）
                                              epochs=200, lr=0.001, patience=10)  # 最终模型训练参数与交叉验证保持一致
        final_training_time = time.time() - final_start_time  # 计算最终模型训练耗时
        
        total_training_time = cv_total_time + final_training_time  # 汇总交叉验证与最终模型的总训练时长
        
        # ===================== 测试集预测 =====================
        final_model.eval()  # 切换最终模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            y_pred = final_model(X_test_tensor.to(device)).cpu().numpy()  # 在测试集上前向预测并将结果取回为 numpy 数组
        
        # 反归一化
        y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 将归一化的真实目标反归一化还原为原始量纲
        y_pred_rescaled = target_scaler.inverse_transform(y_pred).flatten()  # 将归一化预测值反归一化还原为原始量纲
        
        # 计算指标
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))  # 计算均方根误差 RMSE
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE
        
        print(f"  测试集结果 - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")  # 打印测试集评估指标
        print(f"  总训练时长: {total_training_time:.1f}s (交叉验证: {cv_total_time:.1f}s + 最终模型: {final_training_time:.1f}s)")  # 打印总训练时长及其组成
        
        # 保存结果
        all_results.append({  # 把当前组合的成功结果记录为字典
            'combo_id': idx,  # 记录组合编号
            'name': combo['name'],  # 记录组合名称
            'description': combo['description'],  # 记录组合说明
            'num_wavelet_features': len(combo['wavelet_subset']),  # 记录组合包含的小波特征数量
            'wavelet_features': ','.join(combo['wavelet_subset']) if combo['wavelet_subset'] else 'None',  # 记录组合所用小波特征名（逗号分隔，无则记为 None）
            'total_features': len(selected_features),  # 记录组合的总特征数
            'cv_mean_val_loss': cv_mean_loss,  # 记录 5 折交叉验证的平均验证损失
            'cv_std_val_loss': cv_std_loss,  # 记录 5 折交叉验证的验证损失标准差
            'cv_training_time': cv_total_time,  # 记录交叉验证阶段的总训练耗时
            'final_training_time': final_training_time,  # 记录最终模型的训练耗时
            'total_training_time': total_training_time,  # 记录交叉验证与最终模型的合计训练时长
            'r2': r2,  # 记录测试集决定系数 R2
            'rmse': rmse,  # 记录测试集均方根误差 RMSE
            'mae': mae,  # 记录测试集平均绝对误差 MAE
            'num_epochs': len(train_losses),  # 记录最终模型实际训练的轮数
            'final_train_loss': train_losses[-1],  # 记录最终模型最后一轮的训练损失
            'final_val_loss': val_losses[-1],  # 记录最终模型最后一轮的验证损失
            'status': 'success'  # 记录组合状态为成功
        })  # 结束成功结果字典的构造
        
    except Exception as e:  # 捕获当前组合训练过程中的异常
        print(f"  错误: {str(e)}")  # 打印当前组合的错误信息
        all_results.append({  # 记录失败组合，各指标置为 NaN
            'combo_id': idx,  # 记录失败组合的编号
            'name': combo['name'],  # 记录失败组合的名称
            'description': combo['description'],  # 记录失败组合的说明
            'num_wavelet_features': len(combo['wavelet_subset']),  # 记录失败组合的小波特征数量
            'wavelet_features': ','.join(combo['wavelet_subset']) if combo['wavelet_subset'] else 'None',  # 记录失败组合所用小波特征名
            'total_features': len(selected_features),  # 记录失败组合的总特征数
            'cv_mean_val_loss': np.nan,  # 交叉验证平均验证损失记为 NaN
            'cv_std_val_loss': np.nan,  # 交叉验证验证损失标准差记为 NaN
            'cv_training_time': np.nan,  # 交叉验证训练耗时记为 NaN
            'final_training_time': np.nan,  # 最终模型训练耗时记为 NaN
            'total_training_time': np.nan,  # 总训练时长记为 NaN
            'r2': np.nan,  # 决定系数 R2 记为 NaN
            'rmse': np.nan,  # 均方根误差 RMSE 记为 NaN
            'mae': np.nan,  # 平均绝对误差 MAE 记为 NaN
            'num_epochs': np.nan,  # 训练轮数记为 NaN
            'final_train_loss': np.nan,  # 最终训练损失记为 NaN
            'final_val_loss': np.nan,  # 最终验证损失记为 NaN
            'status': f'failed: {str(e)}'  # 记录失败状态与具体错误信息
        })  # 结束失败结果字典的构造

# ===================== 保存结果 =====================
print(f"\n{'='*80}")  # 打印分隔线
print("保存结果...")  # 提示正在保存结果
print(f"{'='*80}")  # 打印分隔线

results_df = pd.DataFrame(all_results)  # 将结果列表转换为 DataFrame
results_df.to_csv('all_wavelet_eca_combinations_results.csv', index=False, encoding='utf-8-sig')  # 保存完整结果到 CSV（带 BOM 便于 Excel 打开）
print("完整结果已保存: all_wavelet_eca_combinations_results.csv")  # 打印完整结果保存路径

# 只显示成功的结果
success_results = results_df[results_df['status'] == 'success'].copy()  # 仅筛选出训练成功的组合
success_results = success_results.sort_values('r2', ascending=False)  # 按 R2 从高到低排序成功组合

print(f"\n成功测试: {len(success_results)}/{len(all_results)} 种组合")  # 打印成功测试的组合数量
print(f"\n前10名最佳组合 (按R2排序):")  # 打印前 10 名组合的标题
print(success_results[['combo_id', 'num_wavelet_features', 'cv_mean_val_loss', 'cv_std_val_loss', 'r2', 'rmse', 'mae', 'total_training_time']].head(10).to_string(index=False))  # 打印前 10 名组合的关键指标明细

# 保存排序后的结果
success_results.to_csv('all_wavelet_eca_combinations_results_sorted.csv', index=False, encoding='utf-8-sig')  # 将排序后的成功结果保存到 CSV
print("\n排序结果已保存: all_wavelet_eca_combinations_results_sorted.csv")  # 打印排序结果保存路径

# ===================== 可视化 =====================
print("\n生成可视化图表...")  # 提示开始生成可视化图表

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 创建 2 行 3 列、尺寸为 18×10 的子图画布

# 按小波特征数量分组
grouped = success_results.groupby('num_wavelet_features')  # 按小波特征数量对成功结果分组

# 1. R2 箱线图
ax = axes[0, 0]  # 选取第一个子图用于绘制 R2 箱线图
data_to_plot = [grouped.get_group(i)['r2'].values for i in sorted(grouped.groups.keys())]  # 收集各分组的 R2 数据用于绘图
ax.boxplot(data_to_plot, labels=[f'{i}个' for i in sorted(grouped.groups.keys())])  # 绘制 R2 分布箱线图
ax.set_xlabel('小波特征数量')  # 设置 R2 箱线图横轴标签
ax.set_ylabel('R2')  # 设置 R2 箱线图纵轴标签
ax.set_title('R2分布（按小波特征数量）')  # 设置 R2 箱线图标题
ax.grid(True, alpha=0.3)  # 为 R2 箱线图添加半透明网格

# 2. RMSE 箱线图
ax = axes[0, 1]  # 选取第二个子图用于绘制 RMSE 箱线图
data_to_plot = [grouped.get_group(i)['rmse'].values for i in sorted(grouped.groups.keys())]  # 收集各分组的 RMSE 数据用于绘图
ax.boxplot(data_to_plot, labels=[f'{i}个' for i in sorted(grouped.groups.keys())])  # 绘制 RMSE 分布箱线图
ax.set_xlabel('小波特征数量')  # 设置 RMSE 箱线图横轴标签
ax.set_ylabel('RMSE')  # 设置 RMSE 箱线图纵轴标签
ax.set_title('RMSE分布（按小波特征数量）')  # 设置 RMSE 箱线图标题
ax.grid(True, alpha=0.3)  # 为 RMSE 箱线图添加半透明网格

# 3. MAE 箱线图
ax = axes[0, 2]  # 选取第三个子图用于绘制 MAE 箱线图
data_to_plot = [grouped.get_group(i)['mae'].values for i in sorted(grouped.groups.keys())]  # 收集各分组的 MAE 数据用于绘图
ax.boxplot(data_to_plot, labels=[f'{i}个' for i in sorted(grouped.groups.keys())])  # 绘制 MAE 分布箱线图
ax.set_xlabel('小波特征数量')  # 设置 MAE 箱线图横轴标签
ax.set_ylabel('MAE')  # 设置 MAE 箱线图纵轴标签
ax.set_title('MAE分布（按小波特征数量）')  # 设置 MAE 箱线图标题
ax.grid(True, alpha=0.3)  # 为 MAE 箱线图添加半透明网格

# 4. 训练时长对比
ax = axes[1, 0]  # 选取第四个子图用于绘制训练时长对比
avg_time = grouped['training_time'].mean()  # 计算各分组的平均训练时长（原代码使用 training_time 列）
ax.bar(range(len(avg_time)), avg_time.values, alpha=0.7, color='steelblue')  # 绘制各分组平均训练时长的柱状图
ax.set_xticks(range(len(avg_time)))  # 设置柱状图横轴刻度位置
ax.set_xticklabels([f'{i}个' for i in avg_time.index])  # 设置柱状图横轴刻度标签
ax.set_xlabel('小波特征数量')  # 设置训练时长图横轴标签
ax.set_ylabel('平均训练时长 (秒)')  # 设置训练时长图纵轴标签
ax.set_title('平均训练时长对比')  # 设置训练时长图标题
ax.grid(True, alpha=0.3, axis='y')  # 为训练时长图仅添加纵向网格

# 5. Top10组合对比
ax = axes[1, 1]  # 选取第五个子图用于绘制 Top10 组合对比
top10 = success_results.head(10)  # 取出 R2 排名前 10 的组合
x_pos = range(len(top10))  # 生成水平条形图的纵轴位置
ax.barh(x_pos, top10['r2'].values, alpha=0.7, color='green')  # 绘制 Top10 组合 R2 的水平条形图
ax.set_yticks(x_pos)  # 设置水平条形图纵轴刻度位置
ax.set_yticklabels([f"组合{int(id)}" for id in top10['combo_id'].values], fontsize=8)  # 设置水平条形图纵轴刻度标签为组合编号
ax.set_xlabel('R2')  # 设置水平条形图横轴标签
ax.set_title('Top 10 组合 R2对比')  # 设置 Top10 对比图标题
ax.grid(True, alpha=0.3, axis='x')  # 为 Top10 对比图仅添加横向网格
ax.invert_yaxis()  # 反转纵轴顺序使最佳组合显示在上方

# 6. 特征数量 vs R2 散点图
ax = axes[1, 2]  # 选取第六个子图用于绘制特征数与 R2 的散点图
scatter = ax.scatter(success_results['total_features'], success_results['r2'],   # 绘制总特征数与 R2 的散点，颜色映射小波特征数
                    c=success_results['num_wavelet_features'], cmap='viridis',   # 散点颜色映射小波特征数量，使用 viridis 色系
                    alpha=0.6, s=100)  # 设置散点透明度与大小
ax.set_xlabel('总特征数')  # 设置散点图横轴标签
ax.set_ylabel('R2')  # 设置散点图纵轴标签
ax.set_title('总特征数 vs R2')  # 设置散点图标题
ax.grid(True, alpha=0.3)  # 为散点图添加半透明网格
cbar = plt.colorbar(scatter, ax=ax)  # 为散点图添加颜色条
cbar.set_label('小波特征数')  # 设置颜色条标签为小波特征数

plt.tight_layout()  # 调整子图布局使之紧凑
plt.savefig('all_eca_combinations_visualization.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存可视化图表
print("可视化图表已保存: all_eca_combinations_visualization.png")  # 打印可视化图表保存路径

# 生成详细统计报告
print(f"\n{'='*80}")  # 打印分隔线
print("统计摘要")  # 打印统计摘要标题
print(f"{'='*80}")  # 打印分隔线
for num_wavelet in sorted(grouped.groups.keys()):  # 遍历每个小波特征数量分组
    group_data = grouped.get_group(num_wavelet)  # 取出当前分组的数据
    print(f"\n{num_wavelet}个小波特征 ({len(group_data)}种组合):")  # 打印当前分组的标题与组合数量
    print(f"  R2    - 平均: {group_data['r2'].mean():.4f}, 最佳: {group_data['r2'].max():.4f}, 最差: {group_data['r2'].min():.4f}")  # 打印该组 R2 的平均、最佳与最差值
    print(f"  RMSE  - 平均: {group_data['rmse'].mean():.4f}, 最佳: {group_data['rmse'].min():.4f}, 最差: {group_data['rmse'].max():.4f}")  # 打印该组 RMSE 的平均、最佳与最差值
    print(f"  MAE   - 平均: {group_data['mae'].mean():.4f}, 最佳: {group_data['mae'].min():.4f}, 最差: {group_data['mae'].max():.4f}")  # 打印该组 MAE 的平均、最佳与最差值
    print(f"  训练时长 - 平均: {group_data['training_time'].mean():.1f}秒")  # 打印该组平均训练时长（原代码使用 training_time 列）

print(f"\n{'='*80}")  # 打印分隔线
print("测试完成！")  # 打印测试完成提示
print(f"{'='*80}")  # 打印分隔线









