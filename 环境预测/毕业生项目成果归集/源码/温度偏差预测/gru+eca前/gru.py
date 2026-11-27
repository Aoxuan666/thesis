"""
GRU + ECA 通道注意力模型训练与评估脚本（温度偏差预测）。

本脚本完成温度偏差预测的完整实验流程：读取温湿度 CSV 数据并进行多编码兼容与缺失值处理，
按时间顺序划分训练/测试集并仅在训练集上拟合归一化器以避免数据泄漏，构造时间窗口序列样本，
构建基于 PyTorch 的 GRU + ECA（Efficient Channel Attention）模型并带早停地训练，
在测试集上评估 R²/RMSE/MAE 并绘图，保存模型与归一化器，最后加载新数据集进行预测与可视化。
"""
import pandas as pd  # 导入 pandas 数据处理库，简写为 pd，用于读写 CSV 与表格操作
import numpy as np  # 导入 numpy 数值计算库，简写为 np，用于数组与矩阵运算
# matplotlib后端配置：必须在导入pyplot之前设置（否则 Agg 后端设置可能不生效）
import matplotlib  # 导入 matplotlib 基础模块，用于在运行前配置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片（适合无图形界面的服务器环境）
import matplotlib.pyplot as plt  # 导入 pyplot 绘图模块，简写为 plt，用于绘制折线等图形
from sklearn.preprocessing import MinMaxScaler  # 导入最小-最大归一化器，把特征缩放到 [0,1] 区间
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入回归评估指标：决定系数 R²、均方误差 MSE、平均绝对误差 MAE
from sklearn.model_selection import train_test_split, KFold  # 导入数据划分与 K 折交叉验证工具（交叉验证部分已注释，当前未使用）
import copy  # 导入 copy 模块，用 deepcopy 保存早停过程中的最佳模型权重
import torch  # 导入 PyTorch 主库，用于张量计算、GPU 加速与模型训练
import torch.nn as nn  # 导入神经网络模块，用于搭建网络层（GRU、Linear 等）
import torch.optim as optim  # 导入优化器模块，这里使用 Adam 更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集基类、数据加载器与张量数据集封装工具
# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若 CUDA 可用则使用 GPU，否则回退到 CPU
print(f"使用设备: {device}")  # 打印当前选定的计算设备
if torch.cuda.is_available():  # 判断系统是否存在可用的 NVIDIA GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图默认中文字体为黑体，避免中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号显示为方块
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，按顺序尝试以兼容不同来源的 CSV
data = None  # 初始化数据占位变量为空，等待成功读取后赋值为 DataFrame
for enc in encodings:  # 遍历每一种候选编码
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 尝试用当前编码读取训练数据文件
        print(f"成功使用 {enc} 编码加载训练数据")  # 读取成功则打印实际使用的编码
        break  # 读取成功后跳出循环，不再尝试后续编码
    except (UnicodeDecodeError, UnicodeError):  # 若当前编码出现解码错误
        continue  # 跳过该编码，继续尝试下一种

if data is None:  # 若所有候选编码都读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 以 5 分钟为间隔生成与数据等长的时间戳列
features = ['日龄','温度偏差','外界温度','CO2','目标温度']  # 定义候选输入特征名称列表
target = '温度偏差'  # 指定预测目标列为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中真实存在的特征列
# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印由 60 个等号组成的分隔线
print("缺失值处理")  # 打印本阶段标题：缺失值处理
print(f"{'='*60}")  # 再次打印分隔线，形成标题上下边框
print(f"原始数据形状: {data.shape}")  # 打印原始数据的行数与列数
print(f"\n各列缺失值统计:")  # 打印各列缺失值统计的标题
for col in features + [target]:  # 遍历所有特征列以及目标列
    if col in data.columns:  # 仅处理数据中确实存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算该列缺失值占比（百分比）
        if nan_count > 0:  # 仅对存在缺失值的列进行输出
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印列名、缺失数量与占比

# 定义连续变量和离散变量（两类变量采用不同的缺失值填充策略）
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续变量（用线性插值+前后填充）
discrete_vars = ['日龄']  # 离散变量（仅用前后填充）

# 对连续变量使用插值+前后填充（线性插值适合时间序列的连续变化）
for col in continuous_vars:  # 遍历每个连续变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值填充，双向以覆盖首尾缺失
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].ffill()  # 前向填充，用前一个有效值填补开头缺失
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].bfill()  # 后向填充，用后一个有效值填补结尾缺失
        
        after_nan = data[col].isna().sum()  # 统计填充后剩余的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前的缺失数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后的缺失数量

# 对离散变量使用前后填充（离散值不适合插值）
for col in discrete_vars:  # 遍历每个离散变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 前向填充
        data[col] = data[col].ffill()  # 用前一个有效值填补离散变量缺失值
        # 后向填充
        data[col] = data[col].bfill()  # 用后一个有效值填补离散变量缺失值
        
        after_nan = data[col].isna().sum()  # 统计填充后剩余的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前的缺失数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后的缺失数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征与目标列中剩余缺失值的总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除在特征或目标上仍含缺失值的样本行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理之后的数据规模
print(f"{'='*60}\n")  # 打印分隔线并换行，结束本阶段

# 保存原始数据副本（用于后续测试集预测）
data_original = data.copy()  # 深拷贝一份处理后的数据备用（当前脚本后续未再使用）

print(f"\n【GRU + ECA注意力模型】")  # 打印模型标题
print(f"使用特征: {features}")  # 打印实际参与建模的特征列表
print(f"特征数量: {len(features)}")  # 打印特征个数

# ===================== 数据集划分 =====================
# 定义时间窗口参数
time_step = 36  # GRU时间窗口：36步 × 5分钟 = 3小时（输入历史长度）
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟（提前预测量）

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印输入窗口步数及对应分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移步数及对应分钟数

# 划分数据集
split_index = int(len(data) * 0.8)  # 取前 80% 位置作为按时间顺序的划分索引
train_data_raw = data.iloc[:split_index].copy()  # 前 80% 样本作为训练集
test_data_raw = data.iloc[split_index:].copy()  # 后 20% 样本作为测试集

print(f"\n数据集划分:")  # 打印数据集划分结果标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw)/len(data)*100:.1f}%)")  # 打印训练集样本数与占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw)/len(data)*100:.1f}%)")  # 打印测试集样本数与占比

# 使用原始特征
train_combined_features = train_data_raw[features].copy()  # 取训练集的特征列副本
test_combined_features = test_data_raw[features].copy()  # 取测试集的特征列副本
feature_names = features  # 记录最终使用的特征名列表

print(f"总特征数: {len(feature_names)}")  # 打印总特征数量

# ===================== 归一化（只在训练集上fit！） =====================
print("\n归一化处理（避免数据泄漏）...")  # 提示开始归一化，强调避免数据泄漏
feature_scaler = MinMaxScaler()  # 创建特征归一化器对象
target_scaler = MinMaxScaler()  # 创建目标值归一化器对象

# 只在训练集上fit
feature_scaler.fit(train_combined_features)  # 仅用训练集特征拟合特征归一化器
target_scaler.fit(train_data_raw[[target]])  # 仅用训练集目标拟合目标归一化器

# 分别transform训练集和测试集
train_combined_features = pd.DataFrame(  # 将归一化后的训练特征重新构建为 DataFrame
    feature_scaler.transform(train_combined_features),  # 用训练集归一化器变换训练特征
    columns=feature_names  # 指定 DataFrame 的列名为特征名
)
test_combined_features = pd.DataFrame(  # 将归一化后的测试特征重新构建为 DataFrame
    feature_scaler.transform(test_combined_features),  # 用同一个归一化器变换测试特征
    columns=feature_names  # 指定 DataFrame 的列名为特征名
)

train_data_raw[target] = target_scaler.transform(train_data_raw[[target]])  # 归一化训练集目标值
test_data_raw[target] = target_scaler.transform(test_data_raw[[target]])  # 归一化测试集目标值

# ===================== 序列数据构造（分别对训练集和测试集） =====================
# time_step 和 pred_offset 已在前面定义

def create_sequences(data, time_step):
    """
    将二维特征表转换为三维时间序列样本。

    Args:
        data: 归一化后的特征 DataFrame，每行代表一个时间步。
        time_step: 输入序列长度，即历史窗口的大小。
    说明：循环上界使用全局的 pred_offset 预留预测偏移，标签需在调用处按该偏移另行构造。
    Returns:
        np.ndarray: 形状为 (样本数, time_step, 特征数) 的序列数组。
    """
    sequences = []  # 用于收集每个时间窗口的样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点，预留 time_step 与 pred_offset
        sequences.append(data.iloc[i:i + time_step].values)  # 取连续 time_step 行作为一个输入样本
    return np.array(sequences)  # 将样本列表转换为 numpy 数组返回

# 分别为训练集和测试集创建序列（避免跨边界）
print("\n创建训练集序列...")  # 提示开始构造训练集序列
X_train = create_sequences(train_combined_features, time_step)  # 由归一化训练特征构造输入序列
y_train = train_data_raw[target].values[time_step + pred_offset:]  # 取偏移之后的归一化目标作为标签
y_train = y_train[:len(X_train)]  # 确保长度匹配：截断标签使其与样本数一致（原注释保留）

print("\n创建测试集序列...")  # 提示开始构造测试集序列
X_test = create_sequences(test_combined_features, time_step)  # 由归一化测试特征构造输入序列
y_test = test_data_raw[target].values[time_step + pred_offset:]  # 取偏移之后的归一化目标作为标签
y_test = y_test[:len(X_test)]  # 确保长度匹配：截断标签使其与样本数一致（原注释保留）

print(f"\n序列数据形状:")  # 打印序列形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集输入与标签形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集输入与标签形状


# ===================== PyTorch模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    通过1D卷积实现高效的通道注意力机制，避免降维带来的信息损失。
    Args:
        channels: 输入特征的通道数（此处等于特征列数）。
        gamma: 用于自适应计算卷积核大小的缩放参数。
        b: 用于自适应计算卷积核大小的偏置参数。
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 层：自适应确定 1D 卷积核大小并构建池化、卷积与激活模块。"""
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 根据通道数按公式计算卷积核尺度 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数（原注释保留）

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道间交互，不加偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活，把通道权重映射到 (0,1)

    def forward(self, x):
        """对输入执行通道注意力加权：全局池化→1D卷积→Sigmoid→按通道乘回。"""
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step) 以便进行池化
        x_permuted = x.permute(0, 2, 1)  # (batch, channels, time_step) 置换维度以便按通道池化

        # 全局平均池化: (batch, channels, 1)
        y = self.avg_pool(x_permuted)  # 每个通道在时间维上做全局平均，得到通道描述

        # 1D卷积: (batch, 1, channels) -> (batch, 1, channels)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 交换维度后用 1D 卷积建模通道相关性再换回

        # Sigmoid激活
        y = self.sigmoid(y)  # (batch, channels, 1) 得到各通道注意力权重

        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 把注意力权重广播乘回各通道特征

        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 还原为原始维度顺序后返回


class GRUECAModel(nn.Module):
    """
    GRU + ECA注意力机制
    结合ECA通道注意力和GRU的时序建模能力。
    Args:
        input_dim: 输入特征维度。
        time_step: 时间窗口长度（仅记录，不参与层构造）。
        gru_hidden: GRU 隐藏层维度。
        gru_layers: GRU 层数。
        dropout: 多层 GRU 时层间 dropout 比例；单层时自动置 0。
    输入: (batch, time_step, input_dim)；输出: (batch, 1)。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):
        """按输入维度构建 ECA 注意力、GRU 编码器与全连接输出层。"""
        super(GRUECAModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏维度

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 构造通道注意力层，通道数等于输入维度

        # 单层GRU编码器：提取时序特征
        self.gru = nn.GRU(  # 定义 GRU 层
            input_size=input_dim,  # 输入特征维度
            hidden_size=gru_hidden,  # 隐藏状态维度
            num_layers=gru_layers,  # GRU 堆叠层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 输入张量以 batch 为第一维
            bidirectional=False  # 使用单向 GRU
        )

        # 输出层：直接从GRU隐藏状态映射到预测值
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层把隐藏维度映射为单个预测值

    def forward(self, x):
        """前向传播：ECA 增强 → GRU 编码 → 取末时刻输出 → 全连接预测。"""
        # x: (batch, time_step, input_dim)
        # ECA注意力增强输入特征
        x = self.eca(x)  # (batch, time_step, input_dim) 对输入做通道注意力加权

        # GRU编码：提取时序特征
        out, h = self.gru(x)  # out: (batch, time_step, hidden), h: (layers, batch, hidden) 获得输出与隐藏态

        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden) 取序列最后一个时间步的输出

        # 预测输出
        return self.fc(out_last)  # (batch, 1) 经全连接层得到预测值


class EarlyStopping:
    """通用早停策略，默认用于最小化验证损失。"""
    def __init__(self, patience=10, min_delta=1e-4, mode='min', restore_best_weights=True, verbose=True):
        """
        Args:
            patience: 连续多少轮没有提升就停止。
            min_delta: 认为"有提升"所需的最小变化量。
            mode: 'min' 表示越小越好（如 loss），'max' 表示越大越好（如 R2）。
            restore_best_weights: 早停后是否恢复到最佳权重。
            verbose: 是否打印日志。
        """
        self.patience = patience  # 保存容忍轮数
        self.min_delta = min_delta  # 保存最小提升阈值
        self.mode = mode  # 保存优化方向模式
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重的开关
        self.verbose = verbose  # 保存是否打印日志的开关

        self.counter = 0  # 未提升轮数计数器
        self.best_score = None  # 记录迄今最佳指标值
        self.best_epoch = 0  # 记录最佳指标所在轮次
        self.early_stop = False  # 早停标志，初始为 False
        self.best_model = None  # 用于存放最佳模型状态字典

    def _is_improvement(self, score):
        """按 mode 判断当前分数相对最佳分数是否有足够提升；首个分数视为提升。"""
        if self.best_score is None:  # 若还没有基准分数
            return True  # 第一个分数直接视为提升

        if self.mode == 'min':  # 越小越好的模式
            return score < self.best_score - self.min_delta  # 当前分数需小于最佳减去阈值
        elif self.mode == 'max':  # 越大越好的模式
            return score > self.best_score + self.min_delta  # 当前分数需大于最佳加上阈值
        else:  # 其他非法模式
            raise ValueError("mode must be 'min' or 'max'")  # 抛出异常提示模式错误

    def __call__(self, score, model, epoch=None):
        """每轮用验证分数更新早停状态；非法分数按未提升处理，若提升则深拷贝保存最佳权重。"""
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 若分数为空或非有限值
            if self.verbose:  # 若开启日志
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印无效分数警告
            self.counter += 1  # 未提升计数器加一
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 提前返回，不更新最佳

        if self._is_improvement(score):  # 若判定为有提升
            self.best_score = score  # 更新最佳分数
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 更新或保留最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝保存当前最佳模型权重
            self.counter = 0  # 重置未提升计数器
            if self.verbose:  # 若开启日志
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印提升信息
        else:  # 若无提升
            self.counter += 1  # 未提升计数器加一
            if self.verbose:  # 若开启日志
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印计数器进度
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停

    def restore(self, model):
        """若开启恢复开关且已保存最佳权重，则把最佳权重加载回模型。"""
        if self.restore_best_weights and self.best_model is not None:  # 满足恢复条件
            model.load_state_dict(self.best_model)  # 将最佳权重载入模型
            if self.verbose:  # 若开启日志
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):
    """训练模型，使用 MSE 损失与 Adam 优化器，并按验证损失早停。"""
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 定义 Adam 优化器并绑定模型参数
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器（监控验证损失最小化）

    train_losses = []  # 记录每轮训练集平均损失
    val_losses = []  # 记录每轮验证集平均损失

    for epoch in range(epochs):  # 遍历训练轮次
        # 训练阶段
        model.train()  # 切换模型为训练模式（启用 dropout 等）
        train_loss = 0.0  # 初始化本轮训练损失累加器
        for batch_x, batch_y in train_loader:  # 遍历训练数据加载器的每个批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把批次数据搬到计算设备

            optimizer.zero_grad()  # 清空上一轮累积的梯度
            outputs = model(batch_x)  # 前向传播得到预测
            loss = criterion(outputs, batch_y)  # 计算预测与真实之间的 MSE 损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新模型参数

            train_loss += loss.item()  # 累加本批次损失值

        train_loss /= len(train_loader)  # 求本轮训练集平均损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换模型为评估模式（关闭 dropout 等）
        val_loss = 0.0  # 初始化本轮验证损失累加器
        with torch.no_grad():  # 关闭梯度计算以节省显存与加速
            for batch_x, batch_y in val_loader:  # 遍历验证数据加载器的每个批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把批次数据搬到计算设备
                outputs = model(batch_x)  # 前向传播得到预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加本批次验证损失

        val_loss /= len(val_loader)  # 求本轮验证集平均损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前轮次的训练与验证损失

        # 早停检查
        early_stopping(val_loss, model)  # 用本轮验证损失更新早停状态
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停发生的轮次
            model.load_state_dict(early_stopping.best_model)  # 恢复最佳权重
            break  # 跳出训练循环

    # 加载最佳模型
    if early_stopping.best_model is not None:  # 若保存过最佳权重
        model.load_state_dict(early_stopping.best_model)  # 训练结束后确保模型处于最佳权重

    return train_losses, val_losses  # 返回训练与验证损失曲线列表

# ===================== K折交叉验证训练（仅在训练集上） =====================
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(feature_names)  # 输入维度等于最终特征数量

# 以下为整段被注释掉的 5 折交叉验证实验代码，保留原文不改（原注释保留）
#
# print("\n========== 开始5折交叉验证（仅训练集） ==========")
# print(f"交叉验证数据量: {len(X_train)} 样本")
#
# for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
#     print(f"\n{'=' * 50}")
#     print(f"第 {fold}/5 折训练")
#     print(f"{'=' * 50}")
#
#     X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
#     y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
#
#     print(f"训练集样本数: {len(X_train_fold)}")
#     print(f"验证集样本数: {len(X_val_fold)}")
#
#     X_train_fold_tensor = torch.FloatTensor(X_train_fold)
#     y_train_fold_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)
#     X_val_fold_tensor = torch.FloatTensor(X_val_fold)
#     y_val_fold_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)
#
#     fold_train_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
#     fold_val_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)
#     fold_train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True)
#     fold_val_loader = DataLoader(fold_val_dataset, batch_size=32, shuffle=False)
#
#     fold_model = GRUECAModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)
#
#     if fold == 1:
#         print(f"模型参数量: {sum(p.numel() for p in fold_model.parameters()):,}")
#
#     fold_train_losses, fold_val_losses = train_model(fold_model, fold_train_loader, fold_val_loader,
#                                                      epochs=200, lr=0.001, patience=10)
#     history_list.append({
#         'train_losses': fold_train_losses,
#         'val_losses': fold_val_losses
#     })
#
#     best_val_loss = min(fold_val_losses)
#     val_losses_all.append(best_val_loss)
#     print(f"\n第 {fold} 折最佳验证损失: {best_val_loss:.6f}")
#
# print(f"\n{'=' * 50}")
# print("5折交叉验证结果汇总")
# print(f"{'=' * 50}")
# for i, loss in enumerate(val_losses_all, 1):
#     print(f"第 {i} 折验证损失: {loss:.6f}")
# print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")
# print(f"验证损失标准差: {np.std(val_losses_all):.6f}")
#
# # ===================== 绘制K折交叉验证训练过程曲线 =====================
# plt.figure(figsize=(15, 10))
#
# plt.subplot(2, 2, 1)
# for i, history in enumerate(history_list, 1):
#     plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)
# plt.title('5-Fold Cross Validation - Training Loss', fontsize=14)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 2)
# for i, history in enumerate(history_list, 1):
#     plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)
# plt.title('5-Fold Cross Validation - Validation Loss', fontsize=14)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 3)
# folds = [f'Fold {i}' for i in range(1, 6)]
# plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)
# plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',
#             label=f'Mean: {np.mean(val_losses_all):.6f}')
# plt.title('Best Validation Loss per Fold', fontsize=14)
# plt.xlabel('Fold')
# plt.ylabel('Validation Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3, axis='y')
#
# plt.subplot(2, 2, 4)
# max_epochs = max(len(h['train_losses']) for h in history_list)
# avg_train_losses = []
# avg_val_losses = []
# for epoch in range(max_epochs):
#     train_vals = [h['train_losses'][epoch] for h in history_list if epoch < len(h['train_losses'])]
#     val_vals = [h['val_losses'][epoch] for h in history_list if epoch < len(h['val_losses'])]
#     if train_vals:
#         avg_train_losses.append(np.mean(train_vals))
#     if val_vals:
#         avg_val_losses.append(np.mean(val_vals))
# plt.plot(avg_train_losses, label='Avg Train Loss', linewidth=2)
# plt.plot(avg_val_losses, label='Avg Val Loss', linewidth=2)
# plt.title('5-Fold Average Loss Curves', fontsize=14)
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('kfold_training_GRUECA_pytorch.png', dpi=600, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRUECA_pytorch.png")

# ===================== 训练最终模型 =====================
input_dim = len(feature_names)  # 使用特征总数作为模型输入维度（原注释保留）

print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印训练最终模型标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练样本数量
print(f"测试集样本数: {len(X_test)}")  # 打印测试样本数量

X_train_tensor = torch.FloatTensor(X_train)  # 把训练序列转为 float 张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 把训练标签转为 float 张量并增加列维
X_test_tensor = torch.FloatTensor(X_test)  # 把测试序列转为 float 张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 把测试标签转为 float 张量并增加列维

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 取训练集前 80% 的划分索引
X_train_split = X_train_tensor[:train_val_split]  # 前 80% 作为实际训练输入
y_train_split = y_train_tensor[:train_val_split]  # 前 80% 作为实际训练标签
X_val_split = X_train_tensor[train_val_split:]  # 后 20% 作为验证输入
y_val_split = y_train_tensor[train_val_split:]  # 后 20% 作为验证标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 封装训练张量数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 封装验证张量数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 封装测试张量数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练加载器，打乱顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证加载器，不打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试加载器，不打乱

final_model = GRUECAModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)  # 实例化最终模型并搬到设备

print(final_model)  # 打印模型结构

print("\n开始训练最终模型...")  # 提示开始训练
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印可训练参数总数
print(f"训练集样本数: {len(X_train_split)}")  # 打印实际训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,
                                       epochs=200, lr=0.001, patience=10)  # 调用训练函数，返回两条损失曲线

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练结束时的最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12x6 英寸画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置图表标题
plt.xlabel('训练轮数 (Epoch)')  # 设置横轴标签
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示半透明网格线
plt.tight_layout()  # 自动调整布局防止裁切
plt.savefig('training_loss_GRUECA_pytorch.png', dpi=600, bbox_inches='tight')  # 保存损失曲线为图片
print("训练损失曲线已保存: training_loss_GRUECA_pytorch.png")  # 打印保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

# 直接使用y_test（已经是归一化后的值）
y_test_normalized = y_test  # 测试标签已是归一化值，直接引用

# 反归一化得到真实值
y_test_rescaled = target_scaler.inverse_transform(y_test_normalized.reshape(-1, 1))  # 将归一化标签还原为真实量纲

print(f"\n测试集数据检查:")  # 打印测试集数据检查标题
print(f"X_test shape: {X_test.shape}")  # 打印测试输入形状
print(f"y_test shape: {y_test.shape}")  # 打印测试标签形状
print(f"y_test_rescaled shape: {y_test_rescaled.shape}")  # 打印反归一化后标签形状

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试序列转为张量并搬到设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 前向预测并把结果搬回 CPU 转为 numpy

# 反归一化预测值
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将归一化预测还原为真实量纲

print(f"y_pred_rescaled shape: {y_pred_rescaled.shape}")  # 打印反归一化后预测形状

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse_test = np.sqrt(mse_test)  # 由 MSE 开方得到 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果（GRU+ECA模型）")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R^2:  {r2:.4f}")  # 打印测试集 R²
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印测试集 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布
# 确保长度匹配
y_test_rescaled_flat = y_test_rescaled.flatten()  # 将真实值展平为一维
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 将预测值展平为一维

# 从test_data_raw获取时间戳（已经是测试集部分）
test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset - 1:]  # 按偏移截取对应时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳使其与绘图数据等长

print(f"\n绘图数据检查:")  # 打印绘图数据检查标题
print(f"test_timestamps length: {len(test_timestamps)}")  # 打印时间戳长度
print(f"y_test_rescaled_flat length: {len(y_test_rescaled_flat)}")  # 打印真实值长度
print(f"y_pred_rescaled_flat length: {len(y_pred_rescaled_flat)}")  # 打印预测值长度

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实值曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+ECA预测', linewidth=2)  # 绘制预测值曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签
plt.title(f'测试集温度偏差结果（GRU+ECA） (R²: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置含指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.savefig('test_prediction_GRUECA_pytorch.png', dpi=600, bbox_inches='tight')  # 保存测试集预测图
print("测试集预测结果已保存: test_prediction_GRUECA_pytorch.png")  # 打印保存提示

# ===================== 模型保存 =====================
torch.save({  # 将模型与相关配置打包保存为检查点
    'model_state_dict': final_model.state_dict(),  # 保存模型权重状态字典
    'input_dim': input_dim,  # 保存输入维度
    'time_step': time_step,  # 保存时间窗口长度
    'feature_names': feature_names,  # 保存特征名列表
    'feature_scaler': feature_scaler,  # 保存特征归一化器
    'target_scaler': target_scaler,  # 保存目标归一化器
}, 'Model_GRUECA_pytorch.pth')  # 指定检查点文件路径
print("模型已保存: Model_GRUECA_pytorch.pth")  # 打印模型保存提示
# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始处理新数据集
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取新数据 CSV（默认编码）
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 由索引按 5 分钟间隔生成时间戳

# 使用原始特征（不进行小波分解）
new_combined_features = new_data[features].copy()  # 取新数据的特征列（原注释保留）

# 使用训练集的scaler进行归一化
new_combined_features = pd.DataFrame(  # 将归一化后的新特征重建为 DataFrame
    feature_scaler.transform(new_combined_features),  # 用训练集归一化器变换新特征
    columns=feature_names  # 指定列名为特征名
)

# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据未归一化的真实目标值

X_new = create_sequences(new_combined_features, time_step)  # 由新数据构造输入序列
y_new = y_new_original[time_step + pred_offset - 1:]  # 按偏移截取对应真实目标值
X_new = X_new[:len(y_new)]  # 截断序列使样本数与标签数一致

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 把新序列转为张量并搬到设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据预测并搬回 CPU 转 numpy

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 反归一化新数据预测值
y_new_rescaled = y_new.reshape(-1, 1)  # 原始值：把真实目标重塑为列向量（原注释保留）

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MSE
rmse_new = np.sqrt(mse_new)  # 计算新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果（GRU+ECA模型）")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R²:   {r2_new:.4f}")  # 打印新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset - 1:]  # 按偏移截取新数据时间戳
plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布
# 确保长度匹配
y_new_rescaled_flat = y_new_rescaled.flatten()  # 展平新数据真实值
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 展平新数据预测值
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制新数据真实曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='GRU+ECA预测', linewidth=2)  # 绘制新数据预测曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签
plt.title(f'新数据集温度偏差结果（GRU+ECA） (R²: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置含指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.savefig('new_data_prediction_GRUECA_pytorch.png', dpi=600, bbox_inches='tight')  # 保存新数据预测图
print("新数据集预测结果已保存: new_data_prediction_GRUECA_pytorch.png")  # 打印保存提示
print("\n========== 所有任务完成（GRU+ECA模型） ==========")  # 打印全部任务完成标志
