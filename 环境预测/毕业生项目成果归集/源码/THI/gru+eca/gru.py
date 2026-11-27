"""
GRU + ECA 通道注意力模型训练与评估脚本（THI 温湿度指数预测）

本脚本实现基于 PyTorch 的「GRU + ECA-Net 高效通道注意力」时序预测的完整流程：
1. 尝试多种编码读取温湿度 CSV 训练数据；
2. 按连续变量（插值+前后填充）与离散变量（前后填充）分别处理缺失值；
3. 按 6:4 划分训练/测试集，仅在训练集上拟合归一化器以避免数据泄漏；
4. 按滑动时间窗口构造序列样本（输入为历史窗口，标签为若干步之后的 THI）；
5. 定义 ECA 注意力层与 GRU 预测模型，从训练集中切出 20% 作为验证集进行带早停的训练；
6. 绘制损失曲线，在测试集与新数据上评估 R²/RMSE/MAE；
7. 保存模型检查点（含归一化器）与各类可视化图片。

数据输入: 温湿度数据.csv（训练/测试）、部分温湿度数据.csv（新数据）
产物输出: training_loss/test_prediction/new_data_prediction 等 PNG 图片、Model_GRUECA_pytorch.pth
"""
import pandas as pd  # 导入 pandas 库，用于 CSV 读取与 DataFrame 数据处理
import numpy as np  # 导入 numpy 库，用于数组与数值计算
import random  # 导入 random 库（随机种子设置当前已注释，保留备用）
import os  # 导入 os 库（环境变量设置当前已注释，保留备用）
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于提前指定绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片，无需图形界面
import matplotlib.pyplot as plt  # 导入 pyplot 子模块，提供绘图接口
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标的归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入回归评估指标：R²、MSE、MAE
from sklearn.model_selection import train_test_split, KFold  # 导入数据集划分与 K 折交叉验证工具（当前用手写划分）
import copy  # 导入 copy 库，用于深拷贝模型权重（早停时恢复最佳权重）
import torch  # 导入 PyTorch 主库，提供张量计算与设备（CPU/GPU）支持
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU、全连接层等
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集与数据加载器，用于批处理训练
# def set_seed(seed=10):
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(55)
# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU（可用时）或 CPU 作为训练设备
print(f"使用设备: {device}")  # 打印本次运行使用的计算设备
if torch.cuda.is_available():  # 若检测到可用 GPU，进一步打印其详细信息
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文标题正常显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号，避免坐标轴负号显示为方块
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，逐一尝试直到成功读取
data = None  # 初始化数据变量为空，稍后由成功读取的编码赋值
for enc in encodings:  # 遍历每种候选编码尝试读取 CSV
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据文件
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码，便于确认
        break  # 读取成功后跳出循环，不再尝试后续编码
    except (UnicodeDecodeError, UnicodeError):  # 若出现解码错误则尝试下一种编码
        continue  # 继续尝试下一个候选编码

if data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔为每行数据构造时间戳
features = ['日龄', 'THI', '当前温度', '目标温度']  # 预定义的候选特征列名称
target = 'THI'  # 指定预测目标为 THI（温湿度指数）
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列，避免缺列报错
# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印分隔线，标识缺失值处理阶段开始
print("缺失值处理")  # 打印阶段标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前数据的行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计表标题
for col in features + [target]:  # 遍历所有特征列与目标列
    if col in data.columns:  # 仅统计数据中存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比（百分比）
        if nan_count > 0:  # 仅打印确实存在缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与占比

# 定义连续变量和离散变量
continuous_vars = ['THI','目标温度', '当前温度']  # 连续型变量列表，采用插值方式填充
discrete_vars = ['日龄']  # 离散型变量列表，采用前后向填充


# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值填充连续变量的缺失值（双向）
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].ffill()  # 用前一个有效值填充剩余缺失（主要填补开头）
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].bfill()  # 用后一个有效值填充剩余缺失（主要填补结尾）
        
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        
        # 前向填充
        data[col] = data[col].ffill()  # 用前一个有效值填充离散变量缺失值
        # 后向填充
        data[col] = data[col].bfill()  # 用后一个有效值填充开头残留的缺失值
        
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计所有特征与目标列残留的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 提示将删除含缺失值的行
    data = data.dropna(subset=features + [target])  # 删除特征或目标中含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据规模
print(f"{'='*60}\n")  # 打印分隔线，标识缺失值处理阶段结束

# 保存原始数据副本（用于后续测试集预测）
data_original = data.copy()  # 深拷贝一份完整数据备份（当前脚本后续未直接使用，保留备用）

print(f"\n【GRU + ECA注意力模型】")  # 打印所用模型名称
print(f"使用特征: {features}")  # 打印最终使用的特征列表
print(f"特征数量: {len(features)}")  # 打印特征总数

# ===================== 数据集划分 =====================
# 定义时间窗口参数
time_step = 36  # GRU时间窗口：36步 × 5分钟 = 3小时
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印输入窗口长度及其对应分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移及其对应分钟数

# 划分数据集
split_index = int(len(data) * 0.6)  # 按 60% 位置计算训练集与测试集的切分索引
train_data_raw = data.iloc[:split_index].copy()  # 取前 60% 作为原始训练数据副本
test_data_raw = data.iloc[split_index:].copy()  # 取后 40% 作为原始测试数据副本

print(f"\n数据集划分:")  # 打印数据集划分结果标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw)/len(data)*100:.1f}%)")  # 打印训练集样本数及占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw)/len(data)*100:.1f}%)")  # 打印测试集样本数及占比

# 使用原始特征
train_combined_features = train_data_raw[features].copy()  # 抽取训练集的特征列副本
test_combined_features = test_data_raw[features].copy()  # 抽取测试集的特征列副本
feature_names = features  # 记录特征名称列表，供后续建模与保存使用

print(f"总特征数: {len(feature_names)}")  # 打印特征总维度

# ===================== 归一化（只在训练集上fit！） =====================
print("\n归一化处理（避免数据泄漏）...")  # 打印归一化阶段提示，强调避免数据泄漏
feature_scaler = MinMaxScaler()  # 创建特征归一化器（缩放到 0~1）
target_scaler = MinMaxScaler()  # 创建目标归一化器（缩放到 0~1）

# 只在训练集上fit
feature_scaler.fit(train_combined_features)  # 仅用训练特征拟合归一化器，避免测试信息泄漏
target_scaler.fit(train_data_raw[[target]])  # 仅用训练目标拟合目标归一化器

# 分别transform训练集和测试集
train_combined_features = pd.DataFrame(  # 将归一化后的训练特征重建为 DataFrame
    feature_scaler.transform(train_combined_features),  # 用训练集拟合的归一化器缩放训练特征
    columns=feature_names  # 保持原有特征列名
)
test_combined_features = pd.DataFrame(  # 将归一化后的测试特征重建为 DataFrame
    feature_scaler.transform(test_combined_features),  # 用同一归一化器缩放测试特征
    columns=feature_names  # 保持原有特征列名
)

train_data_raw[target] = target_scaler.transform(train_data_raw[[target]])  # 将训练集目标值归一化后写回
test_data_raw[target] = target_scaler.transform(test_data_raw[[target]])  # 将测试集目标值归一化后写回

# ===================== 序列数据构造（分别对训练集和测试集） =====================
# time_step 和 pred_offset 已在前面定义

def create_sequences(data, time_step):
    """按滑动窗口把二维特征表转换为三维时序样本。

    对每个起点 i 截取长度为 time_step 的历史窗口作为输入样本；
    对应的标签由调用方按 pred_offset 另行对齐。
    参数:
        data: 归一化后的特征 DataFrame（行代表时间步）。
        time_step: 每个样本包含的历史时间步数。
    返回:
        np.ndarray，形状为 (样本数, time_step, 特征数)。
    """
    sequences = []  # 用于收集各个滑动窗口样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点范围，保证窗口和预测偏移不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取第 i 个历史窗口并转 numpy 数组
    return np.array(sequences)  # 将所有窗口堆叠成三维数组并返回

# 分别为训练集和测试集创建序列（避免跨边界）
print("\n创建训练集序列...")  # 打印训练集序列构造提示
X_train = create_sequences(train_combined_features, time_step)  # 构造训练集输入序列
y_train = train_data_raw[target].values[time_step + pred_offset - 1:]  # 取与每个窗口对应的目标值（偏移后的时刻）
y_train = y_train[:len(X_train)]  # 确保长度匹配  # 确保标签长度与样本数一致

print("\n创建测试集序列...")  # 打印测试集序列构造提示
X_test = create_sequences(test_combined_features, time_step)  # 构造测试集输入序列
y_test = test_data_raw[target].values[time_step + pred_offset - 1:]  # 取测试集对应的目标值（偏移后的时刻）
y_test = y_test[:len(X_test)]  # 确保长度匹配  # 裁剪标签使其与测试样本数一致

print(f"\n序列数据形状:")  # 打印序列形状信息标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集输入与标签的形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集输入与标签的形状


# ===================== PyTorch模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    通过1D卷积实现高效的通道注意力机制，避免降维带来的信息损失
    输入与输出形状均为 (batch, time_step, channels)
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 通道注意力层。

        参数:
            channels: 输入特征的通道数（即特征维度）。
            gamma: 用于自适应计算卷积核大小的缩放系数，默认 2。
            b: 卷积核大小计算的偏置项，默认 1。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应推导卷积核尺寸参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数  # 若 t 为偶数则加 1，保证核大小为奇数

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道间交互，不设偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 将卷积输出映射为 0~1 的注意力权重

    def forward(self, x):
        """对输入执行通道注意力加权。

        参数:
            x: 形状 (batch, time_step, channels) 的输入张量。
        返回:
            形状同为 (batch, time_step, channels) 的加权后张量。
        """
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step) 以便进行池化
        x_permuted = x.permute(0, 2, 1)  # (batch, channels, time_step)  # 将通道维前移，满足 Conv1d 输入格式

        # 全局平均池化: (batch, channels, 1)
        y = self.avg_pool(x_permuted)  # 沿时间维做全局平均池化，得到每通道的全局描述

        # 1D卷积: (batch, 1, channels) -> (batch, 1, channels)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 转置后做 1D 卷积再转回，捕获通道相关性

        # Sigmoid激活
        y = self.sigmoid(y)  # (batch, channels, 1)  # 归一化为通道注意力权重

        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 将注意力权重按通道广播，逐通道加权输入

        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 还原维度顺序为 (batch, time_step, channels) 后返回


class GRUECAModel(nn.Module):
    """
    GRU + ECA注意力机制
    结合ECA通道注意力和GRU的时序建模能力
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):
        """初始化 GRU+ECA 模型。

        参数:
            input_dim: 每个时间步的输入特征维度。
            time_step: 输入序列长度（时间步数）。
            gru_hidden: GRU 隐藏层维度，默认 128。
            gru_layers: GRU 层数，默认 1（为 1 时 dropout 关闭）。
            dropout: 多层 GRU 层间的 dropout 比例，默认 0.2。
        """
        super(GRUECAModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间步长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层维度

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 层，对输入特征做通道注意力增强

        # 单层GRU编码器：提取时序特征
        self.gru = nn.GRU(  # 定义 GRU 循环网络层
            input_size=input_dim,  # 每个时间步的输入维度
            hidden_size=gru_hidden,  # 隐藏状态维度
            num_layers=gru_layers,  # GRU 堆叠层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout，单层设为 0
            batch_first=True,  # 输入张量以 batch 为第一维
            bidirectional=False  # 使用单向 GRU
        )

        # 输出层：直接从GRU隐藏状态映射到预测值
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层将隐藏状态映射为单个预测值

    def forward(self, x):
        """前向传播：ECA 增强 -> GRU 编码 -> 全连接预测。

        参数:
            x: 形状 (batch, time_step, input_dim) 的输入序列。
        返回:
            形状 (batch, 1) 的预测张量。
        """
        # x: (batch, time_step, input_dim)
        # ECA注意力增强输入特征
        x = self.eca(x)  # (batch, time_step, input_dim)  # 先经 ECA 层进行通道注意力加权

        # GRU编码：提取时序特征
        out, h = self.gru(x)  # out: (batch, time_step, hidden), h: (layers, batch, hidden)  # GRU 输出各时刻状态与最终隐状态

        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)  # 取序列最后一个时间步的输出作为整体表征

        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 经全连接层得到最终预测值并返回


class EarlyStopping:
    """通用早停策略，默认用于最小化验证损失"""
    def __init__(self, patience=10, min_delta=1e-4, mode='min', restore_best_weights=True, verbose=True):
        """
        Args:
            patience: 连续多少轮没有提升就停止
            min_delta: 认为"有提升"所需的最小变化量
            mode: 'min' 表示越小越好（如 loss），'max' 表示越大越好（如 R2）
            restore_best_weights: 早停后是否恢复到最佳权重
            verbose: 是否打印日志
        """
        self.patience = patience  # 保存容忍轮数
        self.min_delta = min_delta  # 保存判定提升的最小变化阈值
        self.mode = mode  # 保存优化方向（min/max）
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重的开关
        self.verbose = verbose  # 保存是否打印日志的开关

        self.counter = 0  # 初始化未提升计数器
        self.best_score = None  # 初始化最佳指标值
        self.best_epoch = 0  # 记录最佳指标出现的轮次
        self.early_stop = False  # 早停标志，初始为 False
        self.best_model = None  # 用于存放最佳模型权重副本

    def _is_improvement(self, score):
        """判断本次指标是否优于历史最佳。

        参数:
            score: 当前轮次的验证指标数值。
        返回:
            bool，True 表示有提升。
        """
        if self.best_score is None:  # 若尚无最佳值，第一次调用视为提升
            return True  # 直接判定为提升

        if self.mode == 'min':  # 最小化模式（如损失）
            return score < self.best_score - self.min_delta  # 需小于最佳值一定幅度才算提升
        elif self.mode == 'max':  # 最大化模式（如 R²）
            return score > self.best_score + self.min_delta  # 需大于最佳值一定幅度才算提升
        else:  # 其他非法模式
            raise ValueError("mode must be 'min' or 'max'")  # 抛出异常提示模式参数错误

    def __call__(self, score, model, epoch=None):
        """每轮训练后调用，更新最佳指标并判断是否触发早停。

        参数:
            score: 当前轮验证指标。
            model: 当前模型（提升时深拷贝其 state_dict）。
            epoch: 当前轮次编号，可选。
        返回:
            None。
        """
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 若指标为 None/NaN/无穷等非法值
            if self.verbose:  # 若开启日志
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印非法指标警告
            self.counter += 1  # 非法值也计入未提升次数
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 直接返回，不做后续判断

        if self._is_improvement(score):  # 若本轮指标有提升
            self.best_score = score  # 更新最佳指标
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 记录最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝保存当前最佳模型权重
            self.counter = 0  # 重置未提升计数器
            if self.verbose:  # 若开启日志
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印提升信息
        else:  # 本轮无提升
            self.counter += 1  # 未提升计数加 1
            if self.verbose:  # 若开启日志
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印当前计数
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停

    def restore(self, model):
        """在开启恢复开关且存在最佳权重时，把模型恢复到最佳状态。

        参数:
            model: 待恢复权重的模型对象。
        返回:
            None。
        """
        if self.restore_best_weights and self.best_model is not None:  # 若需恢复且已保存最佳权重
            model.load_state_dict(self.best_model)  # 将最佳权重加载回模型
            if self.verbose:  # 若开启日志
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):
    """训练模型并在验证集上早停。

    参数:
        model: 待训练的 PyTorch 模型。
        train_loader: 训练集 DataLoader。
        val_loader: 验证集 DataLoader。
        epochs: 最大训练轮数，默认 200。
        lr: Adam 优化器学习率，默认 0.001。
        patience: 早停容忍轮数，默认 10。
    返回:
        (train_losses, val_losses)：每轮的训练损失与验证损失列表。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数作为优化目标
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新模型参数
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器（默认最小化验证损失）

    train_losses = []  # 记录每轮训练损失
    val_losses = []  # 记录每轮验证损失

    for epoch in range(epochs):  # 按轮次循环训练
        # 训练阶段
        model.train()  # 切换模型为训练模式（启用 dropout 等）
        train_loss = 0.0  # 初始化本轮累计训练损失
        for batch_x, batch_y in train_loader:  # 遍历训练集中的每个批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将批次数据移动到指定设备

            optimizer.zero_grad()  # 清空上一轮的梯度
            outputs = model(batch_x)  # 前向传播得到预测
            loss = criterion(outputs, batch_y)  # 计算预测与真实标签的损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器根据梯度更新参数

            train_loss += loss.item()  # 累加当前批次的损失值

        train_loss /= len(train_loader)  # 求本轮平均训练损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换模型为评估模式（关闭 dropout 等）
        val_loss = 0.0  # 初始化本轮累计验证损失
        with torch.no_grad():  # 关闭梯度计算以节省显存与加速
            for batch_x, batch_y in val_loader:  # 遍历验证集中的每个批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将批次数据移动到指定设备
                outputs = model(batch_x)  # 前向传播得到预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加当前批次的验证损失

        val_loss /= len(val_loader)  # 求本轮平均验证损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次训练进展
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前轮次的训练/验证损失

        # 早停检查
        early_stopping(val_loss, model, epoch=epoch + 1)  # 用本轮验证损失更新早停状态
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停发生的轮次
            early_stopping.restore(model)  # 恢复最佳权重
            break  # 提前结束训练循环

    # 加载最佳模型
    early_stopping.restore(model)  # 训练结束后再次确保加载的是最佳权重

    return train_losses, val_losses  # 返回训练与验证损失记录


# ===================== K折交叉验证训练（仅在训练集上） =====================
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(feature_names)  # 输入维度取特征总数
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
#     fold_model = GRUECAModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2).to(device)
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
# plt.savefig('kfold_training_GRUECA_pytorch.png', dpi=150, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRUECA_pytorch.png")

# ===================== 训练最终模型 =====================
input_dim = len(feature_names)  # 使用特征总数  # 定义输入维度为特征总数

print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印阶段标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试样本数

X_train_tensor = torch.FloatTensor(X_train)  # 将训练序列转换为浮点张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 将训练标签转张量并升维为 (样本数,1)
X_test_tensor = torch.FloatTensor(X_test)  # 将测试序列转换为浮点张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 将测试标签转张量并升维为 (样本数,1)

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 按 80% 计算训练集内部切分点
X_train_split = X_train_tensor[:train_val_split]  # 切分出的实际训练部分
y_train_split = y_train_tensor[:train_val_split]  # 对应的训练标签
X_val_split = X_train_tensor[train_val_split:]  # 切分出的验证部分（后 20%）
y_val_split = y_train_tensor[train_val_split:]  # 对应的验证标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 构造训练数据集对象
val_dataset = TensorDataset(X_val_split, y_val_split)  # 构造验证数据集对象
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 构造测试数据集对象
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练加载器，打乱数据按批读取
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证加载器，不打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试加载器，不打乱

final_model = GRUECAModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2).to(device)  # 实例化最终模型并移至设备

print(final_model)  # 打印模型结构

print("\n开始训练最终模型...")  # 打印训练开始提示
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 统计并打印模型可训练参数总数
print(f"训练集样本数: {len(X_train_split)}")  # 打印实际训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,
                                       epochs=200, lr=0.001, patience=10)  # 调用训练函数训练最终模型

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练结束后的最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建画布，设置宽高
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置图表标题
plt.xlabel('训练轮数 (Epoch)')  # 设置 X 轴标签
plt.ylabel('损失值 (MSE)')  # 设置 Y 轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线并设透明度
plt.tight_layout()  # 自动调整子图间距
plt.savefig('training_loss_GRUECA_pytorch.png', dpi=150, bbox_inches='tight')  # 保存训练损失曲线图片
print("训练损失曲线已保存: training_loss_GRUECA_pytorch.png")  # 打印保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 打印测试预测开始提示

# 直接使用y_test（已经是归一化后的值）
y_test_normalized = y_test  # 测试标签本身已是归一化数值

# 反归一化得到真实值
y_test_rescaled = target_scaler.inverse_transform(y_test_normalized.reshape(-1, 1))  # 将测试标签反归一化还原为真实 THI

print(f"\n测试集数据检查:")  # 打印数据形状检查标题
print(f"X_test shape: {X_test.shape}")  # 打印测试输入形状
print(f"y_test shape: {y_test.shape}")  # 打印测试标签形状
print(f"y_test_rescaled shape: {y_test_rescaled.shape}")  # 打印反归一化后标签形状

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试序列转张量并移至设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算进行推理
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 前向预测并取回 CPU 转成 numpy 数组

# 反归一化预测值
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测结果反归一化为真实 THI 量纲

print(f"y_pred_rescaled shape: {y_pred_rescaled.shape}")  # 打印反归一化预测值形状

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse_test = np.sqrt(mse_test)  # 计算均方根误差 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果（GRU+ECA模型）")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R^2:  {r2:.4f}")  # 打印 R² 指标
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印 RMSE 指标
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印 MAE 指标
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建画布绘制测试集预测对比
# 确保长度匹配
y_test_rescaled_flat = y_test_rescaled.flatten()  # 将真实值展平为一维
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 将预测值展平为一维

# 从test_data_raw获取时间戳（已经是测试集部分）
test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset - 1:]  # 按预测偏移对齐测试时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 裁剪时间戳使其与绘图数据等长

print(f"\n绘图数据检查:")  # 打印绘图数据检查标题
print(f"test_timestamps length: {len(test_timestamps)}")  # 打印时间戳数量
print(f"y_test_rescaled_flat length: {len(y_test_rescaled_flat)}")  # 打印真实值数量
print(f"y_pred_rescaled_flat length: {len(y_pred_rescaled_flat)}")  # 打印预测值数量

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实THI', linewidth=2)  # 绘制真实 THI 曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+ECA预测', linewidth=2)  # 绘制预测 THI 曲线
plt.xlabel('时间')  # 设置 X 轴标签
plt.ylabel('THI ')  # 设置 Y 轴标签
plt.title(f'测试集THI结果（GRU+ECA） (R²: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置含指标的图表标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.savefig('test_prediction_GRUECA_pytorch.png', dpi=150, bbox_inches='tight')  # 保存测试集预测图
print("测试集预测结果已保存: test_prediction_GRUECA_pytorch.png")  # 打印保存提示

# ===================== 模型保存 =====================
torch.save({  # 保存模型检查点字典
    'model_state_dict': final_model.state_dict(),  # 保存模型权重
    'input_dim': input_dim,  # 保存输入维度，便于加载时重建
    'time_step': time_step,  # 保存时间窗口长度
    'feature_names': feature_names,  # 保存特征名称列表
    'feature_scaler': feature_scaler,  # 保存特征归一化器，供推理复用
    'target_scaler': target_scaler,  # 保存目标归一化器，供反归一化复用
}, 'Model_GRUECA_pytorch.pth')  # 指定检查点文件名
print("模型已保存: Model_GRUECA_pytorch.pth")  # 打印保存提示
# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 打印新数据处理提示
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取待预测的新数据文件
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟间隔从起始时间生成新数据时间戳

# 使用原始特征（不进行小波分解）
new_combined_features = new_data[features].copy()  # 抽取新数据的特征列（与训练时一致）

# 使用训练集的scaler进行归一化
new_combined_features = pd.DataFrame(  # 用训练集归一化器处理新特征
    feature_scaler.transform(new_combined_features),  # 对特征列做 MinMax 归一化
    columns=feature_names  # 保持特征列名
)

# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据中 THI 的原始真实值（未归一化）

X_new = create_sequences(new_combined_features, time_step)  # 构造新数据的输入序列
y_new = y_new_original[time_step + pred_offset:]  # 按预测偏移对齐新数据标签
X_new = X_new[:len(y_new)]  # 裁剪输入序列使其与标签等长

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 将新数据序列转张量并移至设备

final_model.eval()  # 切换为评估模式
with torch.no_grad():  # 关闭梯度计算进行推理
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据预测并取回 numpy 结果

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 将新数据预测值反归一化为真实 THI
y_new_rescaled = y_new.reshape(-1, 1)  # 原始值  # 将新数据真实值重塑为二维列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据的 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据的 MSE
rmse_new = np.sqrt(mse_new)  # 计算新数据的 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据的 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果（GRU+ECA模型）")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R²:   {r2_new:.4f}")  # 打印新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset:]  # 按预测偏移截取新数据的时间戳
plt.figure(figsize=(15, 5))  # 创建画布绘制新数据预测对比
# 确保长度匹配
y_new_rescaled_flat = y_new_rescaled.flatten()  # 展平新数据真实值
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 展平新数据预测值
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实THI', linewidth=2)  # 绘制新数据真实曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='GRU+ECA预测', linewidth=2)  # 绘制新数据预测曲线
plt.xlabel('时间')  # 设置 X 轴标签
plt.ylabel('THI ')  # 设置 Y 轴标签
plt.title(f'新数据集THI结果（GRU+ECA） (R²: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置含指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.savefig('new_data_prediction_GRUECA_pytorch.png', dpi=150, bbox_inches='tight')  # 保存新数据预测图
print("新数据集预测结果已保存: new_data_prediction_GRUECA_pytorch.png")  # 打印保存提示
print("\n========== 所有任务完成（GRU+ECA模型） ==========")  # 打印全部任务完成提示
