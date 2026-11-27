"""
TCN（时间卷积网络）模型训练与评估脚本

本脚本基于温湿度历史数据（THI，温湿指数）构建一个 TCN 时间序列预测模型，
整体流程与 lstm.py 对应，但将网络替换为膨胀因果卷积结构。主要步骤：
1. 固定随机种子以保证实验可复现；
2. 多编码尝试读取训练数据并生成 5 分钟间隔的时间戳；
3. 对缺失值区分连续/离散变量分别进行插值与前后向填充；
4. 按时间顺序 8:2 划分训练/测试集并做 Min-Max 归一化（仅在训练集拟合）；
5. 构造滑动时间窗口序列样本；
6. 定义 Chomp1d 裁剪模块、TemporalBlock 时间块、TCNModel 网络以及
   EarlyStopping 早停机制与训练函数，训练最终模型；
7. 在测试集与新数据上评估 R2 / RMSE / MAE，绘制损失曲线与预测对比图，
   并将模型权重、结构、归一化器保存到 .pth 供推理脚本使用。
"""
import pandas as pd  # 导入 pandas 库并重命名为 pd，用于表格数据的读取与处理
import numpy as np  # 导入 numpy 库并重命名为 np，用于数组与矩阵运算
import random  # 导入 random 模块，用于设置 Python 内置随机种子
import os  # 导入 os 模块，用于设置 PYTHONHASHSEED 环境变量
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于指定绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 模块并重命名为 plt，用于绘制图表
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标的归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、MSE、MAE 评估指标
from sklearn.model_selection import train_test_split, KFold  # 导入数据集划分与 K 折交叉验证工具（K折部分当前已注释）
import copy  # 导入 copy 模块，用于深拷贝最佳模型权重
import torch  # 导入 PyTorch 主库，用于张量计算与 GPU 加速
import torch.nn as nn  # 导入神经网络子库并重命名为 nn，用于搭建网络层
import torch.optim as optim  # 导入优化器子库并重命名为 optim，用于构造 Adam 优化器
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集与数据加载器，用于按批次训练/验证
# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据 GPU 是否可用选择计算设备
print(f"使用设备: {device}")  # 打印本次运行所使用的计算设备
if torch.cuda.is_available():  # 若 GPU 可用，则打印 GPU 详细信息
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体，以便正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，避免负号显示为方块

# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义依次尝试的 CSV 文件编码列表
data = None  # 初始化数据变量为 None，用于存放成功读取的 DataFrame
for enc in encodings:  # 遍历每一种候选编码
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 尝试用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 读取成功则打印所用编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 若出现解码错误则捕获并尝试下一种编码
        continue  # 继续尝试下一个编码

if data is None:  # 若所有编码均尝试失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 以固定起点、5 分钟间隔为每行生成时间戳
features = ['日龄', 'THI', '当前温度', '目标温度']  # 定义候选特征列名列表
target = 'THI'  # 定义预测目标列为 THI
features = [f for f in features if f in data.columns]  # 过滤掉数据中不存在的候选特征
# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印分隔线，标识缺失值处理环节开始
print("缺失值处理")  # 打印环节标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据的行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计标题
for col in features + [target]:  # 遍历所有特征列以及目标列
    if col in data.columns:  # 仅统计数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值数量
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比
        if nan_count > 0:  # 仅当存在缺失值时打印
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印缺失值数量与占比

# 定义连续变量和离散变量
continuous_vars = ['THI', '当前温度','目标温度']  # 定义连续型变量列表（适合线性插值）
discrete_vars = ['日龄']  # 定义离散型变量列表（只适合前后向填充）
# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每一个连续型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录处理前缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 对连续变量做双向线性插值
        data[col] = data[col].ffill()  # 用前一个有效值向前填充剩余缺失
        data[col] = data[col].bfill()  # 用后一个有效值向后填充剩余缺失
        after_nan = data[col].isna().sum()  # 记录处理后缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后缺失值数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每一个离散型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录处理前缺失值数量
        data[col] = data[col].ffill()  # 用前一个有效值向前填充缺失值
        data[col] = data[col].bfill()  # 用后一个有效值向后填充缺失值
        after_nan = data[col].isna().sum()  # 记录处理后缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失值数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除在特征或目标列上仍含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据形状
print(f"{'='*60}\n")  # 打印分隔线，标识缺失值处理环节结束

print(f"\n【TCN模型】")  # 打印当前使用的模型名称
print(f"使用特征: {features}")  # 打印实际参与建模的特征列表
print(f"特征数量: {len(features)}")  # 打印特征数量

# ===================== 数据集划分 =====================
time_step = 36  # TCN时间窗口：48步 × 5分钟 = 4小时
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  TCN时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印窗口步数及对应分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印偏移步数及对应分钟数

# 划分数据集
split_index = int(len(data) * 0.8)  # 计算 80% 位置作为训练/测试划分索引
train_data = data.iloc[:split_index].copy()  # 按时间顺序取前 80% 作为训练集
test_data = data.iloc[split_index:].copy()  # 按时间顺序取后 20% 作为测试集

print(f"\n数据集划分:")  # 打印数据集划分标题
print(f"训练集大小: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")  # 打印训练样本数及占比
print(f"测试集大小: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")  # 打印测试样本数及占比

# ===================== 归一化 =====================
print("\n归一化处理...")  # 提示开始归一化处理
feature_scaler = MinMaxScaler()  # 创建特征归一化器
target_scaler = MinMaxScaler()  # 创建目标归一化器

# 只在训练集上fit
feature_scaler.fit(train_data[features])  # 仅用训练集特征拟合归一化器，避免数据泄漏
target_scaler.fit(train_data[[target]])  # 仅用训练集目标拟合归一化器，避免数据泄漏

# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 构造归一化后的训练特征 DataFrame
    feature_scaler.transform(train_data[features]),  # 用训练集归一化器变换训练特征
    columns=features  # 指定列名
)
test_features = pd.DataFrame(  # 构造归一化后的测试特征 DataFrame
    feature_scaler.transform(test_data[features]),  # 用训练集归一化器变换测试特征
    columns=features  # 指定列名
)

train_target = target_scaler.transform(train_data[[target]]).flatten()  # 归一化训练目标并展平
test_target = target_scaler.transform(test_data[[target]]).flatten()  # 归一化测试目标并展平

# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):  # 定义函数：把二维特征表切分为滑动窗口三维序列
    """
    根据时间窗口长度，将二维特征数据切分为多个滑动窗口序列样本。

    主要参数:
        data (DataFrame): 已归一化的特征表。
        time_step (int): 每个样本包含的历史时间步数。

    返回:
        np.ndarray: 形状为 (样本数, time_step, 特征数) 的三维数组。
        预测偏移 pred_offset 在此处作为外部全局变量使用。
    """
    sequences = []  # 初始化样本列表
    for i in range(len(data) - time_step - pred_offset):  # 窗口起点遍历，预留窗口与偏移长度
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行并转为数组
    return np.array(sequences)  # 堆叠为三维数组返回

print("\n创建训练集序列...")  # 提示开始构造训练集序列
X_train = create_sequences(train_features, time_step)  # 为训练特征构造序列
y_train = train_target[time_step + pred_offset - 1:]  # 取与窗口末端对应的训练标签
y_train = y_train[:len(X_train)]  # 截断标签使其长度与序列样本数一致

print("\n创建测试集序列...")  # 提示开始构造测试集序列
X_test = create_sequences(test_features, time_step)  # 为测试特征构造序列
y_test = test_target[time_step + pred_offset - 1:]  # 取与窗口末端对应的测试标签
y_test = y_test[:len(X_test)]  # 截断标签使其长度与序列样本数一致

print(f"\n序列数据形状:")  # 打印序列形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集序列与标签形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集序列与标签形状

# ===================== TCN模型定义 =====================
class Chomp1d(nn.Module):  # 定义裁剪模块，用于因果卷积（去掉卷积尾部多出的时间步）
    """裁剪模块，用于因果卷积"""
    def __init__(self, chomp_size):  # 构造函数
        """
        初始化裁剪模块。

        主要参数:
            chomp_size (int): 需要从序列尾部裁剪掉的时间步数。
        """
        super(Chomp1d, self).__init__()  # 调用父类构造函数
        self.chomp_size = chomp_size  # 保存裁剪长度

    def forward(self, x):  # 定义前向传播
        """
        前向传播：裁掉输入尾部 chomp_size 个时间步并保证内存连续。

        主要参数:
            x (Tensor): 形状为 (batch, channels, time_steps) 的卷积输出。

        返回:
            Tensor: 裁剪后的连续张量。
        """
        return x[:, :, :-self.chomp_size].contiguous()  # 去掉尾部 chomp_size 个时间步并转为连续内存布局


class TemporalBlock(nn.Module):  # 定义 TCN 基本时间块：两层膨胀因果卷积 + 残差连接
    """TCN的基本时间块"""
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
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,  # 第一层膨胀因果卷积
                               stride=stride, padding=padding, dilation=dilation)  # 指定步长、padding 与膨胀系数
        self.chomp1 = Chomp1d(padding)  # 裁掉第一层卷积尾部多余的 padding
        self.relu1 = nn.ReLU()  # 第一层激活函数
        self.dropout1 = nn.Dropout(dropout)  # 第一层 dropout

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,  # 第二层膨胀因果卷积
                               stride=stride, padding=padding, dilation=dilation)  # 参数与第一层相同
        self.chomp2 = Chomp1d(padding)  # 裁掉第二层卷积尾部多余的 padding
        self.relu2 = nn.ReLU()  # 第二层激活函数
        self.dropout2 = nn.Dropout(dropout)  # 第二层 dropout

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,  # 把主路径各层组合成顺序网络
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 主路径结束
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 通道数不一致时用 1×1 卷积匹配残差维度
        self.relu = nn.ReLU()  # 残差相加后的最终激活函数

    def forward(self, x):  # 定义前向传播
        """
        前向传播：主路径输出与残差相加后再激活。

        主要参数:
            x (Tensor): 形状为 (batch, n_inputs, time_steps) 的输入。

        返回:
            Tensor: 形状为 (batch, n_outputs, time_steps) 的输出。
        """
        out = self.net(x)  # 计算主路径输出
        res = x if self.downsample is None else self.downsample(x)  # 通道一致则直接用输入作残差，否则下采样
        return self.relu(out + res)  # 残差相加并激活后返回


class TCNModel(nn.Module):  # 定义 TCN 预测模型：多个 TemporalBlock 堆叠后接全连接层
    """
    TCN模型
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)
    """
    def __init__(self, input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.2):  # 构造函数
        """
        初始化 TCNModel。

        主要参数:
            input_dim (int): 输入特征维度。
            time_step (int): 时间步长度。
            num_channels (list[int]): 每个时间块的输出通道数，默认 [128]。
            kernel_size (int): 卷积核大小，默认 3。
            dropout (float): dropout 概率，默认 0.2。
        """
        super(TCNModel, self).__init__()  # 调用父类构造函数
        self.time_step = time_step  # 保存时间步长度
        self.input_dim = input_dim  # 保存输入特征维度
        
        layers = []  # 初始化列表，收集各时间块
        num_levels = len(num_channels)  # 计算需要构建的层数
        for i in range(num_levels):  # 按通道配置逐层构建
            dilation_size = 2 ** i  # 膨胀系数按 2 的幂递增
            in_channels = input_dim if i == 0 else num_channels[i-1]  # 首层输入为特征维度，后续为上一层通道数
            out_channels = num_channels[i]  # 本层输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,  # 构造时间块并加入列表
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]  # padding 保证输入输出等长

        self.network = nn.Sequential(*layers)  # 把所有时间块组合成顺序网络
        self.fc = nn.Linear(num_channels[-1], 1)  # 全连接层：输入为最后一层通道数

    def forward(self, x):  # 定义前向传播
        """
        前向传播：输入换维后经过 TCN 编码，取最后时刻特征输出预测。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        # x: (batch, time_step, input_dim)
        # TCN需要 (batch, input_dim, time_step)
        x = x.transpose(1, 2)  # (batch, input_dim, time_step)
        
        # TCN编码
        out = self.network(x)  # (batch, num_channels[-1], time_step)
        
        # 使用最后时刻的特征
        out_last = out[:, :, -1]  # (batch, num_channels[-1])
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)


class EarlyStopping:  # 定义通用早停类，在验证指标不再提升时终止训练
    """通用早停策略，默认用于最小化验证损失"""
    def __init__(self, patience=10, min_delta=1e-4, mode='min', restore_best_weights=True, verbose=True):  # 初始化早停参数
        """
        Args:
            patience: 连续多少轮没有提升就停止
            min_delta: 认为"有提升"所需的最小变化量
            mode: 'min' 表示越小越好（如 loss），'max' 表示越大越好（如 R2）
            restore_best_weights: 早停后是否恢复到最佳权重
            verbose: 是否打印日志
        """
        self.patience = patience  # 保存容忍轮数
        self.min_delta = min_delta  # 保存判定提升的最小变化量
        self.mode = mode  # 保存优化方向
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重
        self.verbose = verbose  # 保存是否打印日志

        self.counter = 0  # 初始化“未提升”计数器
        self.best_score = None  # 初始化最佳指标值
        self.best_epoch = 0  # 初始化最佳轮次
        self.early_stop = False  # 初始化早停标志
        self.best_model = None  # 初始化最佳权重

    def _is_improvement(self, score):  # 内部方法：判断当前指标是否优于历史最佳
        """
        判断当前指标相对于历史最佳是否算作“提升”。

        主要参数:
            score (float): 当前轮的验证指标值。

        返回:
            bool: 有提升返回 True，否则 False。
        """
        if self.best_score is None:  # 若还没有历史最佳
            return True  # 首次调用视为提升

        if self.mode == 'min':  # 若优化方向为最小化
            return score < self.best_score - self.min_delta  # 需低于最佳值至少 min_delta
        elif self.mode == 'max':  # 若优化方向为最大化
            return score > self.best_score + self.min_delta  # 需高于最佳值至少 min_delta
        else:  # 若 mode 取值非法
            raise ValueError("mode must be 'min' or 'max'")  # 抛出异常

    def __call__(self, score, model, epoch=None):  # 使实例可像函数一样调用，更新早停状态
        """
        每轮训练后调用：根据当前验证指标更新最佳权重与早停计数器。

        主要参数:
            score (float): 当前轮的验证指标值。
            model (nn.Module): 当前模型，用于深拷贝权重。
            epoch (int, 可选): 当前轮次编号，仅用于记录最佳轮次。

        返回:
            无返回值。
        """
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 若指标为 None、NaN 或无穷
            if self.verbose:  # 若开启日志
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印无效指标警告
            self.counter += 1  # 累加未提升计数
            if self.counter >= self.patience:  # 若达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 提前返回

        if self._is_improvement(score):  # 若本轮有提升
            self.best_score = score  # 更新最佳指标
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 记录最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝最佳权重
            self.counter = 0  # 重置计数
            if self.verbose:  # 若开启日志
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印更新信息
        else:  # 若本轮无提升
            self.counter += 1  # 计数加 1
            if self.verbose:  # 若开启日志
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印当前计数
            if self.counter >= self.patience:  # 若达到容忍上限
                self.early_stop = True  # 触发早停

    def restore(self, model):  # 将模型权重恢复为历史最佳权重
        """
        在需要时把模型参数恢复到历史最佳权重。

        主要参数:
            model (nn.Module): 待恢复权重的模型对象。

        返回:
            无返回值。
        """
        if self.restore_best_weights and self.best_model is not None:  # 仅当开启恢复且存在最佳权重
            model.load_state_dict(self.best_model)  # 加载最佳权重到模型
            if self.verbose:  # 若开启日志
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 定义模型训练函数
    """训练模型"""
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器
    early_stopping = EarlyStopping(patience=patience)  # 创建早停实例

    train_losses = []  # 记录每轮训练损失
    val_losses = []  # 记录每轮验证损失

    for epoch in range(epochs):  # 外层循环：遍历每个训练轮次
        # 训练阶段
        model.train()  # 切换为训练模式（启用 dropout）
        train_loss = 0.0  # 初始化本轮累计训练损失
        for batch_x, batch_y in train_loader:  # 按批次读取训练数据
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将输入与标签移至目标设备

            optimizer.zero_grad()  # 清空上一批次梯度
            outputs = model(batch_x)  # 前向传播得到预测
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            train_loss += loss.item()  # 累加本批次损失

        train_loss /= len(train_loader)  # 求本轮平均训练损失
        train_losses.append(train_loss)  # 记录到列表

        # 验证阶段
        model.eval()  # 切换为评估模式（关闭 dropout）
        val_loss = 0.0  # 初始化本轮累计验证损失
        with torch.no_grad():  # 关闭梯度计算
            for batch_x, batch_y in val_loader:  # 按批次读取验证数据
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将输入与标签移至目标设备
                outputs = model(batch_x)  # 前向传播得到预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加本批次验证损失

        val_loss /= len(val_loader)  # 求本轮平均验证损失
        val_losses.append(val_loss)  # 记录到列表

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印训练与验证损失

        # 早停检查
        early_stopping(val_loss, model, epoch=epoch + 1)  # 用本轮验证损失更新早停状态
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印触发轮次
            early_stopping.restore(model)  # 恢复最佳权重
            break  # 跳出训练循环

    # 加载最佳模型
    early_stopping.restore(model)  # 训练结束后确保处于最佳权重

    return train_losses, val_losses  # 返回训练与验证损失曲线


# ===================== K折交叉验证训练 =====================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)  # （已注释）创建 5 折交叉验证划分器
# val_losses_all = []  # （已注释）汇总各折最佳验证损失
# history_list = []  # （已注释）汇总各折损失曲线
input_dim = len(features)  # 计算输入特征维度
#
# print("\n========== 开始5折交叉验证（仅训练集） ==========")  # （已注释）打印开始提示
# print(f"交叉验证数据量: {len(X_train)} 样本")  # （已注释）打印样本数
#
# for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):  # （已注释）遍历每一折
#     print(f"\n{'=' * 50}")  # （已注释）打印分隔线
#     print(f"第 {fold}/5 折训练")  # （已注释）打印当前折
#     print(f"{'=' * 50}")  # （已注释）打印分隔线
#
#     X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # （已注释）切分本折训练/验证特征
#     y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # （已注释）切分本折训练/验证标签
#
#     print(f"训练集样本数: {len(X_train_fold)}")  # （已注释）打印训练样本数
#     print(f"验证集样本数: {len(X_val_fold)}")  # （已注释）打印验证样本数
#
#     X_train_tensor = torch.FloatTensor(X_train_fold)  # （已注释）训练特征转张量
#     y_train_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)  # （已注释）训练标签转列向量张量
#     X_val_tensor = torch.FloatTensor(X_val_fold)  # （已注释）验证特征转张量
#     y_val_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)  # （已注释）验证标签转列向量张量
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # （已注释）构造训练数据集
#     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # （已注释）构造验证数据集
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # （已注释）训练数据加载器（打乱）
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # （已注释）验证数据加载器（不打乱）
#
#     # 构建TCN模型（参数量与LSTM接近：~69K）
#     model = TCNModel(input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.2).to(device)  # （已注释）构建本折模型
#
#     if fold == 1:  # （已注释）仅第一折
#         print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # （已注释）打印参数量
#
#     train_losses, val_losses = train_model(model, train_loader, val_loader,  # （已注释）训练本折模型
#                                            epochs=200, lr=0.001, patience=10)  # （已注释）轮数/学习率/早停
#     history_list.append({  # （已注释）保存本折损失曲线
#         'train_losses': train_losses,  # （已注释）训练损失
#         'val_losses': val_losses  # （已注释）验证损失
#     })  # （已注释）字典结束
#
#     best_val_loss = min(val_losses)  # （已注释）取本折最小验证损失
#     val_losses_all.append(best_val_loss)  # （已注释）加入汇总
#     print(f"\n第 {fold} 折最佳验证损失: {best_val_loss:.6f}")  # （已注释）打印本折最佳损失
#
# print(f"\n{'=' * 50}")  # （已注释）打印分隔线
# print("5折交叉验证结果汇总")  # （已注释）打印汇总标题
# print(f"{'=' * 50}")  # （已注释）打印分隔线
# for i, loss in enumerate(val_losses_all, 1):  # （已注释）遍历各折损失
#     print(f"第 {i} 折验证损失: {loss:.6f}")  # （已注释）打印每折损失
# print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")  # （已注释）打印平均值
# print(f"验证损失标准差: {np.std(val_losses_all):.6f}")  # （已注释）打印标准差
#
# # ===================== 绘制K折交叉验证训练过程曲线 =====================
# plt.figure(figsize=(15, 10))  # （已注释）创建画布
#
# plt.subplot(2, 2, 1)  # （已注释）选择第 1 子图
# for i, history in enumerate(history_list, 1):  # （已注释）遍历各折历史
#     plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)  # （已注释）绘制训练损失曲线
# plt.title('5折交叉验证 - 训练损失曲线', fontsize=14)  # （已注释）设置标题
# plt.xlabel('训练轮数 (Epoch)')  # （已注释）设置横轴标签
# plt.ylabel('损失值 (MSE)')  # （已注释）设置纵轴标签
# plt.legend()  # （已注释）显示图例
# plt.grid(True, alpha=0.3)  # （已注释）显示网格
#
# plt.subplot(2, 2, 2)  # （已注释）选择第 2 子图
# for i, history in enumerate(history_list, 1):  # （已注释）遍历各折历史
#     plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)  # （已注释）绘制验证损失曲线
# plt.title('5折交叉验证 - 验证损失曲线', fontsize=14)  # （已注释）设置标题
# plt.xlabel('训练轮数 (Epoch)')  # （已注释）设置横轴标签
# plt.ylabel('损失值 (MSE)')  # （已注释）设置纵轴标签
# plt.legend()  # （已注释）显示图例
# plt.grid(True, alpha=0.3)  # （已注释）显示网格
#
# plt.subplot(2, 2, 3)  # （已注释）选择第 3 子图
# folds = [f'Fold {i}' for i in range(1, 6)]  # （已注释）生成折名称
# plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)  # （已注释）绘制各折最佳损失柱状图
# plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',  # （已注释）绘制平均参考线
#             label=f'平均值: {np.mean(val_losses_all):.6f}')  # （已注释）设置图例标签
# plt.title('各折最佳验证损失对比', fontsize=14)  # （已注释）设置标题
# plt.xlabel('折数')  # （已注释）设置横轴标签
# plt.ylabel('验证损失 (MSE)')  # （已注释）设置纵轴标签
# plt.legend()  # （已注释）显示图例
# plt.grid(True, alpha=0.3, axis='y')  # （已注释）仅显示横向网格
#
# plt.subplot(2, 2, 4)  # （已注释）选择第 4 子图
# max_epochs = max(len(h['train_losses']) for h in history_list)  # （已注释）求最长轮数
# avg_train_losses = []  # （已注释）平均训练损失列表
# avg_val_losses = []  # （已注释）平均验证损失列表
# for epoch in range(max_epochs):  # （已注释）按轮遍历
#     train_vals = [h['train_losses'][epoch] for h in history_list if epoch < len(h['train_losses'])]  # （已注释）收集该轮训练损失
#     val_vals = [h['val_losses'][epoch] for h in history_list if epoch < len(h['val_losses'])]  # （已注释）收集该轮验证损失
#     if train_vals:  # （已注释）若有数据
#         avg_train_losses.append(np.mean(train_vals))  # （已注释）保存平均训练损失
#     if val_vals:  # （已注释）若有数据
#         avg_val_losses.append(np.mean(val_vals))  # （已注释）保存平均验证损失
#
# plt.plot(avg_train_losses, label='平均训练损失', linewidth=2)  # （已注释）绘制平均训练损失曲线
# plt.plot(avg_val_losses, label='平均验证损失', linewidth=2)  # （已注释）绘制平均验证损失曲线
# plt.title('5折平均损失曲线', fontsize=14)  # （已注释）设置标题
# plt.xlabel('训练轮数 (Epoch)')  # （已注释）设置横轴标签
# plt.ylabel('损失值 (MSE)')  # （已注释）设置纵轴标签
# plt.legend()  # （已注释）显示图例
# plt.grid(True, alpha=0.3)  # （已注释）显示网格
#
# plt.tight_layout()  # （已注释）调整布局
# plt.savefig('kfold_training_TCN_pytorch.png', dpi=150, bbox_inches='tight')  # （已注释）保存曲线图
# print("\nK折交叉验证损失曲线已保存: kfold_training_TCN_pytorch.png")  # （已注释）打印提示

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印环节标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练序列样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试序列样本数

X_train_tensor = torch.FloatTensor(X_train)  # 训练序列转浮点张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练标签转列向量张量
X_test_tensor = torch.FloatTensor(X_test)  # 测试序列转浮点张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试标签转列向量张量

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 计算训练集内部 80/20 切分索引
X_train_split = X_train_tensor[:train_val_split]  # 取前 80% 作为真正训练数据
y_train_split = y_train_tensor[:train_val_split]  # 取前 80% 对应标签
X_val_split = X_train_tensor[train_val_split:]  # 取后 20% 作为验证数据
y_val_split = y_train_tensor[train_val_split:]  # 取后 20% 对应标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 构造训练数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 构造验证数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 构造测试数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练加载器：每批 32 且打乱
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证加载器：不打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试加载器：不打乱

final_model = TCNModel(input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.2).to(device)  # 实例化最终 TCN 模型并移至设备

print(final_model)  # 打印模型结构

print("\n开始训练最终模型...")  # 提示开始训练
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印模型参数总量
print(f"训练集样本数: {len(X_train_split)}")  # 打印实际训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,  # 调用训练函数训练最终模型
                                       epochs=200, lr=0.001, patience=10)  # 最多 200 轮，学习率 0.001，早停容忍 10 轮

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练完成与最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12×6 英寸画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 设置横轴标签
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示半透明网格
plt.tight_layout()  # 自动调整布局
plt.savefig('training_loss_TCN_pytorch.png', dpi=150, bbox_inches='tight')  # 保存损失曲线图
print("训练损失曲线已保存: training_loss_TCN_pytorch.png")  # 打印保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将归一化测试标签反归一化回原始 THI

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试序列转张量并移至设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 预测并搬回 CPU 转 NumPy 数组

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测结果反归一化回原始 THI

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算 R2
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算 MSE
rmse_test = np.sqrt(mse_test)  # 计算 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印 R2
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15×5 英寸画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 真实标签展平
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 预测结果展平

test_timestamps = test_data['timestamp'].values[time_step + pred_offset - 1:]  # 取出与测试样本对应的时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳使其与结果长度一致

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实THI', linewidth=2)  # 绘制真实 THI 曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='TCN预测', linewidth=2)  # 绘制 TCN 预测曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('THI ')  # 设置纵轴标签
plt.title(f'测试集THI结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置含指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示半透明网格
plt.savefig('test_prediction_TCN_pytorch.png', dpi=150, bbox_inches='tight')  # 保存测试集预测对比图
print("测试集预测结果已保存: test_prediction_TCN_pytorch.png")  # 打印保存提示

# ===================== 模型保存 =====================
torch.save({  # 打包保存模型相关对象
    'model_state_dict': final_model.state_dict(),  # 保存模型权重
    'input_dim': input_dim,  # 保存输入特征维度
    'time_step': time_step,  # 保存时间窗口长度
    'features': features,  # 保存特征列表
    'feature_scaler': feature_scaler,  # 保存特征归一化器
    'target_scaler': target_scaler,  # 保存目标归一化器
}, 'Model_TCN_pytorch.pth')  # 指定保存路径
print("模型已保存: Model_TCN_pytorch.pth")  # 打印保存提示

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始处理新数据
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取新数据 CSV
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',  # 以索引×5 分钟生成时间戳
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 指定起始基准时间

# 归一化
new_features = pd.DataFrame(  # 构造归一化后的新数据特征
    feature_scaler.transform(new_data[features]),  # 用训练集归一化器变换新数据特征
    columns=features  # 指定列名
)

# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据中未归一化的真实 THI

X_new = create_sequences(new_features, time_step)  # 为新数据构造滑动窗口序列
y_new = y_new_original[time_step + pred_offset:]  # 按窗口与偏移取出对应的真实标签
X_new = X_new[:len(y_new)]  # 截断序列使其长度与标签数量一致

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 新数据序列转张量并移至设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据预测并转 NumPy 数组

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 新数据预测结果反归一化
y_new_rescaled = y_new.reshape(-1, 1)  # 新数据真实标签重塑为列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R2
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MSE
rmse_new = np.sqrt(mse_new)  # 计算新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 打印新数据 R2
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset:]  # 取出与新数据样本对应的时间戳
plt.figure(figsize=(15, 5))  # 创建 15×5 英寸画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 新数据真实标签展平
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 新数据预测结果展平
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实THI', linewidth=2)  # 绘制新数据真实 THI 曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='TCN预测', linewidth=2)  # 绘制新数据预测曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('THI ')  # 设置纵轴标签
plt.title(f'新数据集THI结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置含指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示半透明网格
plt.savefig('new_data_prediction_TCN_pytorch.png', dpi=150, bbox_inches='tight')  # 保存新数据预测对比图
print("新数据集预测结果已保存: new_data_prediction_TCN_pytorch.png")  # 打印保存提示
print("\n========== 所有任务完成 ==========")  # 打印全部完成提示
