"""
温度偏差预测 - TCN（时间卷积网络）模型完整训练与评估脚本。

本脚本是 lstm.py 的 TCN 对照版本：读取温湿度数据并处理缺失值，按时间窗口构造序列，
先做 5 折交叉验证评估模型稳定性，再用全部训练数据训练最终 TCN 模型（带早停），
最后在测试集与新数据集上评估并可视化。训练好的模型与归一化器保存为
Model_TCN_pytorch.pth，供 tcnmodel.py 加载推理。
"""
import pandas as pd  # 导入 pandas 数据分析库并起别名 pd，用于 CSV 读取与表格处理
import numpy as np  # 导入 numpy 数值计算库并起别名 np，用于数组构造与数学运算
import random  # 导入 random 模块，用于设置 Python 内置随机种子
import os  # 导入 os 模块，用于设置环境变量 PYTHONHASHSEED
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于切换绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片  # Agg 无需图形界面，适合服务器离线存图
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口并起别名 plt，用于绘制曲线
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R²、MSE、MAE 评估指标
from sklearn.model_selection import train_test_split, KFold  # 导入数据划分与 K 折交叉验证工具
import copy  # 导入 copy 模块，用于深拷贝模型权重保存最佳状态
import torch  # 导入 PyTorch 主库，用于张量计算与自动求导
import torch.nn as nn  # 导入 PyTorch 神经网络模块并起别名 nn，用于搭建网络层
import torch.optim as optim  # 导入 PyTorch 优化器模块并起别名 optim
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集与数据加载器工具

# ===================== 随机种子设置（确保可复现） =====================
def set_seed(seed=42):
    """固定各库随机种子以保证实验结果可复现。

    参数:
        seed: 随机种子值，默认 42。
    返回:
        无返回值，直接设置全局随机状态与 cuDNN 确定性开关。
    """
    random.seed(seed)  # 固定 Python random 模块的种子
    np.random.seed(seed)  # 固定 numpy 的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 Python 哈希种子，减少哈希随机性
    torch.manual_seed(seed)  # 固定 CPU 上的 torch 随机种子
    torch.cuda.manual_seed(seed)  # 固定当前 GPU 的 torch 随机种子
    torch.cuda.manual_seed_all(seed)  # 固定所有 GPU 的 torch 随机种子
    torch.backends.cudnn.deterministic = True  # 让 cuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，避免算法选择引入不确定性

set_seed(42)  # 以种子 42 调用上述函数，固定本次实验随机性

# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有 GPU 则用 cuda，否则回退 cpu
print(f"使用设备: {device}")  # 打印本次训练使用的计算设备
if torch.cuda.is_available():  # 仅当检测到 GPU 时输出细节
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 型号
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭负号 Unicode 显示

# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
data = None  # 初始化数据变量为空
for enc in encodings:  # 逐个编码尝试读取训练数据
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功编码
        break  # 成功即跳出
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码失败
        continue  # 尝试下一种编码

if data is None:  # 所有编码均失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛错终止

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔生成时间戳
features = ['日龄','温度偏差','外界温度','CO2','目标温度']  # 预定义候选特征列表
target = '温度偏差'  # 指定目标列
features = [f for f in features if f in data.columns]  # 仅保留实际存在的特征列

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 输出原始数据形状
print(f"\n各列缺失值统计:")  # 提示逐列统计缺失
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 只统计存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失数量
        nan_pct = nan_count / len(data) * 100  # 计算缺失占比
        if nan_count > 0:  # 仅输出有缺失的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印缺失统计

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续变量用线性插值填补
discrete_vars = ['日龄']  # 离散变量仅前后填充

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 列存在且有缺失才处理
        print(f"\n处理 {col}:")  # 打印当前列名
        before_nan = data[col].isna().sum()  # 记录处理前缺失数
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 双向线性插值
        data[col] = data[col].ffill()  # 前向填充
        data[col] = data[col].bfill()  # 后向填充
        after_nan = data[col].isna().sum()  # 统计处理后缺失数
        print(f"  插值前: {before_nan} 个缺失值")  # 输出插值前数量
        print(f"  插值后: {after_nan} 个缺失值")  # 输出插值后数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 列存在且有缺失才处理
        print(f"\n处理 {col}:")  # 打印当前列名
        before_nan = data[col].isna().sum()  # 记录处理前缺失数
        data[col] = data[col].ffill()  # 前向填充
        data[col] = data[col].bfill()  # 后向填充
        after_nan = data[col].isna().sum()  # 统计处理后缺失数
        print(f"  填充前: {before_nan} 个缺失值")  # 输出填充前数量
        print(f"  填充后: {after_nan} 个缺失值")  # 输出填充后数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 汇总剩余缺失总数
if remaining_nan > 0:  # 仍有缺失
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 提示删除
    data = data.dropna(subset=features + [target])  # 丢弃含缺失样本

print(f"\n处理后数据形状: {data.shape}")  # 输出处理后形状
print(f"{'='*60}\n")  # 打印分隔线

print(f"\n【TCN模型】")  # 打印模型名称标题
print(f"使用特征: {features}")  # 列出使用特征
print(f"特征数量: {len(features)}")  # 打印特征个数

# ===================== 数据集划分 =====================
time_step = 36  # TCN时间窗口：48步 × 5分钟 = 4小时  # 输入序列回看的历史步数
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟  # 标签相对窗口末尾的未来偏移量

print(f"\n时间窗口配置:")  # 提示输出窗口配置
print(f"  TCN时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印窗口长度与分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印偏移步数与分钟数

# 划分数据集
split_index = int(len(data) * 0.8)  # 按 8:2 计算划分位置
train_data = data.iloc[:split_index].copy()  # 前 80% 作为训练集副本
test_data = data.iloc[split_index:].copy()  # 后 20% 作为测试集副本

print(f"\n数据集划分:")  # 提示输出划分结果
print(f"训练集大小: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")  # 打印训练集样本数与占比
print(f"测试集大小: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")  # 打印测试集样本数与占比

# ===================== 归一化 =====================
print("\n归一化处理...")  # 提示开始归一化
feature_scaler = MinMaxScaler()  # 实例化特征归一化器
target_scaler = MinMaxScaler()  # 实例化目标归一化器

# 只在训练集上fit
feature_scaler.fit(train_data[features])  # 仅用训练集特征拟合缩放范围
target_scaler.fit(train_data[[target]])  # 仅用训练集目标拟合缩放范围

# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 构造归一化训练特征表
    feature_scaler.transform(train_data[features]),  # 变换训练特征
    columns=features  # 指定列名
)
test_features = pd.DataFrame(  # 构造归一化测试特征表
    feature_scaler.transform(test_data[features]),  # 用同一缩放器变换测试特征
    columns=features  # 指定列名
)

train_target = target_scaler.transform(train_data[[target]]).flatten()  # 归一化训练目标并展平
test_target = target_scaler.transform(test_data[[target]]).flatten()  # 归一化测试目标并展平

# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):
    """把特征表滑窗切分为 (样本数, time_step, 特征数) 的时序样本。

    参数:
        data: 归一化后的特征 DataFrame。
        time_step: 每个样本的历史窗口长度。
    返回:
        np.ndarray 三维序列数组。
    """
    sequences = []  # 初始化窗口列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，预留窗口与偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 截取窗口并转数组
    return np.array(sequences)  # 堆叠为三维数组返回

print("\n创建训练集序列...")  # 提示构造训练序列
X_train = create_sequences(train_features, time_step)  # 生成训练输入序列
y_train = train_target[time_step + pred_offset - 1:]  # 按偏移截取训练标签
y_train = y_train[:len(X_train)]  # 截断标签与输入数量对齐

print("\n创建测试集序列...")  # 提示构造测试序列
X_test = create_sequences(test_features, time_step)  # 生成测试输入序列
y_test = test_target[time_step + pred_offset - 1:]  # 按偏移截取测试标签
y_test = y_test[:len(X_test)]  # 截断标签与输入数量对齐

print(f"\n序列数据形状:")  # 提示输出序列形状
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练序列形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试序列形状

# ===================== TCN模型定义 =====================
class Chomp1d(nn.Module):
    """裁剪模块，用于因果卷积：去掉卷积尾部多出的 padding。"""
    def __init__(self, chomp_size):
        """初始化裁剪模块。

        参数:
            chomp_size: 需从尾部裁掉的时间步数。
        """
        super(Chomp1d, self).__init__()  # 调用父类初始化
        self.chomp_size = chomp_size  # 保存裁剪长度

    def forward(self, x):
        """裁掉最后一维尾部若干步并返回连续张量。"""
        return x[:, :, :-self.chomp_size].contiguous()  # 尾部裁剪并转连续内存


class TemporalBlock(nn.Module):
    """TCN的基本时间块：两层膨胀卷积 + 裁剪 + ReLU + Dropout + 残差。"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        """初始化时间块。

        参数:
            n_inputs/n_outputs: 输入/输出通道数；kernel_size: 卷积核大小。
            stride: 步长；dilation: 膨胀系数；padding: 填充；dropout: 丢弃率。
        """
        super(TemporalBlock, self).__init__()  # 调用父类初始化
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)  # 第一层膨胀卷积
        self.chomp1 = Chomp1d(padding)  # 裁掉第一层尾部 padding
        self.relu1 = nn.ReLU()  # 第一层激活
        self.dropout1 = nn.Dropout(dropout)  # 第一层 dropout

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)  # 第二层膨胀卷积
        self.chomp2 = Chomp1d(padding)  # 裁掉第二层尾部 padding
        self.relu2 = nn.ReLU()  # 第二层激活
        self.dropout2 = nn.Dropout(dropout)  # 第二层 dropout

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 串接卷积主支路
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 通道不一致时用 1x1 卷积做残差升维
        self.relu = nn.ReLU()  # 残差相加后的最终激活

    def forward(self, x):
        """前向传播：主支路输出与残差相加后激活。"""
        out = self.net(x)  # 通过卷积主支路
        res = x if self.downsample is None else self.downsample(x)  # 通道一致直接用输入，否则升维
        return self.relu(out + res)  # 残差相加并激活


class TCNModel(nn.Module):
    """
    TCN模型
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)

    由多个膨胀系数递增的 TemporalBlock 堆叠而成，取最后时刻通道输出经全连接层回归。
    """
    def __init__(self, input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.3):
        """初始化 TCN 模型。

        参数:
            input_dim: 每步输入特征数；time_step: 时间步数。
            num_channels: 各层通道数列表；kernel_size: 卷积核大小。
            dropout: 丢弃率，默认 0.3。
        """
        super(TCNModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间步数供查询
        self.input_dim = input_dim  # 保存输入维度供查询
        
        layers = []  # 初始化层列表
        num_levels = len(num_channels)  # 层数等于通道配置长度
        for i in range(num_levels):  # 逐层构建时间块
            dilation_size = 2 ** i  # 膨胀系数按 2 的幂递增
            in_channels = input_dim if i == 0 else num_channels[i-1]  # 首层输入特征数，其余为上层通道数
            out_channels = num_channels[i]  # 本层输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]  # 添加时间块

        self.network = nn.Sequential(*layers)  # 串接所有时间块
        self.fc = nn.Linear(num_channels[-1], 1)  # 末层通道数映射到预测值

    def forward(self, x):
        """前向传播：输入序列 -> 单值预测。

        参数:
            x: (batch, time_step, input_dim) 输入序列。
        返回:
            (batch, 1) 预测张量。
        """
        # x: (batch, time_step, input_dim)
        # TCN需要 (batch, input_dim, time_step)
        x = x.transpose(1, 2)  # (batch, input_dim, time_step)  # 转置适配 Conv1d
        
        # TCN编码
        out = self.network(x)  # (batch, num_channels[-1], time_step)  # 通过 TCN 主体
        
        # 使用最后时刻的特征
        out_last = out[:, :, -1]  # (batch, num_channels[-1])  # 取最后时间步通道输出
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 全连接层输出预测


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
        self.min_delta = min_delta  # 保存提升阈值
        self.mode = mode  # 保存优化方向
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重
        self.verbose = verbose  # 保存是否打印日志

        self.counter = 0  # 初始化未提升计数
        self.best_score = None  # 初始化最佳指标为空
        self.best_epoch = 0  # 记录最佳轮次
        self.early_stop = False  # 早停标志
        self.best_model = None  # 保存最佳权重副本

    def _is_improvement(self, score):
        """判断当前指标是否刷新最佳。

        参数:
            score: 当前轮指标。
        返回:
            bool，是否提升。
        """
        if self.best_score is None:  # 首次必然视为提升
            return True

        if self.mode == 'min':  # 最小化模式
            return score < self.best_score - self.min_delta  # 需低于最佳减阈值
        elif self.mode == 'max':  # 最大化模式
            return score > self.best_score + self.min_delta  # 需高于最佳加阈值
        else:  # 非法模式
            raise ValueError("mode must be 'min' or 'max'")  # 抛错

    def __call__(self, score, model, epoch=None):
        """每轮调用，更新最佳状态并判断是否早停。

        参数:
            score: 当前轮验证指标；model: 当前模型；epoch: 轮次号，可选。
        返回:
            无返回值，通过 self.early_stop 传递信号。
        """
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 指标无效按未提升处理
            if self.verbose:  # 开启日志
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印警告
            self.counter += 1  # 累加计数
            if self.counter >= self.patience:  # 达上限
                self.early_stop = True  # 触发早停
            return  # 提前返回

        if self._is_improvement(score):  # 刷新最佳
            self.best_score = score  # 更新最佳值
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 记录最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝保存权重
            self.counter = 0  # 清零计数
            if self.verbose:  # 开启日志
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印提示
        else:  # 未提升
            self.counter += 1  # 累加计数
            if self.verbose:  # 开启日志
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印进度
            if self.counter >= self.patience:  # 达上限
                self.early_stop = True  # 触发早停

    def restore(self, model):
        """把模型权重恢复为历史最佳。

        参数:
            model: 目标模型。
        返回:
            无返回值，原地修改模型。
        """
        if self.restore_best_weights and self.best_model is not None:  # 配置恢复且保存过权重
            model.load_state_dict(self.best_model)  # 加载最佳权重
            if self.verbose:  # 开启日志
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):
    """训练模型：按轮次训练与验证，结合早停保留最佳权重。

    参数:
        model: 待训练模型；train_loader/val_loader: 训练/验证 DataLoader。
        epochs: 最大轮数；lr: 学习率；patience: 早停容忍轮数。
    返回:
        (train_losses, val_losses) 每轮训练/验证平均损失列表。
    """
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam 优化器
    early_stopping = EarlyStopping(patience=patience)  # 早停控制器

    train_losses = []  # 记录训练损失
    val_losses = []  # 记录验证损失

    for epoch in range(epochs):  # 遍历训练轮次
        # 训练阶段
        model.train()  # 切换训练模式
        train_loss = 0.0  # 累计本轮训练损失
        for batch_x, batch_y in train_loader:  # 遍历训练批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 数据搬到设备

            optimizer.zero_grad()  # 清空梯度
            outputs = model(batch_x)  # 前向预测
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()  # 累加批次损失

        train_loss /= len(train_loader)  # 求平均训练损失
        train_losses.append(train_loss)  # 记录

        # 验证阶段
        model.eval()  # 切换评估模式
        val_loss = 0.0  # 累计本轮验证损失
        with torch.no_grad():  # 关闭梯度
            for batch_x, batch_y in val_loader:  # 遍历验证批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 数据搬到设备
                outputs = model(batch_x)  # 前向预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加

        val_loss /= len(val_loader)  # 求平均验证损失
        val_losses.append(val_loss)  # 记录

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印进度

        # 早停检查
        early_stopping(val_loss, model)  # 更新早停状态
        if early_stopping.early_stop:  # 触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印触发轮次
            model.load_state_dict(early_stopping.best_model)  # 恢复最佳权重
            break  # 终止训练

    # 加载最佳模型
    if early_stopping.best_model is not None:  # 保存过最佳权重
        model.load_state_dict(early_stopping.best_model)  # 最终加载最佳权重

    return train_losses, val_losses  # 返回损失记录


# ===================== K折交叉验证训练 =====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建 5 折交叉验证器，打乱且固定种子
val_losses_all = []  # 收集每折最佳验证损失
history_list = []  # 收集每折训练/验证损失曲线
input_dim = len(features)  # 输入维度等于特征数量

print("\n========== 开始5折交叉验证（仅训练集） ==========")  # 打印交叉验证开始提示
print(f"交叉验证数据量: {len(X_train)} 样本")  # 打印参与交叉验证的样本数

for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):  # 逐折遍历划分
    print(f"\n{'=' * 50}")  # 打印分隔线
    print(f"第 {fold}/5 折训练")  # 打印当前折数
    print(f"{'=' * 50}")  # 打印分隔线

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # 按索引划分本折训练/验证输入
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # 按索引划分本折训练/验证标签

    print(f"训练集样本数: {len(X_train_fold)}")  # 打印本折训练样本数
    print(f"验证集样本数: {len(X_val_fold)}")  # 打印本折验证样本数

    X_train_tensor = torch.FloatTensor(X_train_fold)  # 本折训练输入转张量
    y_train_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)  # 本折训练标签转张量并补维度
    X_val_tensor = torch.FloatTensor(X_val_fold)  # 本折验证输入转张量
    y_val_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)  # 本折验证标签转张量并补维度

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 打包本折训练数据
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # 打包本折验证数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练数据加载器，打乱
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器，不打乱

    # 构建TCN模型（参数量与LSTM接近：~69K）
    model = TCNModel(input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.3).to(device)  # 实例化本折 TCN 模型

    if fold == 1:  # 仅第一折打印参数量
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印参数总数

    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                           epochs=200, lr=0.001, patience=10)  # 训练本折模型
    history_list.append({  # 保存本折损失曲线
        'train_losses': train_losses,  # 训练损失
        'val_losses': val_losses  # 验证损失
    })

    best_val_loss = min(val_losses)  # 取本折最低验证损失
    val_losses_all.append(best_val_loss)  # 记录本折最佳验证损失
    print(f"\n第 {fold} 折最佳验证损失: {best_val_loss:.6f}")  # 打印本折结果

print(f"\n{'=' * 50}")  # 打印分隔线
print("5折交叉验证结果汇总")  # 打印汇总标题
print(f"{'=' * 50}")  # 打印分隔线
for i, loss in enumerate(val_losses_all, 1):  # 遍历各折最佳损失
    print(f"第 {i} 折验证损失: {loss:.6f}")  # 打印每折结果
print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")  # 打印平均验证损失
print(f"验证损失标准差: {np.std(val_losses_all):.6f}")  # 打印验证损失标准差

# ===================== 绘制K折交叉验证训练过程曲线 =====================
plt.figure(figsize=(15, 10))  # 创建 15x10 英寸画布容纳四个子图

plt.subplot(2, 2, 1)  # 第 1 子图：各折训练损失
for i, history in enumerate(history_list, 1):  # 遍历各折历史
    plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制该折训练损失
plt.title('5折交叉验证 - 训练损失曲线', fontsize=14)  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 横轴标签
plt.ylabel('损失值 (MSE)')  # 纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格

plt.subplot(2, 2, 2)  # 第 2 子图：各折验证损失
for i, history in enumerate(history_list, 1):  # 遍历各折历史
    plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制该折验证损失
plt.title('5折交叉验证 - 验证损失曲线', fontsize=14)  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 横轴标签
plt.ylabel('损失值 (MSE)')  # 纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格

plt.subplot(2, 2, 3)  # 第 3 子图：各折最佳验证损失柱状图
folds = [f'Fold {i}' for i in range(1, 6)]  # 生成折标签
plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)  # 绘制各折最佳损失柱状图
plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',
            label=f'平均值: {np.mean(val_losses_all):.6f}')  # 画平均损失参考线
plt.title('各折最佳验证损失对比', fontsize=14)  # 设置标题
plt.xlabel('折数')  # 横轴标签
plt.ylabel('验证损失 (MSE)')  # 纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3, axis='y')  # 仅纵轴网格

plt.subplot(2, 2, 4)  # 第 4 子图：平均训练/验证损失曲线
max_epochs = max(len(h['train_losses']) for h in history_list)  # 取各折中最长训练轮数
avg_train_losses = []  # 收集逐轮平均训练损失
avg_val_losses = []  # 收集逐轮平均验证损失
for epoch in range(max_epochs):  # 逐轮遍历
    train_vals = [h['train_losses'][epoch] for h in history_list if epoch < len(h['train_losses'])]  # 取该轮各折训练损失
    val_vals = [h['val_losses'][epoch] for h in history_list if epoch < len(h['val_losses'])]  # 取该轮各折验证损失
    if train_vals:  # 有训练值
        avg_train_losses.append(np.mean(train_vals))  # 计算平均训练损失
    if val_vals:  # 有验证值
        avg_val_losses.append(np.mean(val_vals))  # 计算平均验证损失

plt.plot(avg_train_losses, label='平均训练损失', linewidth=2)  # 绘制平均训练损失
plt.plot(avg_val_losses, label='平均验证损失', linewidth=2)  # 绘制平均验证损失
plt.title('5折平均损失曲线', fontsize=14)  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 横轴标签
plt.ylabel('损失值 (MSE)')  # 纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格

plt.tight_layout()  # 自动紧凑布局
plt.savefig('kfold_training_TCN_pytorch.png', dpi=600, bbox_inches='tight')  # 保存交叉验证曲线图
print("\nK折交叉验证损失曲线已保存: kfold_training_TCN_pytorch.png")  # 打印保存提示

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印阶段标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试样本数

X_train_tensor = torch.FloatTensor(X_train)  # 训练序列转张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练标签转张量并补维度
X_test_tensor = torch.FloatTensor(X_test)  # 测试序列转张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试标签转张量并补维度

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 训练内部 8:2 划分位置
X_train_split = X_train_tensor[:train_val_split]  # 子训练集输入
y_train_split = y_train_tensor[:train_val_split]  # 子训练集标签
X_val_split = X_train_tensor[train_val_split:]  # 内部验证集输入
y_val_split = y_train_tensor[train_val_split:]  # 内部验证集标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 打包子训练集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 打包验证集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 打包测试集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练加载器，打乱
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试加载器（本脚本未直接使用）

final_model = TCNModel(input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.3).to(device)  # 构建最终 TCN 模型

print(final_model)  # 打印模型结构

print("\n开始训练最终模型...")  # 提示开始训练
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印参数量
print(f"训练集样本数: {len(X_train_split)}")  # 打印训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,
                                       epochs=200, lr=0.001, patience=10)  # 训练最终模型

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练完成及最低验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12x6 英寸画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失
plt.title('模型训练过程 - 损失曲线')  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 横轴标签
plt.ylabel('损失值 (MSE)')  # 纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.tight_layout()  # 紧凑布局
plt.savefig('training_loss_TCN_pytorch.png', dpi=600, bbox_inches='tight')  # 保存训练曲线
print("训练损失曲线已保存: training_loss_TCN_pytorch.png")  # 打印提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试推理

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 测试标签反归一化

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试输入转张量并搬到设备

final_model.eval()  # 切换评估模式
with torch.no_grad():  # 关闭梯度
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 推理并转回 NumPy

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 预测反归一化

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算 MSE
rmse_test = np.sqrt(mse_test)  # 计算 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果")  # 打印标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 输出 R²
print(f"测试集 RMSE: {rmse_test:.4f}")  # 输出 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 输出 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 真实值展平
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 预测值展平

test_timestamps = test_data['timestamp'].values[time_step + pred_offset - 1:]  # 按偏移截取对齐时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳与真实值对齐

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='TCN预测', linewidth=2)  # 绘制 TCN 预测曲线
plt.xlabel('时间')  # 横轴标签
plt.ylabel('温度偏差 (℃)')  # 纵轴标签
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 标题含指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.savefig('test_prediction_TCN_pytorch.png', dpi=600, bbox_inches='tight')  # 保存测试对比图
print("测试集预测结果已保存: test_prediction_TCN_pytorch.png")  # 打印提示

# ===================== 模型保存 =====================
torch.save({  # 打包保存模型与配套对象
    'model_state_dict': final_model.state_dict(),  # 模型权重
    'input_dim': input_dim,  # 输入维度
    'time_step': time_step,  # 时间窗口
    'features': features,  # 特征列表
    'feature_scaler': feature_scaler,  # 特征归一化器
    'target_scaler': target_scaler,  # 目标归一化器
}, 'Model_TCN_pytorch.pth')  # 模型文件名
print("模型已保存: Model_TCN_pytorch.pth")  # 打印保存提示

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始新数据推理
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取新数据 CSV
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟步长生成时间戳

# 归一化
new_features = pd.DataFrame(  # 构造新数据归一化特征表
    feature_scaler.transform(new_data[features]),  # 用训练集缩放器变换新数据
    columns=features  # 指定列名
)

# 保存原始目标值
y_new_original = new_data[target].values  # 提取新数据原始目标值

X_new = create_sequences(new_features, time_step)  # 构造新数据序列
y_new = y_new_original[time_step + pred_offset - 1:]  # 按偏移截取新数据标签
X_new = X_new[:len(y_new)]  # 截断输入与标签对齐

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 新数据输入转张量并搬到设备

final_model.eval()  # 切换评估模式
with torch.no_grad():  # 关闭梯度
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 推理并转回 NumPy

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 新数据预测反归一化
y_new_rescaled = y_new.reshape(-1, 1)  # 整理新数据真实标签为二维数组

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MSE
rmse_new = np.sqrt(mse_new)  # 计算新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果")  # 打印标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 输出新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 输出新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 输出新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset - 1:]  # 按偏移截取对齐的时间戳
plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 新数据真实值展平
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 新数据预测值展平
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='TCN预测', linewidth=2)  # 绘制预测曲线
plt.xlabel('时间')  # 横轴标签
plt.ylabel('温度偏差 (℃)')  # 纵轴标签
plt.title(f'新数据集温度偏差结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 标题含指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.savefig('new_data_prediction_TCN_pytorch.png', dpi=600, bbox_inches='tight')  # 保存新数据对比图
print("新数据集预测结果已保存: new_data_prediction_TCN_pytorch.png")  # 打印提示
print("\n========== 所有任务完成 ==========")  # 打印全部完成标志
