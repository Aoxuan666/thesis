"""
纯 GRU 模型训练脚本（THI 温湿指数预测）。

本脚本完成端到端建模：多编码读取温湿度 CSV、按时间生成时间戳、缺失值填补、
按时间顺序 6:4 划分训练/测试集、仅用训练集拟合 MinMax 归一化、构造带预测偏移的时间序列样本、
构建并训练 GRU 模型（带通用早停类）、在测试集与新数据上评估并绘图、保存模型检查点。
文件下半部分保留了一段被注释掉的 K 折交叉验证实验代码，可按需启用。
"""
import pandas as pd  # 导入 pandas 库，用于表格数据读取与处理（别名 pd）
import numpy as np  # 导入 NumPy 库，用于数组与数值计算（别名 np）
import random  # 导入标准库 random，配合随机种子设置保证可复现（当前种子代码被注释）
import os  # 导入标准库 os，用于设置环境变量 PYTHONHASHSEED
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片（无界面环境也能出图）
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口（别名 plt）
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于归一化特征与目标值
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R²、均方误差、平均绝对误差
from sklearn.model_selection import train_test_split, KFold  # 导入数据集划分与 K 折交叉验证工具（K 折代码当前被注释）
import copy  # 导入标准库 copy，用于深拷贝模型权重实现早停
import torch  # 导入 PyTorch 主库，用于张量计算与模型训练
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU 与全连接层
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据加载工具，按批次组织训练/验证/测试数据
# ===================== 随机种子设置（确保可复现） =====================
# def set_seed(seed=42):
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
# #10是85
# set_seed(99)
# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU
print(f"使用设备: {device}")  # 打印使用的设备
if torch.cuda.is_available():  # 若有 GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印 GPU 型号
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印 GPU 数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的编码列表
data = None  # 初始化读取结果变量
for enc in encodings:  # 逐个编码尝试
    try:  # 捕获解码异常
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功即跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 换下一个编码继续尝试
if data is None:  # 所有编码都失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止
data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按起始时间与 5 分钟频率生成时间戳列
features = ['日龄', 'THI', '当前温度', '目标温度']  # 候选特征列表
target = 'THI'  # 目标变量为 THI
features = [f for f in features if f in data.columns]  # 过滤掉数据中不存在的特征
# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印分隔线
print("缺失值处理")  # 打印处理标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前数据行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计表头
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅处理数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比（百分比）
        if nan_count > 0:  # 仅打印存在缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与占比
# 定义连续变量和离散变量（两类采用不同填充策略）
continuous_vars = ['THI','目标温度', '当前温度']  # 连续变量：线性插值 + 前后填充
discrete_vars = ['日龄']  # 离散变量：前后填充
# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 该列存在且有缺失时才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录处理前缺失值个数
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值，双向填充内部缺失
        data[col] = data[col].ffill()  # 前向填充开头残留缺失
        data[col] = data[col].bfill()  # 后向填充结尾残留缺失
        after_nan = data[col].isna().sum()  # 记录处理后缺失值个数
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后缺失数
# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 该列存在且有缺失时才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失值个数
        data[col] = data[col].fillna(method='ffill')  # 前向填充
        data[col] = data[col].fillna(method='bfill')  # 后向填充
        after_nan = data[col].isna().sum()  # 记录填充后缺失值个数
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数
# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列剩余缺失总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除特征/目标中含缺失值的行
print(f"\n处理后数据形状: {data.shape}")  # 打印处理后数据行列数
print(f"{'=' * 60}\n")  # 打印结束分隔线
print(f"\n【纯GRU模型】")  # 打印模型标题
print(f"使用特征: {features}")  # 打印实际使用的特征
print(f"特征数量: {len(features)}")  # 打印特征数量
# ===================== 数据集划分 =====================
time_step = 36  # GRU时间窗口：48步 × 5分钟 = 4小时（此版本实际使用 36 步窗口）
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟，预测目标相对窗口末尾的偏移
print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印 GRU 窗口对应时长
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移对应时长
# 划分数据集
split_index = int(len(data) * 0.6)  # 按 6:4 划分，计算切分位置
train_data = data.iloc[:split_index].copy()  # 前 60% 作为训练集
test_data = data.iloc[split_index:].copy()  # 后 40% 作为测试集
print(f"\n数据集划分:")  # 打印划分结果标题
print(f"训练集大小: {len(train_data)} ({len(train_data) / len(data) * 100:.1f}%)")  # 打印训练集样本数与占比
print(f"测试集大小: {len(test_data)} ({len(test_data) / len(data) * 100:.1f}%)")  # 打印测试集样本数与占比
# ===================== 归一化 =====================
print("\n归一化处理...")  # 提示开始归一化
feature_scaler = MinMaxScaler()  # 特征归一化器
target_scaler = MinMaxScaler()  # 目标归一化器
# 只在训练集上fit
feature_scaler.fit(train_data[features])  # 仅在训练集上拟合特征归一化器
target_scaler.fit(train_data[[target]])  # 仅在训练集上拟合目标归一化器
# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 变换训练集特征
    feature_scaler.transform(train_data[features]),  # 归一化数值
    columns=features  # 保持列名
)
test_features = pd.DataFrame(  # 变换测试集特征
    feature_scaler.transform(test_data[features]),  # 用训练集归一化器变换
    columns=features  # 保持列名
)
train_target = target_scaler.transform(train_data[[target]]).flatten()  # 训练集目标归一化并拉平
test_target = target_scaler.transform(test_data[[target]]).flatten()  # 测试集目标归一化并拉平
# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):  # 构造输入序列：用 [i, i+time_step-1] 预测 i+time_step+pred_offset-1
    """
    创建时间序列样本。

    参数:
        data: 特征表（行代表时间步）。
        time_step: 输入窗口长度。
    返回:
        ndarray: 形状为 (样本数, time_step, 特征数) 的三维数组。
    说明:
        预测目标位置由全局变量 pred_offset 决定。
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，保证目标位置不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行作为一个样本
    return np.array(sequences)  # 转换为三维数组返回
print("\n创建训练集序列...")  # 提示创建训练集序列
X_train = create_sequences(train_features, time_step)  # 构造训练集输入序列
y_train = train_target[time_step + pred_offset - 1:]  # 取与输入序列对齐的训练目标
y_train = y_train[:len(X_train)]  # 保证目标长度与样本数一致
print("\n创建测试集序列...")  # 提示创建测试集序列
X_test = create_sequences(test_features, time_step)  # 构造测试集输入序列
y_test = test_target[time_step + pred_offset - 1:]  # 取与输入序列对齐的测试目标
y_test = y_test[:len(X_test)]  # 保证目标长度与样本数一致
print(f"\n序列数据形状:")  # 打印序列形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集序列与目标形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集序列与目标形状
# ===================== 纯GRU模型定义 =====================
class GRUModel(nn.Module):  # 纯 GRU 预测模型
    """
    纯GRU模型
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)

    参数:
        input_dim: 输入特征维度。
        time_step: 时间窗口长度。
        gru_hidden: GRU 隐藏层维度。
        gru_layers: GRU 层数。
        dropout: 多层 GRU 时的层间丢弃率（单层时自动置 0）。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):  # 初始化模型结构
        super(GRUModel, self).__init__()  # 调用父类构造函数
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层维度

        # GRU层
        self.gru = nn.GRU(  # 定义 GRU 层
            input_size=input_dim,  # 输入维度
            hidden_size=gru_hidden,  # 隐藏维度
            num_layers=gru_layers,  # GRU 层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 输入形状为 (batch, seq, feature)
            bidirectional=False  # 单向 GRU
        )
        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层将隐藏状态映射为 1 个预测值
    def forward(self, x):  # 前向传播：GRU 编码 -> 取末时刻 -> 全连接
        # x: (batch, time_step, input_dim)
        # GRU编码
        out, h = self.gru(x)  # out: (batch, time_step, hidden)，h 为最终隐藏状态

        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)，取序列最后一个时间步输出

        # 预测输出
        return self.fc(out_last)  # (batch, 1)，经全连接层得到预测
class EarlyStopping:  # 通用早停策略，支持最小化/最大化指标，可恢复最佳权重
    """通用早停策略，默认用于最小化验证损失。"""
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
        self.min_delta = min_delta  # 保存最小改善阈值
        self.mode = mode  # 保存优化方向（min/max）
        self.restore_best_weights = restore_best_weights  # 是否恢复最佳权重标志
        self.verbose = verbose  # 是否打印日志标志

        self.counter = 0  # 未改善计数器
        self.best_score = None  # 记录最佳指标
        self.best_epoch = 0  # 记录最佳轮次
        self.early_stop = False  # 是否触发早停标志
        self.best_model = None  # 保存最佳模型权重副本
    def _is_improvement(self, score):  # 判断当前指标是否优于历史最佳
        if self.best_score is None:  # 首次评估，视为有提升
            return True  # 直接接受该指标

        if self.mode == 'min':  # 越小越好的场景（如损失）
            return score < self.best_score - self.min_delta  # 需下降超过 min_delta 才算提升
        elif self.mode == 'max':  # 越大越好的场景（如 R²）
            return score > self.best_score + self.min_delta  # 需上升超过 min_delta 才算提升
        else:  # mode 取值非法
            raise ValueError("mode must be 'min' or 'max'")  # 抛出异常
    def __call__(self, score, model, epoch=None):  # 每轮验证后调用，更新最佳或累计未改善计数
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 指标非法（空值或非有限数）
            if self.verbose:  # 允许打印时
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印警告
            self.counter += 1  # 未改善计数加一
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 提前返回，不参与正常比较
        if self._is_improvement(score):  # 本轮指标有提升
            self.best_score = score  # 更新最佳指标
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 更新最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝当前模型权重
            self.counter = 0  # 重置未改善计数
            if self.verbose:  # 允许打印时
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印提升信息
        else:  # 本轮无提升
            self.counter += 1  # 未改善计数加一
            if self.verbose:  # 允许打印时
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印计数状态
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
    def restore(self, model):  # 将模型恢复到记录的最佳权重
        if self.restore_best_weights and self.best_model is not None:  # 需要恢复且存在最佳权重
            model.load_state_dict(self.best_model)  # 加载最佳权重
            if self.verbose:  # 允许打印时
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息
def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 训练模型：带验证与早停，返回训练/验证损失曲线
    """
    训练模型。

    参数:
        model: 待训练模型。
        train_loader / val_loader: 训练与验证数据加载器。
        epochs: 最大训练轮数。
        lr: Adam 优化器学习率。
        patience: 早停容忍轮数。
    返回:
        (train_losses, val_losses): 每轮训练与验证平均损失列表。
    """
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam 优化器
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器
    train_losses = []  # 记录每轮训练损失
    val_losses = []  # 记录每轮验证损失

    for epoch in range(epochs):  # 按轮次训练
        # 训练阶段
        model.train()  # 切换训练模式
        train_loss = 0.0  # 本轮累计训练损失
        for batch_x, batch_y in train_loader:  # 遍历训练批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 批次数据送到设备

            optimizer.zero_grad()  # 清零上一轮梯度
            outputs = model(batch_x)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()  # 累加批次损失

        train_loss /= len(train_loader)  # 求训练损失均值
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换评估模式
        val_loss = 0.0  # 本轮累计验证损失
        with torch.no_grad():  # 关闭梯度计算
            for batch_x, batch_y in val_loader:  # 遍历验证批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 批次数据送到设备
                outputs = model(batch_x)  # 前向预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加批次损失

        val_loss /= len(val_loader)  # 求验证损失均值
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次日志
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前轮次与损失

        # 早停检查
        early_stopping(val_loss, model, epoch=epoch + 1)  # 调用早停检查并传入轮次
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印触发轮次
            early_stopping.restore(model)  # 恢复最佳权重
            break  # 跳出训练循环

    # 加载最佳模型
    early_stopping.restore(model)  # 训练结束后确保模型处于最佳状态

    return train_losses, val_losses  # 返回损失记录
# ===================== K折交叉验证训练 =====================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(features)  # 输入维度等于特征数

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
#     X_train_tensor = torch.FloatTensor(X_train_fold)
#     y_train_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)
#     X_val_tensor = torch.FloatTensor(X_val_fold)
#     y_val_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
#     # 构建纯GRU模型
#     model = GRUModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2).to(device)
#
#     if fold == 1:
#         print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
#
#     train_losses, val_losses = train_model(model, train_loader, val_loader,
#                                            epochs=200, lr=0.001, patience=10)
#     history_list.append({
#         'train_losses': train_losses,
#         'val_losses': val_losses
#     })
#
#     best_val_loss = min(val_losses)
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
# plt.title('5折交叉验证 - 训练损失曲线', fontsize=14)
# plt.xlabel('训练轮数 (Epoch)')
# plt.ylabel('损失值 (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 2)
# for i, history in enumerate(history_list, 1):
#     plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)
# plt.title('5折交叉验证 - 验证损失曲线', fontsize=14)
# plt.xlabel('训练轮数 (Epoch)')
# plt.ylabel('损失值 (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 3)
# folds = [f'Fold {i}' for i in range(1, 6)]
# plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)
# plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',
#             label=f'平均值: {np.mean(val_losses_all):.6f}')
# plt.title('各折最佳验证损失对比', fontsize=14)
# plt.xlabel('折数')
# plt.ylabel('验证损失 (MSE)')
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
#
# plt.plot(avg_train_losses, label='平均训练损失', linewidth=2)
# plt.plot(avg_val_losses, label='平均验证损失', linewidth=2)
# plt.title('5折平均损失曲线', fontsize=14)
# plt.xlabel('训练轮数 (Epoch)')
# plt.ylabel('损失值 (MSE)')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('kfold_training_GRU_pytorch.png', dpi=150, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRU_pytorch.png")
# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印训练标题
print(f"{'=' * 50}")  # 再次打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试样本数
X_train_tensor = torch.FloatTensor(X_train)  # 训练集输入转张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练目标转张量并增加特征维
X_test_tensor = torch.FloatTensor(X_test)  # 测试集输入转张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试目标转张量并增加特征维

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 训练集再按 8:2 切出验证集
X_train_split = X_train_tensor[:train_val_split]  # 实际训练输入
y_train_split = y_train_tensor[:train_val_split]  # 实际训练目标
X_val_split = X_train_tensor[train_val_split:]  # 验证输入
y_val_split = y_train_tensor[train_val_split:]  # 验证目标

train_dataset = TensorDataset(X_train_split, y_train_split)  # 封装训练数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 封装验证数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 封装测试数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练数据加载器，打乱顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试数据加载器
final_model = GRUModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2).to(device)  # 构建最终模型并移至设备
print(final_model)  # 打印模型结构
print("\n开始训练最终模型...")  # 打印训练开始提示
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印参数总量
print(f"训练集样本数: {len(X_train_split)}")  # 打印训练样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,  # 调用训练函数
                                       epochs=200, lr=0.001, patience=10)  # 最大轮数 200、学习率 0.001、早停容忍 10 轮

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建损失曲线画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 设置横轴标签
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.tight_layout()  # 紧凑布局
plt.savefig('training_loss_GRU_pytorch.png', dpi=150, bbox_inches='tight')  # 保存损失曲线图
print("训练损失曲线已保存: training_loss_GRU_pytorch.png")  # 打印保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 测试真实目标反归一化到原始尺度

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试输入转张量并送设备

final_model.eval()  # 切换评估模式
with torch.no_grad():  # 关闭梯度计算
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 前向预测并转回 NumPy

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 预测值反归一化到原始尺度

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差
rmse_test = np.sqrt(mse_test)  # 均方误差开方得 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果")  # 打印评估标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印 R²
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印 MAE
print(f"{'=' * 60}")  # 打印分隔线
print(f"{'=' * 60}")  # 再次打印分隔线

plt.figure(figsize=(15, 5))  # 创建预测对比图画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 真实值拉平为一维
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 预测值拉平为一维

test_timestamps = test_data['timestamp'].values[time_step + pred_offset - 1:]  # 目标位置对应的时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 与真实值长度对齐

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实THI', linewidth=2)  # 绘制真实 THI 曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU预测', linewidth=2)  # 绘制预测 THI 曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('THI ')  # 设置纵轴标签
plt.title(f'测试集THI结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置标题并附带指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.savefig('test_prediction_GRU_pytorch.png', dpi=150, bbox_inches='tight')  # 保存预测图
print("测试集预测结果已保存: test_prediction_GRU_pytorch.png")  # 打印保存提示

# ===================== 模型保存 =====================
torch.save({  # 保存模型权重与配置到检查点文件
    'model_state_dict': final_model.state_dict(),  # 模型权重
    'input_dim': input_dim,  # 输入维度
    'time_step': time_step,  # 时间窗口
    'features': features,  # 特征列表
    'feature_scaler': feature_scaler,  # 特征归一化器
    'target_scaler': target_scaler,  # 目标归一化器
}, 'Model_GRU_pytorch.pth')  # 检查点保存路径
print("模型已保存:Model_GRU_pytorch.pth")  # 打印保存提示

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始处理新数据
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取新数据 CSV
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',  # 按索引×5分钟生成时间戳（分钟为单位）
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 时间起点

# 归一化
new_features = pd.DataFrame(  # 用训练集归一化器变换新数据特征
    feature_scaler.transform(new_data[features]),  # 归一化数值
    columns=features  # 保持列名
)

# 保存原始目标值
y_new_original = new_data[target].values  # 新数据目标原始值

X_new = create_sequences(new_features, time_step)  # 构造新数据输入序列
y_new = y_new_original[time_step + pred_offset:]  # 取与新序列对齐的原始目标
X_new = X_new[:len(y_new)]  # 保证输入序列与目标长度一致

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 新数据输入转张量并送设备

final_model.eval()  # 切换评估模式
with torch.no_grad():  # 关闭梯度计算
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 前向预测并转回 NumPy

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 预测值反归一化到原始尺度
y_new_rescaled = y_new.reshape(-1, 1)  # 真实目标调整为列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据决定系数 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据均方误差
rmse_new = np.sqrt(mse_new)  # 均方误差开方得 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据平均绝对误差

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果")  # 打印评估标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 打印新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线
timestamps = new_data['timestamp'][time_step + pred_offset:]  # 新数据目标位置时间戳（Series）
plt.figure(figsize=(15, 5))  # 创建新数据预测对比画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 真实值拉平为一维
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 预测值拉平为一维
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实THI', linewidth=2)  # 绘制真实 THI 曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='GRU预测', linewidth=2)  # 绘制预测 THI 曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('THI ')  # 设置纵轴标签
plt.title(f'新数据集THI结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置标题并附带指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.savefig('new_data_prediction_GRU_pytorch.png', dpi=150, bbox_inches='tight')  # 保存新数据预测图
print("新数据集预测结果已保存: new_data_prediction_GRU_pytorch.png")  # 打印保存提示
print("\n========== 所有任务完成 ==========")  # 打印全部完成提示
