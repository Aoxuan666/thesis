"""
温度偏差预测 - 纯 LSTM（长短期记忆网络）模型完整训练与评估脚本。

本脚本完成从原始温湿度 CSV 数据读取与缺失值处理、按时间窗口构造时序序列，
到 LSTM 模型搭建、带早停机制的训练、测试集与新数据集的评估及可视化全流程。
最终把训练好的模型权重、归一化器等打包保存为 Model_LSTM_pytorch.pth，
供 lstmmodel.py 等推理脚本加载使用。
"""
import pandas as pd  # 导入 pandas 数据分析库并起别名 pd，用于 CSV 读取与表格数据处理
import numpy as np  # 导入 numpy 数值计算库并起别名 np，用于数组构造与数学运算
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于在运行前切换绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片  # Agg 后端无需图形界面，适合服务器环境离线存图
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口并起别名 plt，用于绘制损失曲线和预测对比图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，将特征与目标归一化到 0~1 区间
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R²、均方误差、平均绝对误差等回归评估指标
from sklearn.model_selection import train_test_split, KFold  # 导入随机划分与 K 折交叉验证工具（K 折相关代码当前已注释掉）
import copy  # 导入 copy 模块，用于深拷贝模型状态字典以保存最佳权重
import torch  # 导入 PyTorch 主库，用于张量计算、GPU 加速与自动求导
import torch.nn as nn  # 导入 PyTorch 神经网络模块并起别名 nn，用于搭建网络层
import torch.optim as optim  # 导入 PyTorch 优化器模块并起别名 optim，用于更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集与数据加载器，用于按批次组织训练数据

# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有可用 GPU 则用 cuda，否则回退到 cpu
print(f"使用设备: {device}")  # 打印本次训练实际使用的计算设备
if torch.cuda.is_available():  # 仅当检测到 GPU 时才输出显卡细节
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的具体型号名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印当前机器可用的 GPU 数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图默认中文字体为黑体，避免中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 关闭坐标轴负号的 Unicode 显示，防止负号显示为方块

# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的候选编码列表，兼容不同来源的 CSV
data = None  # 初始化数据变量为空，便于后续判断是否成功读取
for enc in encodings:  # 遍历每种候选编码尝试读取训练数据
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据 CSV
        print(f"成功使用 {enc} 编码加载训练数据")  # 读取成功时打印实际生效的编码
        break  # 读取成功后跳出循环，不再尝试后续编码
    except (UnicodeDecodeError, UnicodeError):  # 若当前编码解码失败则捕获异常
        continue  # 继续尝试列表中的下一种编码

if data is None:  # 所有编码都失败时数据仍为空
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 主动抛错终止，提示数据文件编码不受支持

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔为每行生成时间戳列
features = ['日龄','温度偏差','外界温度','CO2','目标温度']  # 预定义的候选特征列名称列表
target = '温度偏差'  # 指定要预测的目标变量为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列，避免缺列报错

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线，标识缺失值处理阶段开始
print("缺失值处理")  # 打印当前处理阶段标题
print(f"{'='*60}")  # 再次打印分隔线，保持输出排版统一
print(f"原始数据形状: {data.shape}")  # 输出原始数据行列数，便于对比处理前后变化
print(f"\n各列缺失值统计:")  # 提示接下来逐列统计缺失情况
for col in features + [target]:  # 遍历所有特征列及目标列
    if col in data.columns:  # 只统计切实存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占总行数的百分比
        if nan_count > 0:  # 仅对存在缺失的列输出统计结果
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印列名、缺失数量与占比

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续型变量，适合用线性插值填补
discrete_vars = ['日龄']  # 离散型变量，只做前后填充不做插值

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 列存在且确实有缺失时才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录插值前的缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 双向线性插值填补中间缺失
        data[col] = data[col].ffill()  # 用前一个有效值填充开头的剩余缺失
        data[col] = data[col].bfill()  # 用后一个有效值填充结尾的剩余缺失
        after_nan = data[col].isna().sum()  # 统计插值后仍存在的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 输出插值前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 输出插值后缺失数，验证填补效果

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 列存在且有缺失时才填充
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        data[col] = data[col].ffill()  # 用前一个有效值向后填充离散缺失
        data[col] = data[col].bfill()  # 用后一个有效值向前填充开头缺失
        after_nan = data[col].isna().sum()  # 统计填充后剩余缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 输出填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 输出填充后缺失数，验证填补效果

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 汇总所有特征列与目标列的剩余缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 提示将删除含缺失的行
    data = data.dropna(subset=features + [target])  # 丢弃特征或目标存在缺失的样本行

print(f"\n处理后数据形状: {data.shape}")  # 输出缺失值处理完成后数据的行列数
print(f"{'='*60}\n")  # 打印分隔线，标识缺失值处理阶段结束

print(f"\n【纯LSTM模型】")  # 打印当前使用的模型名称标题
print(f"使用特征: {features}")  # 列出最终参与建模的特征
print(f"特征数量: {len(features)}")  # 打印特征个数

# ===================== 数据集划分 =====================
time_step = 36  # LSTM时间窗口：48步 × 5分钟 = 4小时  # 每个输入样本回看的历史时间步数
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟  # 标签相对输入窗口末尾向未来偏移的步数

print(f"\n时间窗口配置:")  # 提示接下来输出时间窗口参数
print(f"  LSTM时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印输入窗口长度及对应分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移步数及对应分钟数

# 划分数据集
split_index = int(len(data) * 0.8)  # 按 8:2 比例计算训练/测试划分的索引位置
train_data = data.iloc[:split_index].copy()  # 取前 80% 作为训练集（副本，避免视图修改）
test_data = data.iloc[split_index:].copy()  # 取后 20% 作为测试集（副本）

print(f"\n数据集划分:")  # 提示接下来输出划分结果
print(f"训练集大小: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")  # 打印训练集样本数及占比
print(f"测试集大小: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")  # 打印测试集样本数及占比

# ===================== 归一化 =====================
print("\n归一化处理...")  # 提示开始归一化处理
feature_scaler = MinMaxScaler()  # 实例化特征归一化器
target_scaler = MinMaxScaler()  # 实例化目标归一化器

# 只在训练集上fit
feature_scaler.fit(train_data[features])  # 仅用训练集特征拟合缩放范围，防止测试集信息泄漏
target_scaler.fit(train_data[[target]])  # 仅用训练集目标拟合缩放范围

# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 构造归一化后的训练特征 DataFrame
    feature_scaler.transform(train_data[features]),  # 用训练集拟合的缩放器变换训练特征
    columns=features  # 指定列名，保持与原始特征一致
)
test_features = pd.DataFrame(  # 构造归一化后的测试特征 DataFrame
    feature_scaler.transform(test_data[features]),  # 用同一个缩放器变换测试特征
    columns=features  # 指定列名，保持与原始特征一致
)

train_target = target_scaler.transform(train_data[[target]]).flatten()  # 归一化训练目标并展平为一维数组
test_target = target_scaler.transform(test_data[[target]]).flatten()  # 归一化测试目标并展平为一维数组

# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):
    """将二维表格数据滑窗切分为三维时序样本。

    参数:
        data: 归一化后的特征 DataFrame（行=时间，列=特征）。
        time_step: 每个样本回看的历史时间步数。
    返回:
        np.ndarray，形状为 (样本数, time_step, 特征数) 的序列数组。
    """
    sequences = []  # 初始化列表，用于收集每个时间窗口
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，预留窗口长度与预测偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 [i, i+time_step) 的窗口并转为 NumPy 数组
    return np.array(sequences)  # 将列表堆叠为三维数组并返回

print("\n创建训练集序列...")  # 提示开始构造训练序列
X_train = create_sequences(train_features, time_step)  # 由训练特征生成训练输入序列
y_train = train_target[time_step + pred_offset - 1:]  # 标签取窗口末尾再偏移 pred_offset 步的位置
y_train = y_train[:len(X_train)]  # 截断标签使其与输入序列数量一一对齐

print("\n创建测试集序列...")  # 提示开始构造测试序列
X_test = create_sequences(test_features, time_step)  # 由测试特征生成测试输入序列
y_test = test_target[time_step + pred_offset - 1:]  # 标签按相同偏移规则截取
y_test = y_test[:len(X_test)]  # 截断标签使其与测试输入序列数量对齐

print(f"\n序列数据形状:")  # 提示接下来输出各序列的形状
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集输入与标签的形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集输入与标签的形状

# ===================== 纯LSTM模型定义 =====================
class LSTMModel(nn.Module):
    """
    纯LSTM模型
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)

    由单层（或多层）单向 LSTM 加一个全连接输出层构成，
    取序列最后一个时刻的隐藏状态作为整段序列的表征来回归预测目标值。
    """
    def __init__(self, input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.3):
        """初始化 LSTM 网络结构。

        参数:
            input_dim: 每个时间步的输入特征数。
            time_step: 输入序列的时间步数（仅记录，不改变结构）。
            lstm_hidden: LSTM 隐藏单元数，默认 128。
            lstm_layers: LSTM 堆叠层数，默认 1；多层时层间加 dropout。
            dropout: 多层时的 dropout 比例。
        """
        super(LSTMModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间步数供外部查询
        self.input_dim = input_dim  # 保存输入特征维度供外部查询
        self.lstm_hidden = lstm_hidden  # 保存隐藏层维度供外部查询
        
        # LSTM层
        self.lstm = nn.LSTM(  # 定义 LSTM 循环层
            input_size=input_dim,  # 每个时间步输入的特征维度
            hidden_size=lstm_hidden,  # 隐藏状态维度
            num_layers=lstm_layers,  # LSTM 堆叠层数
            dropout=dropout if lstm_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 输入张量的第 0 维为 batch
            bidirectional=False  # 使用单向 LSTM
        )
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden, 1)  # 将隐藏状态映射到单个预测值

    def forward(self, x):
        """前向传播：输入序列 -> 预测值。

        参数:
            x: 形状 (batch, time_step, input_dim) 的输入序列张量。
        返回:
            形状 (batch, 1) 的预测张量。
        """
        # x: (batch, time_step, input_dim)
        # LSTM编码
        out, (h, c) = self.lstm(x)  # out: (batch, time_step, hidden)  # 运行 LSTM 得到每步输出与隐藏/细胞状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)  # 取序列最后一个时间步的输出作为整段表征
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 经全连接层得到最终单值预测


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
        self.mode = mode  # 保存优化方向模式
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重的开关
        self.verbose = verbose  # 保存是否打印日志的开关

        self.counter = 0  # 初始化未提升计数器
        self.best_score = None  # 初始化最佳指标值为空
        self.best_epoch = 0  # 记录出现最佳指标时的轮次
        self.early_stop = False  # 早停触发标志，初始为否
        self.best_model = None  # 用于存放最佳模型权重副本

    def _is_improvement(self, score):
        """判断当前指标相对历史最佳是否有实质性提升。

        参数:
            score: 当前轮的验证指标。
        返回:
            bool，True 表示刷新了最佳值。
        """
        if self.best_score is None:  # 尚无历史最佳时，第一次必定视为提升
            return True

        if self.mode == 'min':  # 最小化模式：越小越好
            return score < self.best_score - self.min_delta  # 需小于历史最佳减去阈值才算提升
        elif self.mode == 'max':  # 最大化模式：越大越好
            return score > self.best_score + self.min_delta  # 需大于历史最佳加上阈值才算提升
        else:  # 传入了非法模式
            raise ValueError("mode must be 'min' or 'max'")  # 抛出取值错误

    def __call__(self, score, model, epoch=None):
        """每轮训练后调用，更新最佳状态并决定是否触发早停。

        参数:
            score: 当前轮验证指标。
            model: 当前模型，指标提升时会被深拷贝保存。
            epoch: 当前轮次号，可选。
        返回:
            无返回值，通过 self.early_stop 传递早停信号。
        """
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 指标无效时按一次"未提升"处理
            if self.verbose:  # 开启日志时
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印无效指标警告
            self.counter += 1  # 无效指标同样累加未提升计数
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 提前返回，不参与后续比较

        if self._is_improvement(score):  # 当前指标刷新最佳
            self.best_score = score  # 更新历史最佳值
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 记录最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝并保存此时的模型权重
            self.counter = 0  # 清零未提升计数
            if self.verbose:  # 开启日志时
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印最佳值更新提示
        else:  # 指标未提升
            self.counter += 1  # 累加未提升计数
            if self.verbose:  # 开启日志时
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印当前计数进度
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停

    def restore(self, model):
        """把模型权重恢复为历史最佳状态。

        参数:
            model: 需要被恢复权重的目标模型。
        返回:
            无返回值，直接原地修改传入模型。
        """
        if self.restore_best_weights and self.best_model is not None:  # 配置了恢复且确实保存过最佳权重
            model.load_state_dict(self.best_model)  # 将最佳权重加载回模型
            if self.verbose:  # 开启日志时
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):
    """训练模型：按轮次执行训练与验证，并结合早停保留最佳权重。

    参数:
        model: 待训练的 PyTorch 模型。
        train_loader: 训练数据 DataLoader。
        val_loader: 验证数据 DataLoader。
        epochs: 最大训练轮数，默认 200。
        lr: Adam 优化器学习率，默认 0.001。
        patience: 早停容忍轮数，默认 10。
    返回:
        (train_losses, val_losses) 两个列表，分别为每轮训练/验证平均损失。
    """
    criterion = nn.MSELoss()  # 使用均方误差作为回归损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新全部模型参数
    early_stopping = EarlyStopping(patience=patience)  # 实例化早停控制器（最小化验证损失）

    train_losses = []  # 记录每轮训练平均损失
    val_losses = []  # 记录每轮验证平均损失

    for epoch in range(epochs):  # 外层循环：遍历训练轮次
        # 训练阶段
        model.train()  # 切换模型到训练模式（启用 dropout 等）
        train_loss = 0.0  # 累计当前轮训练损失
        for batch_x, batch_y in train_loader:  # 遍历训练数据每个批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将批次数据搬到计算设备

            optimizer.zero_grad()  # 清空上一轮累积的梯度
            outputs = model(batch_x)  # 前向传播得到预测
            loss = criterion(outputs, batch_y)  # 计算预测与真实之间的损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据梯度更新模型参数

            train_loss += loss.item()  # 累加当前批次的损失值

        train_loss /= len(train_loader)  # 求当前轮训练的平均损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换模型到评估模式（关闭 dropout 等）
        val_loss = 0.0  # 累计当前轮验证损失
        with torch.no_grad():  # 验证阶段关闭梯度计算以节省显存
            for batch_x, batch_y in val_loader:  # 遍历验证数据每个批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将批次数据搬到计算设备
                outputs = model(batch_x)  # 前向传播得到预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加当前批次验证损失

        val_loss /= len(val_loader)  # 求当前轮验证的平均损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次进度
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 输出轮次与训练/验证损失

        # 早停检查
        early_stopping(val_loss, model)  # 用验证损失更新早停状态
        if early_stopping.early_stop:  # 若已触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停发生的轮次
            model.load_state_dict(early_stopping.best_model)  # 恢复最佳权重
            break  # 终止训练循环

    # 加载最佳模型
    if early_stopping.best_model is not None:  # 若保存过最佳权重
        model.load_state_dict(early_stopping.best_model)  # 训练结束后最终加载最佳权重

    return train_losses, val_losses  # 返回训练与验证损失记录


# ===================== K折交叉验证训练 =====================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
# input_dim = len(features)
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
#     # 构建纯LSTM模型
#     model = LSTMModel(input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.3).to(device)
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
# plt.savefig('kfold_training_LSTM_pytorch.png', dpi=600, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_LSTM_pytorch.png")

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线，标识最终模型训练阶段开始
print("训练最终模型（使用全部训练数据）")  # 打印阶段标题
print(f"{'=' * 50}")  # 再次打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 输出用于训练的序列样本数量
print(f"测试集样本数: {len(X_test)}")  # 输出测试序列样本数量

X_train_tensor = torch.FloatTensor(X_train)  # 将训练序列转为 float 张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练标签转张量并补出特征维度 (N,1)
X_test_tensor = torch.FloatTensor(X_test)  # 将测试序列转为 float 张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试标签转张量并补出特征维度 (N,1)

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 计算训练集内部 8:2 再划分的位置
X_train_split = X_train_tensor[:train_val_split]  # 内部划分后的子训练集输入
y_train_split = y_train_tensor[:train_val_split]  # 内部划分后的子训练集标签
X_val_split = X_train_tensor[train_val_split:]  # 内部划分后的验证集输入
y_val_split = y_train_tensor[train_val_split:]  # 内部划分后的验证集标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 将子训练集打包为 TensorDataset
val_dataset = TensorDataset(X_val_split, y_val_split)  # 将验证集打包为 TensorDataset
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 将测试集打包为 TensorDataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集按 32 批量并打乱顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证集按 32 批量且不打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试集按 32 批量且不打乱（本脚本未直接使用）

final_model = LSTMModel(input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.3).to(device)  # 构建最终 LSTM 模型并搬到计算设备

print(final_model)  # 打印模型结构概览

print("\n开始训练最终模型...")  # 提示训练开始
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 统计并打印模型可训练参数总数
print(f"训练集样本数: {len(X_train_split)}")  # 打印实际参与训练的样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印用于早停验证的样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,
                                       epochs=200, lr=0.001, patience=10)  # 调用训练函数，最多 200 轮，早停耐心 10

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练结束及最低验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12x6 英寸画布用于绘制损失曲线
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失随轮次变化曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失随轮次变化曲线
plt.title('模型训练过程 - 损失曲线')  # 设置图表标题
plt.xlabel('训练轮数 (Epoch)')  # 设置横轴标签
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启半透明网格线
plt.tight_layout()  # 自动紧凑布局，避免标签被截断
plt.savefig('training_loss_LSTM_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 DPI 保存训练损失曲线
print("训练损失曲线已保存: training_loss_LSTM_pytorch.png")  # 打印保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集推理

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将归一化后的测试标签反归一化回原始量纲

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试输入转为张量并搬到计算设备

final_model.eval()  # 切换最终模型到评估模式
with torch.no_grad():  # 关闭梯度计算进行推理
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 得到测试集预测并转回 CPU 上的 NumPy 数组

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测值反归一化回原始量纲

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算测试集均方误差
rmse_test = np.sqrt(mse_test)  # 对均方误差开方得到 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算测试集平均绝对误差

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 输出测试集 R²
print(f"测试集 RMSE: {rmse_test:.4f}")  # 输出测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 输出测试集 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布用于绘制预测对比图
y_test_rescaled_flat = y_test_rescaled.flatten()  # 将真实值展平为一维
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 将预测值展平为一维

test_timestamps = test_data['timestamp'].values[time_step + pred_offset - 1:]  # 按偏移截取与标签对齐的测试时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳使其与真实值数量一致

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实温度偏差曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='LSTM预测', linewidth=2)  # 绘制 LSTM 预测温度偏差曲线
plt.xlabel('时间')  # 设置横轴标签为时间
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签为温度偏差
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置标题并附带评估指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启半透明网格线
plt.savefig('test_prediction_LSTM_pytorch.png', dpi=600, bbox_inches='tight')  # 保存测试集预测对比图
print("测试集预测结果已保存: test_prediction_LSTM_pytorch.png")  # 打印保存提示

# ===================== 模型保存 =====================
torch.save({  # 将模型及配套对象打包保存到磁盘
    'model_state_dict': final_model.state_dict(),  # 保存最终模型权重
    'input_dim': input_dim,  # 保存输入特征维度，供加载时重建模型
    'time_step': time_step,  # 保存时间窗口长度
    'features': features,  # 保存特征名称列表
    'feature_scaler': feature_scaler,  # 保存特征归一化器，保证推理口径一致
    'target_scaler': target_scaler,  # 保存目标归一化器，保证推理口径一致
}, 'Model_LSTM_pytorch.pth')  # 指定模型文件名
print("模型已保存: Model_LSTM_pytorch.pth")  # 打印保存成功提示

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始处理独立的新数据集
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取新采集的温湿度数据 CSV
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟步长、指定起始时刻生成时间戳

# 归一化
new_features = pd.DataFrame(  # 构造新数据的归一化特征 DataFrame
    feature_scaler.transform(new_data[features]),  # 用训练集拟合的缩放器变换新数据特征
    columns=features  # 指定列名保持特征一致
)

# 保存原始目标值
y_new_original = new_data[target].values  # 提取新数据未经归一化的原始目标值

X_new = create_sequences(new_features, time_step)  # 构造新数据的时间序列输入
y_new = y_new_original[time_step + pred_offset - 1:]  # 按相同偏移规则截取新数据标签
X_new = X_new[:len(y_new)]  # 截断输入序列使其与标签数量对齐

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 将新数据输入转为张量并搬到计算设备

final_model.eval()  # 切换模型到评估模式
with torch.no_grad():  # 关闭梯度计算进行推理
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 得到新数据预测并转回 NumPy 数组

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 将新数据预测反归一化
y_new_rescaled = y_new.reshape(-1, 1)  # 将新数据真实标签整理为二维数组

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据均方误差
rmse_new = np.sqrt(mse_new)  # 计算新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 输出新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 输出新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 输出新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset - 1:]  # 按偏移截取与新数据标签对齐的时间戳
plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布用于绘制新数据对比图
y_new_rescaled_flat = y_new_rescaled.flatten()  # 将新数据真实值展平
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 将新数据预测值展平
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制新数据真实曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='LSTM预测', linewidth=2)  # 绘制新数据预测曲线
plt.xlabel('时间')  # 设置横轴标签为时间
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签为温度偏差
plt.title(f'新数据集温度偏差结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置标题并附带指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启半透明网格线
plt.savefig('new_data_prediction_LSTM_pytorch.png', dpi=600, bbox_inches='tight')  # 保存新数据预测对比图
print("新数据集预测结果已保存: new_data_prediction_LSTM_pytorch.png")  # 打印保存提示
print("\n========== 所有任务完成 ==========")  # 打印全部任务结束标志
