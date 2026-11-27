"""
纯 GRU 模型实验脚本：基于温湿度历史数据预测“温度偏差”。

主要流程：
1. 读取 CSV 数据，构造时间戳并对缺失值做插值与前后向填充；
2. 按 8:2 划分训练集与测试集，分别做 MinMax 归一化；
3. 用滑动窗口把二维表格转换为监督学习所需的时序样本；
4. 定义 GRU 回归模型、早停机制与通用训练函数；
5. 从训练集中再划出 20% 作验证集训练最终模型，并绘制损失曲线；
6. 在测试集与新数据集上评估 R2/RMSE/MAE，绘制预测对比图；
7. 把模型结构、参数与归一化器保存为 Model_GRU_pytorch.pth。

文件中还保留了一段被整体注释掉的 5 折交叉验证实验代码，便于需要时恢复使用。
"""
import pandas as pd  # 导入 pandas，用于读取 CSV 并进行表格数据处理
import numpy as np  # 导入 numpy，用于数组运算与统计指标计算
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，以便设置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片  # 补充：Agg 后端不依赖图形界面，仅用于保存图片
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制损失曲线与预测对比图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，把特征与目标值缩放到 0-1 区间
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、均方误差、平均绝对误差三个评价指标
from sklearn.model_selection import train_test_split, KFold  # 导入训练集划分与 K 折交叉验证工具
import torch  # 导入 PyTorch 主模块，负责张量运算与设备选择
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU 网络与损失函数
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集与数据加载器工具，构建批量训练管道

# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有可用 GPU 时选择 CUDA，否则回退到 CPU
print(f"使用设备: {device}")  # 打印当前使用的计算设备
if torch.cuda.is_available():  # 检测是否存在可用的 CUDA GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 配置中文字体为黑体，保证图中中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，避免坐标轴负号乱码
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 按优先级排列的候选 CSV 文件编码列表
data = None  # 初始化数据变量为空，用于标记是否读取成功
for enc in encodings:  # 遍历每一种候选编码
    try:  # 进入 try 块，捕获解码异常
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取温湿度训练数据 CSV
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功读取所使用的编码
        break  # 读取成功则不再尝试其他编码
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码相关异常
        continue  # 当前编码失败，继续尝试下一种

if data is None:  # 若所有编码都失败，data 仍为 None
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常提示训练数据文件无法读取

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔构造与数据等长的时间戳列
features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 候选输入特征名称列表
target = '温度偏差'  # 指定预测目标列为温度偏差
features = [f for f in features if f in data.columns]  # 只保留数据中实际存在的候选特征，避免列缺失报错
# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印分隔线，标识缺失值处理开始
print("缺失值处理")  # 打印缺失值处理标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前的数据形状
print(f"\n各列缺失值统计:")  # 提示即将输出各列缺失值统计
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅统计数据集中存在的列
        nan_count = data[col].isna().sum()  # 统计当前列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算当前列缺失值占比（百分比）
        if nan_count > 0:  # 仅当存在缺失值时才打印统计
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列的缺失数量与占比
# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度', '相对湿度']  # 连续型变量列表，采用线性插值填补
discrete_vars = ['日龄']  # 离散型变量列表，采用前后向填充
# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录插值前的缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 使用线性插值填补缺失，双向填充覆盖首尾
        data[col] = data[col].ffill()  # 再用前值填充开头残留的缺失
        data[col] = data[col].bfill()  # 最后用后值填充结尾残留的缺失
        after_nan = data[col].isna().sum()  # 记录插值后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前的缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后的缺失值数量
# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        data[col] = data[col].fillna(method='ffill')  # 用前值向后填充缺失
        data[col] = data[col].fillna(method='bfill')  # 再用后值反向填充开头的缺失
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前的缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后的缺失值数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列中剩余缺失值总数
if remaining_nan > 0:  # 若仍有缺失值则进入删除行分支
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 警告仍有缺失并提示将删除对应行
    data = data.dropna(subset=features + [target])  # 删除特征或目标列中含缺失值的样本行
print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据形状
print(f"{'=' * 60}\n")  # 打印分隔线并换行
print(f"\n【纯GRU模型】")  # 打印模型类型标题，标明为纯 GRU 模型
print(f"使用特征: {features}")  # 打印实际使用的特征列表
print(f"特征数量: {len(features)}")  # 打印特征数量

# ===================== 数据集划分 =====================
time_step = 36  # GRU时间窗口：48步 × 5分钟 = 4小时  # 补充：定义 GRU 输入序列的时间步长
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟  # 补充：定义标签相对输入窗口的预测偏移量

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印时间窗口步数及对应分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移步数及对应分钟数

# 划分数据集
split_index = int(len(data) * 0.8)  # 取前 80% 样本作为训练集切分点
train_data = data.iloc[:split_index].copy()  # 切分出训练集并复制，避免修改原数据
test_data = data.iloc[split_index:].copy()  # 切分出测试集并复制

print(f"\n数据集划分:")  # 打印数据集划分标题
print(f"训练集大小: {len(train_data)} ({len(train_data) / len(data) * 100:.1f}%)")  # 打印训练集样本数与占比
print(f"测试集大小: {len(test_data)} ({len(test_data) / len(data) * 100:.1f}%)")  # 打印测试集样本数与占比

# ===================== 归一化 =====================
print("\n归一化处理...")  # 提示开始归一化处理
feature_scaler = MinMaxScaler()  # 创建特征归一化器
target_scaler = MinMaxScaler()  # 创建目标值归一化器

# 只在训练集上fit
feature_scaler.fit(train_data[features])  # 仅在训练集特征上拟合归一化器，避免数据泄漏
target_scaler.fit(train_data[[target]])  # 仅在训练集目标上拟合目标归一化器

# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 把归一化后的训练特征重建为 DataFrame
    feature_scaler.transform(train_data[features]),  # 对训练集特征执行归一化变换
    columns=features  # 指定 DataFrame 的列名为原特征名
)  # 训练特征 DataFrame 构造结束
test_features = pd.DataFrame(  # 把归一化后的测试特征重建为 DataFrame
    feature_scaler.transform(test_data[features]),  # 对测试集特征执行归一化变换
    columns=features  # 指定与训练集一致的列名
)  # 测试特征 DataFrame 构造结束

train_target = target_scaler.transform(train_data[[target]]).flatten()  # 训练目标归一化后展平为一维数组
test_target = target_scaler.transform(test_data[[target]]).flatten()  # 测试目标归一化后展平为一维数组


# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):  # 定义滑动窗口序列构造函数
    """把二维特征表转换为滑动窗口时序样本。

    参数：
        data: 归一化后的特征 DataFrame。
        time_step: 每个样本包含的历史步数。
    返回：形状为 (样本数, time_step, 特征数) 的 numpy 数组。
    """
    sequences = []  # 初始化序列样本列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点，保证窗口与预测偏移都不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取长度为 time_step 的连续片段作为一个样本
    return np.array(sequences)  # 把样本列表转换为 numpy 数组返回


print("\n创建训练集序列...")  # 提示开始创建训练集序列
X_train = create_sequences(train_features, time_step)  # 由归一化训练特征构造滑动窗口序列
y_train = train_target[time_step + pred_offset - 1:]  # 按时间步长与偏移对齐训练标签
y_train = y_train[:len(X_train)]  # 截断训练标签，使样本数与 X_train 一致

print("\n创建测试集序列...")  # 提示开始创建测试集序列
X_test = create_sequences(test_features, time_step)  # 由归一化测试特征构造滑动窗口序列
y_test = test_target[time_step + pred_offset - 1:]  # 按时间步长与偏移对齐测试标签
y_test = y_test[:len(X_test)]  # 截断测试标签，使样本数与 X_test 一致

print(f"\n序列数据形状:")  # 提示打印序列数据形状
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集特征与标签的形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集特征与标签的形状


# ===================== 纯GRU模型定义 =====================
class GRUModel(nn.Module):  # 定义纯 GRU 回归模型类
    """
    纯GRU模型
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)

    参数：
        input_dim: 输入特征维度。
        time_step: 序列时间步长，保存为实例属性供参考。
        gru_hidden: GRU 隐藏单元数，默认 128。
        gru_layers: GRU 堆叠层数，默认 1。
        dropout: 多层时的丢弃比例，默认 0.3。
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 初始化 GRU 层与全连接输出层
        """初始化 GRU 层与全连接输出层，并保存模型基本配置。"""
        super(GRUModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间步长配置
        self.input_dim = input_dim  # 保存输入特征维度配置
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏单元数配置

        # GRU层
        self.gru = nn.GRU(  # 定义单向 GRU 层，参数见下方数行
            input_size=input_dim,  # GRU 输入特征维度
            hidden_size=gru_hidden,  # GRU 隐藏层单元数
            num_layers=gru_layers,  # GRU 堆叠层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅当多层时启用 dropout，单层时置 0
            batch_first=True,  # 输入张量按 batch 优先排列
            bidirectional=False  # 使用单向 GRU
        )  # GRU 层定义结束

        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层把隐藏状态映射为单一预测值

    def forward(self, x):  # 定义前向传播方法
        """前向传播：取最后时刻的输出接入全连接层，得到形状 (batch, 1) 的预测。"""
        # x: (batch, time_step, input_dim)
        # GRU编码
        out, h = self.gru(x)  # out: (batch, time_step, hidden)  # 前向传播：得到每个时间步的输出，h 为最终隐状态

        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)  # 取最后一个时间步的输出作为整段序列的表征

        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 经全连接层输出预测值


class EarlyStopping:  # 定义早停控制器类
    """早停策略：验证损失长期不改善时终止训练并保留最佳模型参数。

    参数：
        patience: 允许的最大未改善轮数，默认 10。
        min_delta: 判定为改善所需的最小下降幅度，默认 1e-4。
    """

    def __init__(self, patience=10, min_delta=1e-4):  # 初始化早停的计数器、阈值与最佳参数
        """初始化早停相关的计数器、阈值与最佳参数缓存。"""
        self.patience = patience  # 保存早停容忍轮数
        self.min_delta = min_delta  # 保存最小改善阈值，小于该值视为未改善
        self.counter = 0  # 初始化未改善轮数计数器
        self.best_loss = None  # 初始化最佳验证损失
        self.early_stop = False  # 初始化早停触发标志
        self.best_model = None  # 初始化最佳模型参数存放位置

    def __call__(self, val_loss, model):  # 定义按验证损失更新早停状态的回调
        """根据当前验证损失更新最佳模型并判断是否触发早停。"""
        if self.best_loss is None:  # 首次调用时最佳损失为空
            self.best_loss = val_loss  # 将当前验证损失设为初始最佳
            self.best_model = model.state_dict().copy()  # 保存当前模型参数副本
        elif val_loss > self.best_loss - self.min_delta:  # 若验证损失没有明显下降则视为未改善
            self.counter += 1  # 未改善轮数加一
            if self.counter >= self.patience:  # 达到容忍轮数上限
                self.early_stop = True  # 置位早停标志
        else:  # 否则说明验证损失获得了改善
            self.best_loss = val_loss  # 更新最佳验证损失
            self.best_model = model.state_dict().copy()  # 更新最佳模型参数副本
            self.counter = 0  # 重置未改善计数


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 定义模型训练与验证函数
    """训练模型并在每个 epoch 做验证，返回两路损失曲线。

    参数：
        model: 待训练的神经网络模型。
        train_loader / val_loader: 训练集与验证集数据加载器。
        epochs: 最大训练轮数，默认 200。
        lr: Adam 学习率，默认 0.001。
        patience: 早停容忍轮数，默认 10。
    返回：
        (train_losses, val_losses)，每轮的平均训练损失与验证损失列表。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新模型参数
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器

    train_losses = []  # 初始化训练损失记录列表
    val_losses = []  # 初始化验证损失记录列表

    for epoch in range(epochs):  # 按轮次进行训练
        # 训练阶段
        model.train()  # 切换模型为训练模式，启用 dropout
        train_loss = 0.0  # 清零本轮训练损失累加器
        for batch_x, batch_y in train_loader:  # 遍历训练集的每个批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把批次数据移动到指定设备

            optimizer.zero_grad()  # 清空上一轮累积的梯度
            outputs = model(batch_x)  # 前向传播得到预测值
            loss = criterion(outputs, batch_y)  # 计算预测与真值的损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新模型参数

            train_loss += loss.item()  # 累加当前批次的损失

        train_loss /= len(train_loader)  # 求本轮平均训练损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换为评估模式，关闭 dropout
        val_loss = 0.0  # 清零本轮验证损失累加器
        with torch.no_grad():  # 验证阶段关闭梯度计算
            for batch_x, batch_y in val_loader:  # 遍历验证集各批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把验证批次移动到指定设备
                outputs = model(batch_x)  # 前向传播得到验证预测
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加验证损失

        val_loss /= len(val_loader)  # 求本轮平均验证损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 个 epoch 打印一次训练状态
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前轮次的训练与验证损失

        # 早停检查
        early_stopping(val_loss, model)  # 调用早停控制器更新状态
        if early_stopping.early_stop:  # 若触发早停则进入收尾分支
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停触发时的轮次
            model.load_state_dict(early_stopping.best_model)  # 恢复验证损失最小时的模型参数
            break  # 跳出训练循环

    # 加载最佳模型
    if early_stopping.best_model is not None:  # 训练结束后若保存过最佳模型
        model.load_state_dict(early_stopping.best_model)  # 把最佳参数装入模型作为最终结果

    return train_losses, val_losses  # 返回训练与验证损失曲线


# ===================== K折交叉验证训练 =====================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(features)  # 输入维度等于实际使用的特征数量
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
#     # 构建纯GRU模型
#     model = GRUModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)
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
# plt.savefig('kfold_training_GRU_pytorch.png', dpi=600, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRU_pytorch.png")

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印训练最终模型的标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练集序列样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试集序列样本数
X_train_tensor = torch.FloatTensor(X_train)  # 训练特征序列转换为浮点张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 训练标签转换为浮点张量并增加特征轴
X_test_tensor = torch.FloatTensor(X_test)  # 测试特征序列转换为浮点张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试标签转换为浮点张量并增加特征轴

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 计算从训练集中切分验证集的索引
X_train_split = X_train_tensor[:train_val_split]  # 取前 80% 作为子训练集特征
y_train_split = y_train_tensor[:train_val_split]  # 取前 80% 作为子训练集标签
X_val_split = X_train_tensor[train_val_split:]  # 剩余部分作为验证集特征
y_val_split = y_train_tensor[train_val_split:]  # 剩余部分作为验证集标签

train_dataset = TensorDataset(X_train_split, y_train_split)  # 把子训练集张量封装为数据集
val_dataset = TensorDataset(X_val_split, y_val_split)  # 把验证集张量封装为数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 把测试集张量封装为数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构建训练集加载器，打乱样本顺序
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 构建验证集加载器，不打乱顺序
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 构建测试集加载器，不打乱顺序
final_model = GRUModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)  # 实例化最终 GRU 模型并移至设备
print(final_model)  # 打印模型结构摘要
print("\n开始训练最终模型...")  # 提示开始训练最终模型
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印模型参数总量
print(f"训练集样本数: {len(X_train_split)}")  # 打印用于训练的样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印用于早停验证的样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,  # 训练最终模型，参数见下一行
                                       epochs=200, lr=0.001, patience=10)  # 训练参数：200 轮、学习率 0.001、早停容忍 10 轮

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练完成提示与最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12x6 英寸的损失曲线画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失随轮次变化曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失随轮次变化曲线
plt.title('模型训练过程 - 损失曲线')  # 设置图表标题
plt.xlabel('训练轮数 (Epoch)')  # 设置横轴标签为训练轮数
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签为 MSE 损失
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.ylim(bottom=0)  # 设置纵轴下限为 0
import matplotlib.ticker as ticker  # 导入刻度定位与格式化工具
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.005))  # 设置纵轴主刻度间隔为 0.005
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  # 设置纵轴刻度保留三位小数
plt.tight_layout()  # 自动调整布局防止标签重叠
plt.savefig('training_loss_GRU_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存训练损失曲线
print("训练损失曲线已保存: training_loss_GRU_pytorch.png")  # 打印损失曲线保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 把测试真值反归一化到原始量纲

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试特征转换为浮点张量并移至设备

final_model.eval()  # 切换最终模型为评估模式
with torch.no_grad():  # 关闭梯度计算以加速预测
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 执行预测并把结果转回 numpy 数组

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 把预测值反归一化到原始量纲

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算测试集 R2
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算测试集均方误差
rmse_test = np.sqrt(mse_test)  # 对 MSE 开方得到 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算测试集平均绝对误差

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果")  # 打印测试集评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印测试集 R2
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印测试集 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15x5 英寸的测试集预测对比画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 把测试真值展平成一维
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 把测试预测值展平成一维

test_timestamps = test_data['timestamp'].values[time_step + pred_offset:]  # 从测试集时间戳中截取与标签对齐的时间轴
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间轴使其与预测结果等长

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实温度偏差曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU预测', linewidth=2)  # 绘制 GRU 预测温度偏差曲线
plt.xlabel('时间')  # 设置横轴标签为时间
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签为温度偏差（摄氏度）
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置标题并附带 R2 与 RMSE
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.savefig('test_prediction_GRU_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存测试集预测图
print("测试集预测结果已保存: test_prediction_GRU_pytorch.png")  # 打印测试集预测图保存提示

# ===================== 模型保存 =====================
torch.save({  # 开始构造要保存的模型检查点字典
    'model_state_dict': final_model.state_dict(),  # 保存模型权重参数
    'input_dim': input_dim,  # 保存输入特征维度
    'time_step': time_step,  # 保存时间步长
    'features': features,  # 保存特征名称列表
    'feature_scaler': feature_scaler,  # 保存特征归一化器对象
    'target_scaler': target_scaler,  # 保存目标归一化器对象
}, 'Model_GRU_pytorch.pth')  # 把检查点写入 Model_GRU_pytorch.pth 文件
print("模型已保存:Model_GRU_pytorch.pth")  # 打印模型保存成功提示

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始处理新数据
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取待预测的新温湿度数据 CSV
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',  # 按每步 5 分钟、起点 2025/6/21 0:04 构造新数据时间戳
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 时间戳构造参数续行

# 归一化
new_features = pd.DataFrame(  # 用训练集归一化器把新特征归一化并重建 DataFrame
    feature_scaler.transform(new_data[features]),  # 对新数据特征执行归一化变换
    columns=features  # 指定与训练一致的列名
)  # DataFrame 构造结束

# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据的原始目标值（未归一化）

X_new = create_sequences(new_features, time_step)  # 由归一化后的新特征构造滑动窗口序列
y_new = y_new_original[time_step + pred_offset - 1:]  # 按偏移截取与新序列对应的原始目标标签
X_new = X_new[:len(y_new)]  # 截断 X_new 使其与 y_new 长度一致

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 新特征序列转换为浮点张量并移至设备

final_model.eval()  # 切换模型为评估模式
with torch.no_grad():  # 关闭梯度计算进行预测
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据执行预测并转回 numpy

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 把新数据的预测值反归一化到原始量纲
y_new_rescaled = y_new.reshape(-1, 1)  # 把新数据真值整理成二维列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R2
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据均方误差
rmse_new = np.sqrt(mse_new)  # 对 MSE 开方得到 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据平均绝对误差

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 打印新数据 R2
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线
timestamps = new_data['timestamp'][time_step + pred_offset - 1:]  # 截取与新数据标签对齐的时间戳序列
plt.figure(figsize=(15, 5))  # 创建 15x5 英寸的新数据预测对比画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 把新数据真值展平成一维
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 把新数据预测值展平成一维
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制新数据真实温度偏差曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='GRU预测', linewidth=2)  # 绘制新数据 GRU 预测曲线
plt.xlabel('时间')  # 设置横轴标签为时间
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签为温度偏差（摄氏度）
plt.title(f'新数据集温度偏差结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置标题并附带 R2 与 RMSE
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.savefig('new_data_prediction_GRU_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存新数据预测图
print("新数据集预测结果已保存: new_data_prediction_GRU_pytorch.png")  # 打印新数据预测图保存提示
print("\n========== 所有任务完成 ==========")  # 打印全部任务完成提示