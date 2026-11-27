"""
GRU 模型测试/推理脚本（THI 温湿度指数预测，支持小波变换描述）

本脚本用于加载已经训练好的 GRU + ECA 模型检查点（Model_GRUECA_pytorch.pth），
在新采集的温湿度数据上完成端到端预测与评估。主要流程：
1. 定义与训练时一致的 ECA 通道注意力层、GRUOnlyModel 与 GRUECAModel 模型结构；
2. 多编码尝试读取测试数据，处理时间戳与缺失值；
3. 从检查点读取输入维度、时间窗口、特征列表及归一化器，并校验特征一致性；
4. 用保存的归一化器预处理测试数据、按滑动窗口构造序列；
5. 加载模型权重进行推理，计算 R²/MSE/RMSE/MAE 等指标；
6. 输出预测对比图与包含散点、直方图、时序图、箱线图的误差分析图及统计信息。

说明：脚本注释中提到支持 GRU-Only 与 GRU+小波模型，但当前实际加载并实例化的是 GRU+ECA 模型。
"""
import pandas as pd  # 导入 pandas 库，用于 CSV 读取与 DataFrame 数据处理
import numpy as np  # 导入 numpy 库，用于数组与数值计算
import matplotlib  # 导入 matplotlib 基础模块，用于提前指定绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片，无需图形界面
import matplotlib.pyplot as plt  # 导入 pyplot 子模块，提供绘图接口
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，作为无保存归一化器时的备用拟合工具
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入回归评估指标：R²、MSE、MAE
import torch  # 导入 PyTorch 主库，提供张量与设备支持
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU、全连接层等
# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU（可用时）或 CPU 作为推理设备
print(f"使用设备: {device}")  # 打印本次运行使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号，避免坐标轴负号显示为方块


# ===================== 模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    通过1D卷积实现高效的通道注意力机制
    输入与输出形状均为 (batch, time_step, channels)
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 通道注意力层。

        参数:
            channels: 输入特征的通道数（特征维度）。
            gamma: 自适应卷积核大小计算的缩放系数，默认 2。
            b: 卷积核大小计算的偏置项，默认 1。
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按通道数自适应推导卷积核尺寸参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数  # 若 t 为偶数则加 1，保证核大小为奇数
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩为 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道间交互，无偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 将卷积输出映射为 0~1 的注意力权重
    
    def forward(self, x):
        """对输入执行通道注意力加权。

        参数:
            x: 形状 (batch, time_step, channels) 的输入张量。
        返回:
            形状同为 (batch, time_step, channels) 的加权后张量。
        """
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step)
        x_permuted = x.permute(0, 2, 1)  # 将通道维前移以适配 Conv1d 输入格式
        
        # 全局平均池化
        y = self.avg_pool(x_permuted)  # 沿时间维做全局平均池化，得到每通道全局描述
        
        # 1D卷积
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 转置后做 1D 卷积再转回，捕获通道相关性
        
        # Sigmoid激活
        y = self.sigmoid(y)  # 归一化为 0~1 的通道注意力权重
        
        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 按通道广播权重，对输入逐通道加权
        
        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 还原维度顺序后返回


class GRUOnlyModel(nn.Module):
    """GRU-Only 模型 + ECA-Net注意力机制"""

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):
        """初始化 GRU+ECA 模型（GRU-Only 命名变体）。

        参数:
            input_dim: 每步输入特征维度。
            time_step: 输入序列时间步数。
            gru_hidden: GRU 隐藏层维度，默认 128。
            gru_layers: GRU 层数，默认 1（单层时关闭 dropout）。
            dropout: 多层 GRU 层间丢弃率，默认 0.2。
        """
        super(GRUOnlyModel, self).__init__()  # 调用父类初始化方法
        self.time_step = time_step  # 保存时间步长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层维度

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 层，对输入特征做通道注意力增强
        
        self.gru = nn.GRU(  # 定义单向 GRU 层
            input_size=input_dim,  # 每步输入维度
            hidden_size=gru_hidden,  # 隐藏状态维度
            num_layers=gru_layers,  # GRU 层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 输入以 batch 为第一维
            bidirectional=False  # 单向 GRU
        )

        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层将隐藏状态映射为单个预测值

    def forward(self, x):
        """前向传播：ECA 增强 -> GRU 编码 -> 全连接预测。

        参数:
            x: (batch, time_step, input_dim) 输入序列。
        返回:
            (batch, 1) 的预测张量。
        """
        # ECA注意力增强输入特征
        x = self.eca(x)  # 经 ECA 层进行通道注意力加权
        
        # GRU编码
        out, h = self.gru(x)  # GRU 输出各时刻状态与最终隐状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # 取最后一个时间步的输出作为整体表征
        
        return self.fc(out_last)  # 经全连接层得到预测值并返回


class GRUECAModel(nn.Module):
    """GRU + ECA注意力机制"""
    
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):
        """初始化 GRU+ECA 模型。

        参数:
            input_dim: 每步输入特征维度。
            time_step: 输入序列时间步数。
            gru_hidden: GRU 隐藏层维度，默认 128。
            gru_layers: GRU 层数，默认 1（单层时关闭 dropout）。
            dropout: 多层 GRU 层间丢弃率，默认 0.2。
        """
        super(GRUECAModel, self).__init__()  # 调用父类初始化方法
        self.time_step = time_step  # 保存时间步长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层维度

        # ECA-Net注意力模块
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 层，增强输入特征
        
        # 单层GRU编码器
        self.gru = nn.GRU(  # 定义单向 GRU 层
            input_size=input_dim,  # 每步输入维度
            hidden_size=gru_hidden,  # 隐藏状态维度
            num_layers=gru_layers,  # GRU 层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 输入以 batch 为第一维
            bidirectional=False  # 单向 GRU
        )
        
        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层将隐藏状态映射为单个预测值

    def forward(self, x):
        """前向传播：ECA 增强 -> GRU 编码 -> 全连接预测。

        参数:
            x: (batch, time_step, input_dim) 输入序列。
        返回:
            (batch, 1) 的预测张量。
        """
        # ECA注意力增强
        x = self.eca(x)  # 经 ECA 层进行通道注意力加权
        
        # GRU编码
        out, h = self.gru(x)  # GRU 输出各时刻状态与最终隐状态
        out_last = out[:, -1, :]  # 取最后一个时间步的输出作为整体表征

        # 预测输出
        return self.fc(out_last)  # 经全连接层得到预测值并返回


# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印测试数据加载提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，逐一尝试直到成功
data = None  # 初始化数据变量为空
for enc in encodings:  # 遍历每种候选编码
    try:
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 解码失败则尝试下一种
        continue  # 继续尝试下一个编码

if data is None:  # 若所有编码均失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常终止程序
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟间隔从起始时间生成时间戳
# 特征配置
features = ['日龄','THI','目标温度','当前温度']  # 预定义的候选特征列
target = 'THI'  # 指定预测目标为 THI

# 过滤特征列（只保留数据中存在的列）
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征

print(f"使用特征: {features}")  # 打印最终使用的特征
print(f"数据形状: {data.shape}")  # 打印测试数据规模

# 检查目标列是否存在
if target not in data.columns:  # 若测试数据中缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带THI的CSV文件。")  # 抛出异常提示数据文件不对

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前数据规模
print(f"\n各列缺失值统计:")  # 打印缺失值统计标题
for col in features + [target]:  # 遍历所有特征列与目标列
    if col in data.columns:  # 仅统计存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失占比
        if nan_count > 0:  # 仅打印含缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印缺失数与占比

# 定义连续变量和离散变量
continuous_vars = ['THI', '外界温度', 'CO2', '目标温度']  # 连续变量  # 连续型变量列表，采用插值填充
discrete_vars = ['日龄']  # 离散变量  # 离散型变量列表，采用前后向填充

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失数
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值填充连续变量缺失（双向）
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].fillna(method='ffill')  # 用前一个有效值填充剩余缺失
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].fillna(method='bfill')  # 用后一个有效值填充开头残留缺失
        
        after_nan = data[col].isna().sum()  # 记录填充后缺失数
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失数
        
        # 前向填充
        data[col] = data[col].fillna(method='ffill')  # 用前一个有效值填充离散变量缺失
        # 后向填充
        data[col] = data[col].fillna(method='bfill')  # 用后一个有效值填充开头残留缺失
        
        after_nan = data[col].isna().sum()  # 记录填充后缺失数
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计残留缺失值总数
if remaining_nan > 0:  # 若仍有缺失
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 提示将删除缺失行
    data = data.dropna(subset=features + [target])  # 删除特征或目标中含缺失的行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后数据规模
print(f"{'='*60}\n")  # 打印分隔线

# 重置索引
data = data.reset_index(drop=True)  # 重置索引，保证后续 iloc/序号连续

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 打印模型加载提示

# 加载GRU+ECA模型
MODEL_PATH = 'Model_GRUECA_pytorch.pth'  # 定义模型检查点文件路径

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载检查点到指定设备（含归一化器等对象）
    print(f"成功加载GRU+ECA模型: {MODEL_PATH}")  # 打印加载成功提示
except FileNotFoundError:  # 文件不存在时
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛出异常提示模型文件缺失

input_dim = checkpoint['input_dim']  # 从检查点读取输入维度
time_step = checkpoint['time_step']  # 从检查点读取时间窗口长度

# 从模型中获取特征列表
feature_names_saved = checkpoint.get('feature_names', None)  # 尝试从检查点读取保存的特征列表
if feature_names_saved:  # 若检查点保存了特征列表
    model_features = feature_names_saved  # 采用模型中保存的特征顺序
    print(f"\n从模型加载特征列表: {model_features}")  # 打印加载到的特征列表
else:  # 若未保存
    model_features = features  # 退回使用测试数据推断的特征列表
    print(f"\n⚠️  警告：模型未保存特征列表，使用测试数据的特征: {model_features}")  # 打印警告信息

# 检查测试数据是否包含所需特征
missing_features = [f for f in model_features if f not in data.columns]  # 找出数据中缺失的必需特征
if missing_features:  # 若存在缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常提示缺列

# 使用模型训练时的特征
features = model_features  # 统一使用模型要求的特征列表与顺序
print(f"\n最终使用特征: {features}")  # 打印最终特征

# ===================== 数据预处理 =====================
# +-
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表（训练数据）
train_data = None  # 初始化训练数据变量为空
for enc in encodings:  # 遍历每种候选编码
    try:
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功编码
        break  # 成功则跳出
    except (UnicodeDecodeError, UnicodeError):  # 解码失败继续尝试
        continue  # 尝试下一个编码
if train_data is None:  # 若全部失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常

print("\n使用原始特征（不进行小波分解）...")  # 打印预处理方式说明

# 使用原始特征
train_combined = train_data[features]  # 抽取训练数据的特征列
test_combined = data[features]  # 抽取测试数据的特征列

# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若检查点含归一化器
    print("\n使用模型保存的归一化器...")  # 打印提示
    feature_scaler = checkpoint['feature_scaler']  # 直接复用训练时保存的特征归一化器
    target_scaler = checkpoint['target_scaler']  # 直接复用训练时保存的目标归一化器
else:  # 若未保存
    print("\n使用训练数据重新拟合归一化器...")  # 打印提示
    feature_scaler = MinMaxScaler()  # 新建特征归一化器
    target_scaler = MinMaxScaler()  # 新建目标归一化器
    feature_scaler.fit(train_combined)  # 用训练特征拟合
    target_scaler.fit(train_data[[target]])  # 用训练目标拟合

# 保存原始列名
original_columns = test_combined.columns  # 记录归一化前的列名，用于重建 DataFrame

test_combined = pd.DataFrame(  # 归一化测试特征并重建为 DataFrame
    feature_scaler.transform(test_combined),  # 用归一化器缩放测试特征
    columns=original_columns  # 保留原列名
)

print(f"  特征数量: {len(test_combined.columns)}")  # 打印测试特征数量

# ===================== 序列构造 =====================
pred_offset = 6  # 预测偏移：用当前窗口预测 6 步（30 分钟）之后的 THI

def create_sequences(data, time_step):
    """
    创建时间序列样本
    注意：这里是对单个数据集（机器人新数据）进行序列构造，不涉及训练/测试划分
    
    参数:
        data: 归一化后的特征 DataFrame。
        time_step: 每个样本的历史窗口长度。
    返回:
        np.ndarray 三维序列 (样本数, time_step, 特征数)。
    """
    sequences = []  # 收集各滑动窗口样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，保证窗口与偏移不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取第 i 个历史窗口
    return np.array(sequences)  # 堆叠成三维数组返回

print("\n创建序列数据...")  # 打印序列构造提示
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 用目标归一化器缩放测试标签

# 创建序列
X_test = create_sequences(test_combined, time_step)  # 构造测试输入序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 按偏移对齐目标并展平
y_test = y_test[:len(X_test)]  # 确保长度匹配  # 裁剪标签使其与样本数一致

print(f"\n测试数据形状:")  # 打印数据形状检查标题
print(f"  X_test: {X_test.shape}")  # 打印测试输入形状
print(f"  y_test: {y_test.shape}")  # 打印测试标签形状

# ===================== 构建模型 =====================
model = GRUECAModel(  # 实例化待加载权重的模型
    input_dim=input_dim,  # 输入维度来自检查点
    time_step=time_step,  # 时间步来自检查点
    gru_hidden=128,  # 隐藏层维度，与训练时一致
    gru_layers=1,  # GRU 层数，与训练时一致
    dropout=0.2  # dropout 比例，与训练时一致
).to(device)  # 将模型移至指定设备

model.load_state_dict(checkpoint['model_state_dict'])  # 加载训练好的权重
model.eval()  # 切换为评估模式

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型参数总数

# ===================== 模型预测 =====================
print("\n开始预测...")  # 打印预测开始提示
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试序列转张量并移至设备

with torch.no_grad():  # 关闭梯度计算进行推理
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并取回 CPU 的 numpy 结果

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将真实标签反归一化为真实 THI
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测值反归一化为真实 THI

# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 计算均方根误差 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

model_name = "GRU+ECA"  # 定义模型名称，用于图表标注

print("\n" + "=" * 60)  # 打印分隔线
print(f"{model_name} 模型测试结果")  # 打印模型评估结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R²
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
plot_length = min(1600, len(y_test_rescaled))  # 限制绘图点数，最多取前 1600 个
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的预测值

# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成绘图 X 轴的步骤索引

plt.figure(figsize=(15, 8))  # 创建画布
y_test_flat = y_test_plot.flatten()  # 展平真实值
y_pred_flat = y_pred_plot.flatten()  # 展平预测值
plt.tick_params(axis='both', labelsize=16)  # 设置坐标轴刻度字号
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线（蓝色）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值曲线（橙色）
plt.xlabel('Timestep', fontsize=16)  # 设置 X 轴标签
plt.ylabel('THI', fontsize=16)  # 设置 Y 轴标签
# plt.title(f'{model_name} Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.tight_layout()  # 自动调整子图间距
plt.text(0.5, 0.95, '(d)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', va='top', ha='center')  # 在图上添加子图编号 (d)
plt.tight_layout()  # 再次调整布局以容纳新增文本
output_filename = 'test_GRUECA_prediction.png'  # 定义输出图片文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 保存预测对比图（高分辨率）
print(f"\n预测结果图已保存: {output_filename}")  # 打印保存提示
plt.close()  # 关闭图形，释放内存

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算全部预测误差（预测 - 真实）

# 仅用于绘图：取前300个点
max_points = min(300, len(errors))  # 限制绘图用的误差点数，最多 300
y_test_plot_err = y_test_rescaled[:max_points].flatten()  # 取前 N 个真实值
y_pred_plot_err = y_pred_rescaled[:max_points].flatten()  # 取前 N 个预测值
errors_plot = y_pred_plot_err - y_test_plot_err  # 计算绘图用的前 N 个误差

print(f"\n误差分析图使用前 {max_points} 个数据点")  # 打印误差分析使用的数据点数

plt.figure(figsize=(15, 10))  # 创建 2x2 误差分析画布

# 子图1: True value vs Predicted value（前300个点）
plt.subplot(2, 2, 1)  # 选中第 1 个子图
plt.scatter(  # 绘制真实值-预测值散点图
    y_test_plot_err,  # X 轴：真实值
    y_pred_plot_err,  # Y 轴：预测值
    alpha=0.35,  # 点透明度
    s=45,  # 点大小
    color='#b58f86',  # 点颜色
    label='Predicted value'  # 图例标签
)
min_val = min(y_test_plot_err.min(), y_pred_plot_err.min())  # 取真实与预测的最小值作为坐标轴下界
max_val = max(y_test_plot_err.max(), y_pred_plot_err.max())  # 取真实与预测的最大值作为坐标轴上界
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')  # 绘制 y=x 参考线（红色虚线）
plt.xlabel('True value', fontsize=11)  # 设置 X 轴标签
plt.ylabel('Predicted value', fontsize=11)  # 设置 Y 轴标签
plt.title('True value vs Predicted value', fontsize=12)  # 设置子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.4)  # 显示网格线
margin = (max_val - min_val) * 0.05  # 计算坐标轴上下浮动的边距
plt.xlim(min_val - margin, max_val + margin)  # 设置 X 轴范围
plt.ylim(min_val - margin, max_val + margin)  # 设置 Y 轴范围

# 子图2: 误差分布直方图（前300个点）
plt.subplot(2, 2, 2)  # 选中第 2 个子图
plt.hist(errors_plot, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差分布直方图（50 个箱子）
plt.xlabel('Prediction Error', fontsize=11)  # 设置 X 轴标签
plt.ylabel('Frequency', fontsize=11)  # 设置 Y 轴标签
plt.title(f'Error Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 在 0 误差处画红色虚线
plt.grid(True, alpha=0.3, axis='y')  # 显示 Y 方向网格线

# 子图3: 误差时序图（前300个点）
plt.subplot(2, 2, 3)  # 选中第 3 个子图
error_steps = np.arange(len(errors_plot))  # 生成误差时序图的 X 轴步骤索引
plt.plot(error_steps, errors_plot, alpha=0.7, linewidth=1)  # 绘制误差随步骤变化的曲线
plt.xlabel('Timestep', fontsize=11)  # 设置 X 轴标签
plt.ylabel('Prediction Error', fontsize=11)  # 设置 Y 轴标签
plt.title(f'Error Time Series (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 在 0 误差处画红色虚线
plt.grid(True, alpha=0.3)  # 显示网格线

# 子图4: 误差箱线图（前300个点）
plt.subplot(2, 2, 4)  # 选中第 4 个子图
plt.boxplot([errors_plot], tick_labels=[model_name])  # 绘制误差箱线图
plt.ylabel('Prediction Error', fontsize=11)  # 设置 Y 轴标签
plt.title(f'Error Statistics Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 在 0 误差处画红色虚线
plt.grid(True, alpha=0.3, axis='y')  # 显示 Y 方向网格线

plt.tight_layout()  # 自动调整子图间距

error_filename = 'test_GRUECA_error_analysis.png'  # 定义误差分析图文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 保存误差分析图（高分辨率）
print(f"误差分析图已保存: {error_filename}")  # 打印保存提示
plt.close()  # 关闭图形，释放内存

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("Error Statistics")  # 打印误差统计标题
print("=" * 60)  # 打印分隔线
print(f"Mean Error:     {np.mean(errors):.4f}")  # 打印误差均值
print(f"Std Deviation:  {np.std(errors):.4f}")  # 打印误差标准差
print(f"Max Error:      {np.max(errors):.4f}")  # 打印最大误差
print(f"Min Error:      {np.min(errors):.4f}")  # 打印最小误差
print(f"Median Error:   {np.median(errors):.4f}")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\nTest Completed!")  # 打印测试完成提示
