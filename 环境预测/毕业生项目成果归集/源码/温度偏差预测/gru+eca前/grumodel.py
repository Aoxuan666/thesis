"""
GRU模型测试脚本（支持小波变换）
用于加载训练好的GRU模型并进行预测
支持：GRU-Only模型 和 GRU+小波变换模型

本脚本是训练阶段（gru.py）的推理配套代码：加载已保存的 GRU+ECA 模型检查点，读取新采集的温湿度 CSV，
进行多编码读取、缺失值处理、特征对齐与滑动窗口序列构造，重建模型并载入权重进行预测，
最后输出 R²/MSE/RMSE/MAE 指标并绘制预测对比图与四子图误差分析图。
"""
import pandas as pd  # 导入 pandas 数据处理库，简写为 pd，用于读取 CSV 与处理表格数据
import numpy as np  # 导入 numpy 数值计算库，简写为 np，用于数组与矩阵运算
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用非交互式 Agg 后端，只保存图片不弹出窗口
import matplotlib.pyplot as plt  # 导入 pyplot 绘图模块，简写为 plt
from sklearn.preprocessing import MinMaxScaler  # 导入最小-最大归一化器，用于特征缩放
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R²、MSE、MAE
import torch  # 导入 PyTorch 主库，用于张量计算与 GPU 加速
import torch.nn as nn  # 导入神经网络模块，用于构建网络层
# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若 CUDA 可用则选 GPU，否则使用 CPU
print(f"使用设备: {device}")  # 打印当前选定的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图默认中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免负号显示为方块


# ===================== 模型定义 =====================
class ECALayer(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    通过1D卷积实现高效的通道注意力机制。
    Args:
        channels: 输入特征通道数。
        gamma: 自适应卷积核尺度公式中的缩放参数。
        b: 自适应卷积核尺度公式中的偏置参数。
    """
    def __init__(self, channels, gamma=2, b=1):
        """初始化 ECA 层：自适应确定卷积核大小并构建池化、卷积、激活模块。"""
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按公式由通道数计算尺度 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数（原注释保留）
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，将时间维压缩到 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D 卷积学习通道交互，无偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid 把通道权重映射到 (0,1)
    
    def forward(self, x):
        """前向传播：维度置换→全局池化→1D卷积→Sigmoid→通道加权→还原维度。"""
        # x: (batch, time_step, channels)
        # 转换为 (batch, channels, time_step)
        x_permuted = x.permute(0, 2, 1)  # 置换维度，使通道维位于时间维之前
        
        # 全局平均池化
        y = self.avg_pool(x_permuted)  # 每个通道在时间维上做全局平均
        
        # 1D卷积
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 换维后做 1D 卷积建模通道相关性再换回
        
        # Sigmoid激活
        y = self.sigmoid(y)  # 得到各通道注意力权重
        
        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 权重广播乘回各通道特征
        
        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 还原维度顺序后返回


class GRUOnlyModel(nn.Module):
    """GRU-Only 模型 + ECA-Net注意力机制。

    Args:
        input_dim: 输入特征维度。
        time_step: 时间窗口长度（仅记录）。
        gru_hidden: GRU 隐藏层维度。
        gru_layers: GRU 层数。
        dropout: 多层时的层间 dropout 比例。
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):
        """构建 ECA 注意力、GRU 编码器与全连接输出层。"""
        super(GRUOnlyModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层维度

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 构造通道注意力层
        
        self.gru = nn.GRU(  # 定义 GRU 层
            input_size=input_dim,  # 输入维度
            hidden_size=gru_hidden,  # 隐藏维度
            num_layers=gru_layers,  # 层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 以 batch 为第一维
            bidirectional=False  # 单向 GRU
        )

        self.fc = nn.Linear(gru_hidden, 1)  # 输出层：隐藏维度映射到单个预测值

    def forward(self, x):
        """前向传播：ECA 增强→GRU 编码→取末时刻输出→全连接预测。"""
        # ECA注意力增强输入特征
        x = self.eca(x)  # 对输入做通道注意力加权
        
        # GRU编码
        out, h = self.gru(x)  # 得到序列输出与隐藏状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # 取最后一个时间步的输出
        
        return self.fc(out_last)  # 经全连接层输出预测值


class GRUECAModel(nn.Module):
    """GRU + ECA注意力机制。

    Args:
        input_dim: 输入特征维度。
        time_step: 时间窗口长度（仅记录）。
        gru_hidden: GRU 隐藏层维度。
        gru_layers: GRU 层数。
        dropout: 多层时的层间 dropout 比例。
    """
    
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):
        """构建 ECA 注意力、单层 GRU 编码器与全连接输出层。"""
        super(GRUECAModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层维度

        # ECA-Net注意力模块
        self.eca = ECALayer(channels=input_dim)  # 构造通道注意力层
        
        # 单层GRU编码器
        self.gru = nn.GRU(  # 定义 GRU 层
            input_size=input_dim,  # 输入维度
            hidden_size=gru_hidden,  # 隐藏维度
            num_layers=gru_layers,  # 层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅多层时启用 dropout
            batch_first=True,  # 以 batch 为第一维
            bidirectional=False  # 单向 GRU
        )
        
        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层输出单个预测值

    def forward(self, x):
        """前向传播：ECA 增强→GRU 编码→取末时刻输出→全连接预测。"""
        # ECA注意力增强
        x = self.eca(x)  # 通道注意力加权输入
        
        # GRU编码
        out, h = self.gru(x)  # 得到序列输出与隐藏状态
        out_last = out[:, -1, :]  # 取最后时间步输出

        # 预测输出
        return self.fc(out_last)  # 返回全连接层预测结果


# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始加载测试数据
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
data = None  # 初始化数据占位变量
for enc in encodings:  # 遍历候选编码
    try:
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码错误
        continue  # 尝试下一种编码

if data is None:  # 若所有编码都失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 由索引按 5 分钟间隔构造时间戳列
# 特征配置
features = ['日龄','温度偏差','外界温度','CO2','目标温度']  # 候选特征列名称
target = '温度偏差'  # 预测目标列

# 过滤特征列（只保留数据中存在的列）
features = [f for f in features if f in data.columns]  # 仅保留数据中真实存在的特征

print(f"使用特征: {features}")  # 打印实际使用的特征列表
print(f"数据形状: {data.shape}")  # 打印测试数据规模

# 检查目标列是否存在
if target not in data.columns:  # 若缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带温度偏差的CSV文件。")  # 抛出关键错误提示

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据规模
print(f"\n各列缺失值统计:")  # 打印缺失统计标题
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅处理存在的列
        nan_count = data[col].isna().sum()  # 统计缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失占比
        if nan_count > 0:  # 仅输出存在缺失的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印列名与缺失情况

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续变量（插值+填充）（原注释保留）
discrete_vars = ['日龄']  # 离散变量（前后填充）（原注释保留）

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理含缺失的列
        print(f"\n处理 {col}:")  # 打印当前列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失数
        
        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 双向线性插值填充
        
        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].fillna(method='ffill')  # 前向填充开头缺失
        
        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].fillna(method='bfill')  # 后向填充结尾缺失
        
        after_nan = data[col].isna().sum()  # 统计填充后缺失数
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量列
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理含缺失的列
        print(f"\n处理 {col}:")  # 打印当前列名
        before_nan = data[col].isna().sum()  # 记录填充前缺失数
        
        # 前向填充
        data[col] = data[col].fillna(method='ffill')  # 前向填充离散变量缺失
        # 后向填充
        data[col] = data[col].fillna(method='bfill')  # 后向填充离散变量缺失
        
        after_nan = data[col].isna().sum()  # 统计填充后缺失数
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计剩余缺失值总数
if remaining_nan > 0:  # 若仍有缺失
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告
    data = data.dropna(subset=features + [target])  # 删除仍含缺失的样本行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后数据规模
print(f"{'='*60}\n")  # 打印分隔线并换行

# 重置索引
data = data.reset_index(drop=True)  # 重置索引，保证后续按位置取数连续

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 提示开始加载模型

# 加载GRU+ECA模型
MODEL_PATH = 'Model_GRUECA_pytorch.pth'  # 指定检查点文件路径

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载检查点到指定设备
    print(f"成功加载GRU+ECA模型: {MODEL_PATH}")  # 打印加载成功提示
except FileNotFoundError:  # 若文件不存在
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛出异常提示

input_dim = checkpoint['input_dim']  # 从检查点读取输入维度
time_step = checkpoint['time_step']  # 从检查点读取时间窗口长度

# 从模型中获取特征列表
feature_names_saved = checkpoint.get('feature_names', None)  # 尝试读取保存的特征名列表
if feature_names_saved:  # 若检查点包含特征名
    model_features = feature_names_saved  # 采用模型保存的特征顺序
    print(f"\n从模型加载特征列表: {model_features}")  # 打印加载到的特征列表
else:  # 若未保存特征名
    model_features = features  # 回退使用测试数据的特征列表
    print(f"\n⚠️  警告：模型未保存特征列表，使用测试数据的特征: {model_features}")  # 打印警告

# 检查测试数据是否包含所需特征
missing_features = [f for f in model_features if f not in data.columns]  # 找出测试数据中缺失的特征
if missing_features:  # 若有缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常提示

# 使用模型训练时的特征
features = model_features  # 统一采用模型训练时的特征列表
print(f"\n最终使用特征: {features}")  # 打印最终特征列表

# ===================== 数据预处理 =====================
# +-
# 尝试多种编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表（用于读取训练数据）
train_data = None  # 初始化训练数据占位变量
for enc in encodings:  # 遍历候选编码
    try:
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功编码
        break  # 成功后跳出
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码错误
        continue  # 尝试下一种编码
if train_data is None:  # 若所有编码都失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常

print("\n使用原始特征（不进行小波分解）...")  # 提示使用原始特征

# 使用原始特征
train_combined = train_data[features]  # 取训练数据的特征列
test_combined = data[features]  # 取测试数据的特征列

# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若检查点含归一化器
    print("\n使用模型保存的归一化器...")  # 提示直接使用保存的归一化器
    feature_scaler = checkpoint['feature_scaler']  # 读取特征归一化器
    target_scaler = checkpoint['target_scaler']  # 读取目标归一化器
else:  # 否则重新拟合
    print("\n使用训练数据重新拟合归一化器...")  # 提示重新拟合
    feature_scaler = MinMaxScaler()  # 创建特征归一化器
    target_scaler = MinMaxScaler()  # 创建目标归一化器
    feature_scaler.fit(train_combined)  # 用训练特征拟合
    target_scaler.fit(train_data[[target]])  # 用训练目标拟合

# 保存原始列名
original_columns = test_combined.columns  # 记录归一化前的列名

test_combined = pd.DataFrame(  # 将归一化后的测试特征重建为 DataFrame
    feature_scaler.transform(test_combined),  # 用特征归一化器变换测试特征
    columns=original_columns  # 恢复原始列名
)

print(f"  特征数量: {len(test_combined.columns)}")  # 打印特征数量

# ===================== 序列构造 =====================
pred_offset = 6  # 预测偏移步数（6 步 × 5 分钟 = 30 分钟）

def create_sequences(data, time_step):
    """
    创建时间序列样本。
    注意：这里是对单个数据集（机器人新数据）进行序列构造，不涉及训练/测试划分。

    Args:
        data: 归一化后的特征 DataFrame。
        time_step: 输入序列长度（历史窗口）。
    Returns:
        np.ndarray: 形状为 (样本数, time_step, 特征数) 的序列数组。
    """
    sequences = []  # 收集各时间窗口样本
    for i in range(len(data) - time_step - pred_offset + 1):  # 滑动窗口起点，预留 time_step 与偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 取连续 time_step 行作为样本
    return np.array(sequences)  # 转换为 numpy 数组返回

print("\n创建序列数据...")  # 提示开始构造序列
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 用目标归一化器变换测试目标值

# 创建序列
X_test = create_sequences(test_combined, time_step)  # 由测试特征构造输入序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 按偏移截取并展平目标作为标签
X_test = X_test[:len(y_test)]  # 确保长度匹配：截断序列使其与标签数一致（原注释保留）

print(f"\n测试数据形状:")  # 打印测试数据形状标题
print(f"  X_test: {X_test.shape}")  # 打印输入序列形状
print(f"  y_test: {y_test.shape}")  # 打印标签形状

# ===================== 构建模型 =====================
model = GRUECAModel(  # 实例化 GRU+ECA 模型
    input_dim=input_dim,  # 输入维度来自检查点
    time_step=time_step,  # 时间窗口来自检查点
    gru_hidden=128,  # 隐藏维度 128
    gru_layers=1,  # 单层 GRU
    dropout=0.1  # dropout 比例 0.1
).to(device)  # 把模型搬到计算设备

model.load_state_dict(checkpoint['model_state_dict'])  # 载入训练好的权重
model.eval()  # 切换为评估模式

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型参数总数

# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始预测
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试序列转为张量并搬到设备

with torch.no_grad():  # 关闭梯度计算
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并搬回 CPU 转 numpy

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 还原真实标签量纲
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 还原预测值量纲

# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 由 MSE 开方得到 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

model_name = "GRU+ECA"  # 记录模型名称，用于图表标注

print("\n" + "=" * 60)  # 打印分隔线
print(f"{model_name} 模型测试结果")  # 打印模型测试结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R²
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
# 只绘制前120个时间步长
plot_length = min(1600, len(y_test_rescaled))  # 确定绘制长度，最多 1600 个点
y_test_plot = y_test_rescaled[:plot_length]  # 截取真实值绘图片段
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取预测值绘图片段

# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成横轴步数索引

plt.figure(figsize=(15, 8))  # 创建 15x8 英寸画布
y_test_flat = y_test_plot.flatten()  # 展平真实值
y_pred_flat = y_pred_plot.flatten()  # 展平预测值
plt.tick_params(axis='both', labelsize=16)  # 设置两轴刻度字号为 16
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线（蓝色）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值曲线（橙色）
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签
plt.ylabel('TD (℃)', fontsize=16)  # 设置纵轴标签
# plt.title(f'{model_name} Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=14)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.tight_layout()  # 自动调整布局
# 在图中间靠上添加标签 (a)
plt.text(0.5, 0.95, '(d)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', va='top', ha='center')  # 在子图上方中央添加标签 (d)
plt.tight_layout()  # 再次调整布局以容纳新增文本

output_filename = 'test_GRUECA_prediction.png'  # 指定预测图文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 保存预测对比图
print(f"\n预测结果图已保存: {output_filename}")  # 打印保存提示

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算预测误差（预测减真实）

plt.figure(figsize=(15, 10))  # 创建 15x10 英寸画布

# 子图1: 预测值 vs 真实值散点图
plt.subplot(2, 2, 1)  # 选中 2x2 布局的第 1 个子图
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, s=20)  # 绘制真实值对预测值的散点
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],
         [y_test_rescaled.min(), y_test_rescaled.max()],
         'r--', lw=2, label='理想预测线')  # 绘制 y=x 理想预测参考线
plt.xlabel('真实温度偏差 (℃)', fontsize=11)  # 设置横轴标签
plt.ylabel('预测温度偏差 (℃)', fontsize=11)  # 设置纵轴标签
plt.title(f'预测值 vs 真实值 (R2={r2:.4f})', fontsize=12)  # 设置子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线

# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 选中第 2 个子图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差分布直方图（50 个区间）
plt.xlabel('预测误差 (℃)', fontsize=11)  # 设置横轴标签
plt.ylabel('频数', fontsize=11)  # 设置纵轴标签
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 设置子图标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考竖线
plt.grid(True, alpha=0.3, axis='y')  # 仅显示横向网格线

# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 选中第 3 个子图
error_steps = np.arange(min(120, len(errors)))  # 生成误差图横轴步数（最多 120）
plt.plot(error_steps, errors[:120], alpha=0.7, linewidth=1)  # 绘制误差随时间变化曲线
plt.xlabel('Timestep', fontsize=11)  # 设置横轴标签
plt.ylabel('Prediction Error (℃)', fontsize=11)  # 设置纵轴标签
plt.title('Error Time Series', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考横线
plt.grid(True, alpha=0.3)  # 显示网格线

# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 选中第 4 个子图
box_data = [errors]  # 把误差序列封装为箱线图输入
plt.boxplot(box_data, tick_labels=[model_name])  # 绘制误差箱线图
plt.ylabel('预测误差 (℃)', fontsize=11)  # 设置纵轴标签
plt.title('误差统计分布', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考横线
plt.grid(True, alpha=0.3, axis='y')  # 仅显示横向网格线

plt.tight_layout()  # 自动调整子图布局

error_filename = 'test_GRUECA_error_analysis.png'  # 指定误差图文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印保存提示

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("误差统计信息")  # 打印误差统计标题
print("=" * 60)  # 打印分隔线
print(f"误差均值:     {np.mean(errors):.4f} ℃")  # 打印误差均值
print(f"误差标准差:   {np.std(errors):.4f} ℃")  # 打印误差标准差
print(f"误差最大值:   {np.max(errors):.4f} ℃")  # 打印误差最大值
print(f"误差最小值:   {np.min(errors):.4f} ℃")  # 打印误差最小值
print(f"误差中位数:   {np.median(errors):.4f} ℃")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\n测试完成！")  # 打印全部测试完成标志
