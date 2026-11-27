"""
TCN (Temporal Convolutional Network) 模型定义
用于加载训练好的TCN模型并进行预测

本脚本是 tcn.py 的推理 companion：定义与训练时一致的 TCN 模型结构，
加载 Model_TCN_pytorch.pth 权重与归一化器，对新采集温湿度数据做温度偏差预测，
并输出预测曲线、误差分析四子图以及误差统计信息。
"""
import pandas as pd  # 导入 pandas 数据分析库并起别名 pd，用于 CSV 读取与表格处理
import numpy as np  # 导入 numpy 数值计算库并起别名 np，用于数组构造与误差统计
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用非交互式后端直接保存图片  # Agg 后端无需图形界面，适合离线出图
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口并起别名 plt，用于绘制预测与误差图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，作为无保存 scaler 时重新拟合的兜底方案
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R²、MSE、MAE 等回归评估指标
import torch  # 导入 PyTorch 主库，用于张量计算与模型加载
import torch.nn as nn  # 导入 PyTorch 神经网络模块并起别名 nn，用于搭建网络层
import torch.nn.functional as F  # 导入函数式接口并起别名 F（本脚本中未直接使用）

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有 GPU 则用 cuda，否则回退 cpu
print(f"使用设备: {device}")  # 打印推理使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭负号 Unicode 显示
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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
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

    由多个膨胀系数递增的 TemporalBlock 堆叠，取最后时刻通道输出经全连接层回归。
    """
    def __init__(self, input_dim, time_step, num_channels=[128], kernel_size=3, dropout=0.1):
        """初始化 TCN 模型。

        参数:
            input_dim: 每步输入特征数；time_step: 时间步数。
            num_channels: 各层通道数列表；kernel_size: 卷积核大小。
            dropout: 丢弃率，默认 0.1。
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
        """前向传播：输入序列 -> 单值预测。"""
        # x: (batch, time_step, input_dim)
        # TCN需要 (batch, input_dim, time_step)
        x = x.transpose(1, 2)  # (batch, input_dim, time_step)  # 转置适配 Conv1d
        
        # TCN编码
        out = self.network(x)  # (batch, num_channels[-1], time_step)  # 通过 TCN 主体
        
        # 使用最后时刻的特征
        out_last = out[:, :, -1]  # (batch, num_channels[-1])  # 取最后时间步通道输出
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 全连接层输出预测


# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始加载测试数据
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
data = None  # 初始化数据变量为空
for enc in encodings:  # 逐个编码尝试读取
    try:
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功编码
        break  # 成功即跳出
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码失败
        continue  # 尝试下一种编码

if data is None:  # 所有编码均失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛错终止

data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟步长与指定起点生成时间戳

# 特征配置
features = ['日龄','温度偏差','外界温度','CO2','目标温度']  # 预定义候选特征列表
target = '温度偏差'  # 指定目标列

# 过滤特征列
features = [f for f in features if f in data.columns]  # 仅保留实际存在的特征列

print(f"使用特征: {features}")  # 打印最终使用特征
print(f"数据形状: {data.shape}")  # 打印读入数据形状

# 检查目标列是否存在
if target not in data.columns:  # 目标列缺失无法评估
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带温度偏差的CSV文件。")  # 抛错提示

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 输出处理前形状
print(f"\n各列缺失值统计:")  # 提示逐列统计缺失
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 只统计存在列
        nan_count = data[col].isna().sum()  # 统计缺失数量
        nan_pct = nan_count / len(data) * 100  # 计算缺失占比
        if nan_count > 0:  # 仅输出有缺失列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印统计

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续变量用线性插值
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
        data[col] = data[col].fillna(method='ffill')  # 前向填充
        data[col] = data[col].fillna(method='bfill')  # 后向填充
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

# 重置索引
data = data.reset_index(drop=True)  # 删除行后重新连续编号

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 提示开始加载权重

MODEL_PATH = 'Model_TCN_pytorch.pth'  # 定义模型权重路径

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载检查点并映射到设备
    print(f"成功加载模型: {MODEL_PATH}")  # 打印成功提示
except FileNotFoundError:  # 文件不存在
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛错提示先训练

input_dim = checkpoint['input_dim']  # 读取输入特征维度
time_step = checkpoint['time_step']  # 读取时间窗口

print(f"\n模型配置:")  # 提示输出配置
print(f"  输入维度: {input_dim}")  # 打印输入维度
print(f"  时间步长: {time_step}")  # 打印时间步长

# 检查测试数据是否包含所需特征
missing_features = [f for f in features if f not in data.columns]  # 找出缺失特征
if missing_features:  # 有缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛错终止

print(f"\n使用特征: {features}")  # 打印最终特征

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
print("\n加载训练数据...")  # 提示加载训练数据用于兜底拟合
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
train_data = None  # 初始化为空
for enc in encodings:  # 逐个编码尝试读取
    try:
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功编码
        break  # 成功即跳出
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码失败
        continue  # 尝试下一编码
if train_data is None:  # 全部失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛错终止

# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 检查点含归一化器
    print("\n使用模型保存的归一化器...")  # 提示直接复用
    feature_scaler = checkpoint['feature_scaler']  # 取保存的特征归一化器
    target_scaler = checkpoint['target_scaler']  # 取保存的目标归一化器
else:  # 未保存归一化器
    print("\n使用训练数据重新拟合归一化器...")  # 提示重新拟合
    feature_scaler = MinMaxScaler()  # 新建特征归一化器
    target_scaler = MinMaxScaler()  # 新建目标归一化器
    feature_scaler.fit(train_data[features])  # 用训练特征拟合
    target_scaler.fit(train_data[[target]])  # 用训练目标拟合

# 归一化测试数据
test_features = pd.DataFrame(  # 构造归一化测试特征表
    feature_scaler.transform(data[features]),  # 变换测试特征
    columns=features  # 指定列名
)

print(f"  特征数量: {len(test_features.columns)}")  # 打印特征数量

# ===================== 序列构造 =====================
pred_offset = 6  # 预测偏移步数（与训练一致）

def create_sequences(data, time_step):
    """创建时间序列样本。

    参数:
        data: 归一化后的特征 DataFrame。
        time_step: 历史窗口长度。
    返回:
        np.ndarray，形状 (样本数, time_step, 特征数)。
    """
    sequences = []  # 初始化窗口列表
    for i in range(len(data) - time_step - pred_offset + 1):  # 滑动起点，预留窗口与偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 截取窗口转数组
    return np.array(sequences)  # 堆叠为三维数组返回

print("\n创建序列数据...")  # 提示构造序列
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 目标列归一化

# 创建序列
X_test = create_sequences(test_features, time_step)  # 构造测试输入序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 偏移截取并展平测试标签
X_test = X_test[:len(y_test)]  # 截断输入与标签对齐

print(f"\n测试数据形状:")  # 提示输出测试形状
print(f"  X_test: {X_test.shape}")  # 打印测试输入形状
print(f"  y_test: {y_test.shape}")  # 打印测试标签形状

# ===================== 构建模型 =====================
print("\n构建模型...")  # 提示构建模型
model = TCNModel(  # 按检查点参数实例化 TCN
    input_dim=input_dim,  # 输入维度来自检查点
    time_step=time_step,  # 时间步长来自检查点
    num_channels=[128],  # 通道配置（需与训练一致）
    kernel_size=3,  # 卷积核大小（需与训练一致）
    dropout=0.1  # 丢弃率（评估模式下不生效）
).to(device)  # 搬到计算设备

model.load_state_dict(checkpoint['model_state_dict'])  # 加载训练权重
model.eval()  # 切换评估模式

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印参数量

# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始推理
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试序列转张量并搬到设备

with torch.no_grad():  # 关闭梯度
    y_pred = model(X_test_tensor).cpu().numpy()  # 推理并转回 NumPy

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 真实标签反归一化
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 预测值反归一化

# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算 MSE
rmse = np.sqrt(mse)  # 计算 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算 MAE

print("\n" + "=" * 60)  # 打印分隔线
print("TCN模型测试结果")  # 打印结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 输出 R²
print(f"MSE:       {mse:.4f}")  # 输出 MSE
print(f"RMSE:      {rmse:.4f}")  # 输出 RMSE
print(f"MAE:       {mae:.4f}")  # 输出 MAE
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
# 只绘制前1600个时间步长
plot_length = min(1600, len(y_test_rescaled))  # 绘图长度取 1600 与样本数较小值
y_test_plot = y_test_rescaled[:plot_length]  # 截取真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取预测值
# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成步数索引横轴
plt.figure(figsize=(15, 8))  # 创建 15x8 英寸画布
y_test_flat = y_test_plot.flatten()  # 展平真实值
y_pred_flat = y_pred_plot.flatten()  # 展平预测值
plt.tick_params(axis='both', labelsize=16)  # 放大刻度字号
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实曲线（蓝）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测曲线（橙）
plt.xlabel('Timestep', fontsize=16)  # 横轴标签
plt.ylabel('TD (℃)', fontsize=16)  # 纵轴标签
# plt.title(f'TCN Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.tight_layout()  # 紧凑布局
plt.text(0.5, 0.95, '(b)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', va='top', ha='center')  # 顶部居中添加 (b) 子图编号
plt.tight_layout()  # 再次紧凑布局容纳文字
output_filename = 'test_TCN_prediction.png'  # 定义预测图文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 保存预测图
print(f"\n预测结果图已保存: {output_filename}")  # 打印提示

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算预测减真实的残差

plt.figure(figsize=(15, 10))  # 创建 15x10 英寸画布容纳四子图

# 子图1: 预测值 vs 真实值散点图
plt.subplot(2, 2, 1)  # 第 1 子图
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, s=20)  # 真实-预测散点
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],
         [y_test_rescaled.min(), y_test_rescaled.max()],
         'r--', lw=2, label='理想预测线')  # 绘制 y=x 理想参考线
plt.xlabel('真实温度偏差 (℃)', fontsize=11)  # 横轴标签
plt.ylabel('预测温度偏差 (℃)', fontsize=11)  # 纵轴标签
plt.title(f'预测值 vs 真实值 (R2={r2:.4f})', fontsize=12)  # 子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格

# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 第 2 子图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 50 分箱误差直方图
plt.xlabel('预测误差 (℃)', fontsize=11)  # 横轴标签
plt.ylabel('频数', fontsize=11)  # 纵轴标签
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 子图标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 零误差参考线
plt.grid(True, alpha=0.3, axis='y')  # 仅纵轴网格

# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 第 3 子图
error_steps = np.arange(min(120, len(errors)))  # 前 120 个误差点索引
plt.plot(error_steps, errors[:120], alpha=0.7, linewidth=1)  # 误差时序折线
plt.xlabel('Time Steps', fontsize=11)  # 横轴标签
plt.ylabel('Prediction Error (℃)', fontsize=11)  # 纵轴标签
plt.title('Error Time Series', fontsize=12)  # 子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 零误差参考线
plt.grid(True, alpha=0.3)  # 开启网格

# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 第 4 子图
box_data = [errors]  # 封装误差为箱线图输入
plt.boxplot(box_data, tick_labels=['TCN'])  # 绘制误差箱线图
plt.ylabel('预测误差 (℃)', fontsize=11)  # 纵轴标签
plt.title('误差统计分布', fontsize=12)  # 子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 零误差参考线
plt.grid(True, alpha=0.3, axis='y')  # 仅纵轴网格

plt.tight_layout()  # 紧凑布局

error_filename = '../../test_TCN_error_analysis.png'  # 定义误差图输出路径（上两级目录）
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印提示

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("误差统计信息")  # 打印统计标题
print("=" * 60)  # 打印分隔线
print(f"误差均值:     {np.mean(errors):.4f} ℃")  # 输出误差均值
print(f"误差标准差:   {np.std(errors):.4f} ℃")  # 输出误差标准差
print(f"误差最大值:   {np.max(errors):.4f} ℃")  # 输出最大误差
print(f"误差最小值:   {np.min(errors):.4f} ℃")  # 输出最小误差
print(f"误差中位数:   {np.median(errors):.4f} ℃")  # 输出误差中位数
print("=" * 60)  # 打印分隔线

print("\n测试完成！")  # 打印全部完成标志
