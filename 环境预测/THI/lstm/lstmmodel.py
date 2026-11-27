"""
纯LSTM模型定义
用于加载训练好的LSTM模型并进行预测

本脚本是 lstm.py 训练脚本对应的推理/测试脚本：它复用与训练阶段一致的
LSTMModel 网络结构，从磁盘加载已保存的 .pth 权重、归一化器与模型配置，
对“部分温湿度数据.csv”中的测试数据进行滑动窗口切分与预测，并输出
R2 / MSE / RMSE / MAE 评估指标、预测对比图以及多维度的误差分析图。
主要流程：数据读取与缺失值处理 -> 加载 checkpoint -> 重建模型并载入权重
-> 归一化与序列构造 -> 前向预测与反归一化 -> 指标计算与绘图。
"""
import pandas as pd  # 导入 pandas 库并重命名为 pd，用于表格数据的读取与处理
import numpy as np  # 导入 numpy 库并重命名为 np，用于数组与矩阵运算
import matplotlib  # 导入 matplotlib 基础模块，用于指定绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 模块并重命名为 plt，用于绘制图表
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标的归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、MSE、MAE 回归评估指标
import torch  # 导入 PyTorch 主库，用于张量计算与 GPU 加速
import torch.nn as nn  # 导入神经网络子库并重命名为 nn，用于搭建 LSTM 网络层

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据 GPU 是否可用选择计算设备
print(f"使用设备: {device}")  # 打印本次运行所使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体，以便正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，避免负号显示为方块


# ===================== 纯LSTM模型定义 =====================
class LSTMModel(nn.Module):  # 定义纯 LSTM 预测模型，继承 nn.Module
    """
    纯LSTM模型
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)

    网络由一层（可多层）LSTM 加一个全连接输出层构成，取窗口最后一个时间步的
    隐藏状态作为整段序列的表征来输出单个预测值。
    """
    def __init__(self, input_dim, time_step, lstm_hidden=128, lstm_layers=1, dropout=0.2):  # 构造函数，定义网络各层
        """
        初始化 LSTMModel 的各网络层。

        主要参数:
            input_dim (int): 每个时间步的输入特征维度。
            time_step (int): 输入序列的时间步长度。
            lstm_hidden (int): LSTM 隐藏层维度，默认 128。
            lstm_layers (int): LSTM 堆叠层数，默认 1。
            dropout (float): 多层 LSTM 之间的 dropout 概率，默认 0.2。
        """
        super(LSTMModel, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.time_step = time_step  # 保存时间步长度到实例属性
        self.input_dim = input_dim  # 保存输入特征维度到实例属性
        self.lstm_hidden = lstm_hidden  # 保存 LSTM 隐藏层维度到实例属性
        # LSTM层
        self.lstm = nn.LSTM(  # 定义 LSTM 层
            input_size=input_dim,  # 每个时间步输入的特征维度
            hidden_size=lstm_hidden,  # LSTM 隐藏状态维度
            num_layers=lstm_layers,  # LSTM 堆叠的层数
            dropout=dropout if lstm_layers > 1 else 0.0,  # 仅多层时启用 dropout（单层为 0）
            batch_first=True,  # 输入张量第一维为 batch
            bidirectional=False  # 使用单向 LSTM
        )
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden, 1)  # 全连接层：把隐藏状态映射为 1 个预测值

    def forward(self, x):  # 定义前向传播逻辑
        """
        前向传播：输入序列经 LSTM 编码后，用最后时刻的隐藏状态预测目标值。

        主要参数:
            x (Tensor): 形状为 (batch, time_step, input_dim) 的输入序列。

        返回:
            Tensor: 形状为 (batch, 1) 的预测值。
        """
        # x: (batch, time_step, input_dim)
        # LSTM编码
        out, (h, c) = self.lstm(x)  # out: (batch, time_step, hidden)
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)


# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始读取测试数据
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义依次尝试的 CSV 文件编码列表
data = None  # 初始化数据变量为 None，用于存放成功读取的 DataFrame
for enc in encodings:  # 遍历每一种候选编码
    try:
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 尝试用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 读取成功则打印所用编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 若出现解码错误则捕获并尝试下一种编码
        continue  # 继续尝试下一个编码

if data is None:  # 若所有编码均尝试失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常终止程序
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 以索引×5 分钟为偏移量生成时间戳
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 指定时间戳起始基准时间
# 特征配置
features = ['日龄', 'THI', '当前温度', '目标温度']  # 定义候选特征列名列表
target = 'THI'  # 定义预测目标列为 THI（温湿指数）

# 过滤特征列
features = [f for f in features if f in data.columns]  # 过滤掉数据中不存在的候选特征

print(f"使用特征: {features}")  # 打印实际参与预测的特征列表
print(f"数据形状: {data.shape}")  # 打印测试数据的行列数

# 检查目标列是否存在
if target not in data.columns:  # 若测试数据中缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带THI的CSV文件。")  # 抛出异常提示需使用带 THI 的文件

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线，标识缺失值处理环节开始
print("缺失值处理")  # 打印环节标题
print(f"{'='*60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据的行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计标题
for col in features + [target]:  # 遍历所有特征列以及目标列
    if col in data.columns:  # 仅统计数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值数量
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占总行数的百分比
        if nan_count > 0:  # 仅当存在缺失值时才打印
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印缺失值数量与占比

# 定义连续变量和离散变量
continuous_vars = ['THI', '目标温度', '当前温度']  # 定义连续型变量列表（适合线性插值）
discrete_vars = ['日龄']  # 定义离散型变量列表（只适合前后向填充）

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每一个连续型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录处理前的缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 对连续变量做双向线性插值
        data[col] = data[col].ffill()  # 用前一个有效值向前填充剩余缺失
        data[col] = data[col].bfill()  # 用后一个有效值向后填充剩余缺失
        after_nan = data[col].isna().sum()  # 记录处理后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后缺失值数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每一个离散型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录处理前的缺失值数量
        data[col] = data[col].ffill()  # 用前一个有效值向前填充缺失值
        data[col] = data[col].bfill()  # 用后一个有效值向后填充缺失值
        after_nan = data[col].isna().sum()  # 记录处理后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失值数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除在特征或目标列上仍含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据形状
print(f"{'='*60}\n")  # 打印分隔线，标识缺失值处理环节结束

# 重置索引
data = data.reset_index(drop=True)  # 重置索引，保证行号连续（删除行后会留下空档）

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 提示开始加载模型

MODEL_PATH = 'Model_LSTM_pytorch.pth'  # 定义模型权重文件的路径

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载 checkpoint 并映射到当前设备
    print(f"成功加载模型: {MODEL_PATH}")  # 打印加载成功提示
except FileNotFoundError:  # 若模型文件不存在
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛出更明确的异常

input_dim = checkpoint['input_dim']  # 从 checkpoint 中读取输入特征维度
time_step = checkpoint['time_step']  # 从 checkpoint 中读取时间窗口长度

print(f"\n模型配置:")  # 打印模型配置标题
print(f"  输入维度: {input_dim}")  # 打印输入维度
print(f"  时间步长: {time_step}")  # 打印时间步长

# 检查测试数据是否包含所需特征
missing_features = [f for f in features if f not in data.columns]  # 找出测试数据中缺失的特征
if missing_features:  # 若存在缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常终止

print(f"\n使用特征: {features}")  # 再次打印参与预测的特征列表

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
print("\n加载训练数据...")  # 提示开始读取训练数据（用于备用归一化器）
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义依次尝试的编码列表
train_data = None  # 初始化训练数据变量为 None
for enc in encodings:  # 遍历每一种候选编码
    try:
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 尝试用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 读取成功则打印所用编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 若出现解码错误则捕获并继续
        continue  # 尝试下一个编码
if train_data is None:  # 若所有编码均失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止
# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若 checkpoint 中保存了归一化器
    print("\n使用模型保存的归一化器...")  # 打印提示
    feature_scaler = checkpoint['feature_scaler']  # 直接复用训练时保存的特征归一化器
    target_scaler = checkpoint['target_scaler']  # 直接复用训练时保存的目标归一化器
else:  # 若 checkpoint 中未保存归一化器
    print("\n使用训练数据重新拟合归一化器...")  # 打印提示
    feature_scaler = MinMaxScaler()  # 新建特征归一化器
    target_scaler = MinMaxScaler()  # 新建目标归一化器
    feature_scaler.fit(train_data[features])  # 用训练数据拟合特征归一化器
    target_scaler.fit(train_data[[target]])  # 用训练数据拟合目标归一化器
# 归一化测试数据
test_features = pd.DataFrame(  # 构造归一化后的测试特征 DataFrame
    feature_scaler.transform(data[features].values),  # 用归一化器变换测试特征
    columns=features  # 指定列名与原特征一致
)
print(f"  特征数量: {len(test_features.columns)}")  # 打印归一化后的特征数量
# ===================== 序列构造 =====================
pred_offset = 6  # 定义预测偏移步数（与训练阶段一致）
def create_sequences(data, time_step):  # 定义函数：把特征表切分为滑动窗口序列样本
    """创建时间序列样本"""
    sequences = []  # 初始化列表，存放每一个滑动窗口样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点遍历，预留窗口与偏移长度
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行并转为 NumPy 数组
    return np.array(sequences)  # 堆叠为三维 NumPy 数组并返回

print("\n创建序列数据...")  # 提示开始构造序列
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 用目标归一化器变换测试数据的真实 THI
# 创建序列
X_test = create_sequences(test_features, time_step)  # 为归一化测试特征构造滑动窗口序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 取与窗口末端对应的归一化标签并展平
y_test = y_test[:len(X_test)]  # 截断标签使其长度与序列样本数一致
print(f"\n测试数据形状:")  # 打印测试数据形状标题
print(f"  X_test: {X_test.shape}")  # 打印测试序列的形状
print(f"  y_test: {y_test.shape}")  # 打印测试标签的形状
# ===================== 构建模型 =====================
print("\n构建模型...")  # 提示开始构建模型
model = LSTMModel(  # 实例化 LSTM 模型，结构参数与训练阶段保持一致
    input_dim=input_dim,  # 输入特征维度取自 checkpoint
    time_step=time_step,  # 时间窗口长度取自 checkpoint
    lstm_hidden=128,  # 隐藏层维度，需与训练时一致
    lstm_layers=1,  # LSTM 层数，需与训练时一致
    dropout=0.2  # dropout 概率（评估模式下不生效，但结构需一致）
).to(device)  # 将模型移至目标设备
model.load_state_dict(checkpoint['model_state_dict'])  # 将保存的权重加载到模型中
model.eval()  # 将模型切换为评估模式（关闭 dropout）
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型参数总量
# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始预测
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试序列转为张量并移至目标设备

with torch.no_grad():  # 关闭梯度计算
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并把结果搬回 CPU 转为 NumPy 数组
# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将归一化真实标签反归一化回原始 THI
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将归一化预测结果反归一化回原始 THI
# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 对 MSE 开方得到均方根误差 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE
print("\n" + "=" * 60)  # 打印分隔线
print("纯LSTM模型测试结果")  # 打印结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R2
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
plot_length = min(1600, len(y_test_rescaled))  # 确定绘图点数上限（最多 1600 个）
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的预测值
# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成绘图用的步数索引数组作为横轴
plt.figure(figsize=(15, 8))  # 创建 15×8 英寸的绘图画布
y_test_flat = y_test_plot.flatten()  # 将真实值展平为一维数组
y_pred_flat = y_pred_plot.flatten()  # 将预测值展平为一维数组
plt.tick_params(axis='both', labelsize=16)  # 设置两个坐标轴的刻度字体大小
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值折线（蓝色）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值折线（橙色）
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签
plt.ylabel('THI', fontsize=16)  # 设置纵轴标签
# plt.title(f'LSTM Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)  # （原注释）可选标题，已注释
plt.legend(fontsize=16)  # 显示图例并设置字体大小
plt.grid(True, alpha=0.3)  # 显示半透明网格
plt.tight_layout()  # 自动调整布局
plt.text(0.5, 0.95, '(a)', transform=plt.gca().transAxes,  # 在子图顶部中央添加子图编号 (a)
         fontsize=16, fontweight='bold', va='top', ha='center')  # 设置编号文本的字体与对齐方式
plt.tight_layout()  # 再次调整布局，确保文本不被裁切
output_filename = 'test_LSTM_prediction.png'  # 定义预测结果图的输出文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存预测对比图
print(f"\n预测结果图已保存: {output_filename}")  # 打印保存提示
plt.close()  # 关闭当前画布，释放内存

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算全部样本的预测误差（预测值减真实值）

# 仅用于绘图：取前300个点
max_points = min(300, len(errors))  # 确定误差分析图使用的点数上限（最多 300）
y_test_plot_err = y_test_rescaled[:max_points].flatten()  # 截取前 max_points 个真实值
y_pred_plot_err = y_pred_rescaled[:max_points].flatten()  # 截取前 max_points 个预测值
errors_plot = y_pred_plot_err - y_test_plot_err  # 计算用于绘图的预测误差

print(f"\n误差分析图使用前 {max_points} 个数据点")  # 打印实际使用的绘图点数

plt.figure(figsize=(15, 10))  # 创建 15×10 英寸的绘图画布

# 子图1: True value vs Predicted value（前300个点）
plt.subplot(2, 2, 1)  # 选择 2×2 布局中的第 1 个子图
plt.scatter(  # 绘制真实值-预测值的散点图
    y_test_plot_err,  # 横轴：真实值
    y_pred_plot_err,  # 纵轴：预测值
    alpha=0.35,  # 设置点的透明度
    s=45,  # 设置点的大小
    color='#b58f86',  # 设置点的颜色
    label='Predicted value'  # 设置图例标签
)
min_val = min(y_test_plot_err.min(), y_pred_plot_err.min())  # 求真实值与预测值中的最小值
max_val = max(y_test_plot_err.max(), y_pred_plot_err.max())  # 求真实值与预测值中的最大值
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')  # 绘制 y=x 参考对角线
plt.xlabel('True value', fontsize=11)  # 设置横轴标签
plt.ylabel('Predicted value', fontsize=11)  # 设置纵轴标签
plt.title('True value vs Predicted value', fontsize=12)  # 设置子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.4)  # 显示半透明网格
margin = (max_val - min_val) * 0.05  # 计算坐标范围的 5% 作为边距
plt.xlim(min_val - margin, max_val + margin)  # 设置横轴显示范围
plt.ylim(min_val - margin, max_val + margin)  # 设置纵轴显示范围

# 子图2: 误差分布直方图（前300个点）
plt.subplot(2, 2, 2)  # 选择第 2 个子图
plt.hist(errors_plot, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差分布直方图（50 个分箱）
plt.xlabel('Prediction Error', fontsize=11)  # 设置横轴标签
plt.ylabel('Frequency', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 绘制误差为 0 的红色参考竖线
plt.grid(True, alpha=0.3, axis='y')  # 仅显示横向网格

# 子图3: 误差时序图（前300个点）
plt.subplot(2, 2, 3)  # 选择第 3 个子图
error_steps = np.arange(len(errors_plot))  # 生成误差序列的步数索引作为横轴
plt.plot(error_steps, errors_plot, alpha=0.7, linewidth=1)  # 绘制误差随时间变化的折线
plt.xlabel('Timestep', fontsize=11)  # 设置横轴标签
plt.ylabel('Prediction Error', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Time Series (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制误差为 0 的红色参考横线
plt.grid(True, alpha=0.3)  # 显示半透明网格

# 子图4: 误差箱线图（前300个点）
plt.subplot(2, 2, 4)  # 选择第 4 个子图
plt.boxplot([errors_plot], tick_labels=['LSTM'])  # 绘制误差的箱线图（标签为 LSTM）
plt.ylabel('Prediction Error', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Statistics Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制误差为 0 的红色参考横线
plt.grid(True, alpha=0.3, axis='y')  # 仅显示横向网格

plt.tight_layout()  # 自动调整各子图布局
error_filename = 'test_LSTM_error_analysis.png'  # 定义误差分析图的输出文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印保存提示
plt.close()  # 关闭画布，释放内存

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
