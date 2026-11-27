"""
纯GRU模型定义
用于加载训练好的GRU模型并进行预测

本脚本用于推理阶段：读取“部分温湿度数据.csv”、做缺失值处理、用检查点中保存的
归一化器与配置构造 GRUModel 并加载权重，在新数据上输出 R²/MSE/RMSE/MAE，
绘制预测对比图与四联误差分析图（真值-预测散点、误差直方图、误差时序、箱线图）。
"""
import pandas as pd  # 导入 pandas 库，用于表格数据读取与处理（别名 pd）
import numpy as np  # 导入 NumPy 库，用于数组与数值计算（别名 np）
import matplotlib  # 导入 matplotlib 基础模块，用于配置绘图后端
matplotlib.use('Agg')  # 使用 Agg 非交互式后端，允许在无界面环境保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口（别名 plt）
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于必要时重新拟合归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R²、均方误差、平均绝对误差
import torch  # 导入 PyTorch 主库，用于张量计算与模型加载
import torch.nn as nn  # 导入神经网络模块，用于定义 GRU 模型结构
# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU 或 CPU
print(f"使用设备: {device}")  # 打印使用的设备
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ===================== 纯GRU模型定义 =====================
class GRUModel(nn.Module):  # 纯 GRU 预测模型定义
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
# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 提示开始加载测试数据
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的编码列表
data = None  # 初始化读取结果变量
for enc in encodings:  # 逐个编码尝试
    try:  # 捕获解码异常
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 读取成功即跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 换下一个编码继续尝试

if data is None:  # 所有编码都失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常终止
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 按索引×5分钟生成时间戳（分钟为单位）
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 时间起点
# 特征配置 - 必须与训练时的顺序一致
features = ['日龄', 'THI', '当前温度', '目标温度']  # 特征列表（顺序需与训练一致）
target = 'THI'  # 目标变量为 THI

# 过滤特征列
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征

print(f"使用特征: {features}")  # 打印实际使用的特征
print(f"数据形状: {data.shape}")  # 打印数据行列数

# 检查目标列是否存在
if target not in data.columns:  # 若缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带THI的CSV文件。")  # 抛出异常提示

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印处理标题
print(f"{'='*60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前数据行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计表头
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅处理数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比（百分比）
        if nan_count > 0:  # 仅打印存在缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与占比

# 定义连续变量和离散变量（两类采用不同填充策略）
continuous_vars = ['THI', '外界温度', 'CO2', '目标温度']  # 连续变量：线性插值 + 前后填充
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
print(f"{'='*60}\n")  # 打印结束分隔线

# 重置索引
data = data.reset_index(drop=True)  # 删除缺失行后重置索引，保证序列构造连续

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 提示开始加载模型

MODEL_PATH = 'Model_GRU_pytorch.pth'  # 模型检查点路径

try:  # 捕获文件不存在异常
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载检查点到指定设备
    print(f"成功加载模型: {MODEL_PATH}")  # 打印加载成功提示
except FileNotFoundError:  # 模型文件不存在
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛出异常提示

input_dim = checkpoint['input_dim']  # 从检查点读取输入维度
time_step = checkpoint['time_step']  # 从检查点读取时间步长

print(f"\n模型配置:")  # 打印模型配置标题
print(f"  输入维度: {input_dim}")  # 打印输入维度
print(f"  时间步长: {time_step}")  # 打印时间步长

# 检查测试数据是否包含所需特征
missing_features = [f for f in features if f not in data.columns]  # 找出测试数据中缺失的特征
if missing_features:  # 若存在缺失特征
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常

print(f"\n使用特征: {features}")  # 打印最终使用的特征

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
print("\n加载训练数据...")  # 提示开始加载训练数据
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的编码列表
train_data = None  # 初始化训练数据变量
for enc in encodings:  # 逐个编码尝试
    try:  # 捕获解码异常
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功即跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码错误
        continue  # 换下一个编码继续尝试
if train_data is None:  # 所有编码都失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止

# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 检查点包含归一化器
    print("\n使用模型保存的归一化器...")  # 打印提示
    feature_scaler = checkpoint['feature_scaler']  # 直接取训练时保存的特征归一化器
    target_scaler = checkpoint['target_scaler']  # 直接取训练时保存的目标归一化器
else:  # 检查点未保存归一化器
    print("\n使用训练数据重新拟合归一化器...")  # 打印提示
    feature_scaler = MinMaxScaler()  # 新建特征归一化器
    target_scaler = MinMaxScaler()  # 新建目标归一化器
    feature_scaler.fit(train_data[features])  # 用训练数据拟合特征归一化器
    target_scaler.fit(train_data[[target]])  # 用训练数据拟合目标归一化器

# 归一化测试数据
test_features = pd.DataFrame(  # 变换测试数据特征
    feature_scaler.transform(data[features]),  # 归一化数值
    columns=features  # 保持列名
)

print(f"  特征数量: {len(test_features.columns)}")  # 打印特征数量

# ===================== 序列构造 =====================
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟（需与训练一致）

def create_sequences(data, time_step):  # 创建时间序列样本
    """创建时间序列样本。

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
print("\n创建序列数据...")  # 提示创建序列数据
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 对目标列做归一化（二维）
# 创建序列
X_test = create_sequences(test_features, time_step)  # 构造测试输入序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 取对齐的归一化目标并拉平
y_test = y_test[:len(X_test)]  # 保证目标长度与样本数一致
print(f"\n测试数据形状:")  # 打印测试数据形状标题
print(f"  X_test: {X_test.shape}")  # 打印输入序列形状
print(f"  y_test: {y_test.shape}")  # 打印目标形状
# ===================== 构建模型 =====================
print("\n构建模型...")  # 提示构建模型
model = GRUModel(  # 实例化 GRU 模型
    input_dim=input_dim,  # 输入维度
    time_step=time_step,  # 时间步长
    gru_hidden=128,  # GRU 隐藏维度（需与训练一致）
    gru_layers=1,  # GRU 层数
    dropout=0.2  # dropout 比例
).to(device)  # 将模型移至设备
model.load_state_dict(checkpoint['model_state_dict'])  # 加载训练好的权重
model.eval()  # 切换评估模式
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印参数总量
# ===================== 模型预测 =====================
print("\n开始预测...")  # 提示开始预测
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试输入转张量并送设备

with torch.no_grad():  # 关闭梯度计算
    y_pred = model(X_test_tensor).cpu().numpy()  # 前向预测并转回 NumPy

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 真实目标反归一化到原始尺度
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 预测值反归一化到原始尺度

# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差
rmse = np.sqrt(mse)  # 均方误差开方得 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差

print("\n" + "=" * 60)  # 打印分隔线
print("纯GRU模型测试结果")  # 打印评估结果标题
print("=" * 60)  # 再次打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R²
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
plot_length = min(1600, len(y_test_rescaled))  # 限制绘图点数，最多 1600
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的预测值
# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成横轴步数索引
plt.figure(figsize=(15, 8))  # 创建预测对比图画布
y_test_flat = y_test_plot.flatten()  # 真实值拉平为一维
y_pred_flat = y_pred_plot.flatten()  # 预测值拉平为一维
plt.tick_params(axis='both', labelsize=16)  # 设置刻度字体大小
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线（蓝色）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值曲线（橙色）
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签
plt.ylabel('THI', fontsize=16)  # 设置纵轴标签
# plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 开启网格
plt.tight_layout()  # 紧凑布局
plt.text(0.5, 0.95, '(c)', transform=plt.gca().transAxes,  # 在图上添加子图编号“(c)”
         fontsize=16, fontweight='bold', va='top', ha='center')
plt.tight_layout()  # 再次紧凑布局
output_filename = 'test_GRU_prediction.png'  # 预测图保存文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 保存预测图，600 dpi
print(f"\n预测结果图已保存: {output_filename}")  # 打印保存提示
plt.close()  # 关闭画布释放内存

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算全体样本的预测误差（预测-真实）

# 仅用于绘图：取前300个点
max_points = min(300, len(errors))  # 限制误差分析绘图点数，最多 300
y_test_plot_err = y_test_rescaled[:max_points].flatten()  # 误差分析用真实值（前 300）
y_pred_plot_err = y_pred_rescaled[:max_points].flatten()  # 误差分析用预测值（前 300）
errors_plot = y_pred_plot_err - y_test_plot_err  # 误差分析用误差序列

print(f"\n误差分析图使用前 {max_points} 个数据点")  # 打印实际绘图点数

plt.figure(figsize=(15, 10))  # 创建四联误差分析画布

# 子图1: True value vs Predicted value（前300个点）
plt.subplot(2, 2, 1)  # 选中第 1 个子图
plt.scatter(  # 绘制真值-预测值散点图
    y_test_plot_err,  # 横轴：真实值
    y_pred_plot_err,  # 纵轴：预测值
    alpha=0.35,  # 点透明度
    s=45,  # 点大小
    color='#b58f86',  # 点颜色
    label='Predicted value'  # 图例文字
)
min_val = min(y_test_plot_err.min(), y_pred_plot_err.min())  # 取真实/预测的最小值作为坐标下界
max_val = max(y_test_plot_err.max(), y_pred_plot_err.max())  # 取真实/预测的最大值作为坐标上界
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')  # 绘制 y=x 理想参考线
plt.xlabel('True value', fontsize=11)  # 设置横轴标签
plt.ylabel('Predicted value', fontsize=11)  # 设置纵轴标签
plt.title('True value vs Predicted value', fontsize=12)  # 设置子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.4)  # 开启网格
margin = (max_val - min_val) * 0.05  # 计算坐标轴 5% 边距
plt.xlim(min_val - margin, max_val + margin)  # 设置横轴范围
plt.ylim(min_val - margin, max_val + margin)  # 设置纵轴范围

# 子图2: 误差分布直方图（前300个点）
plt.subplot(2, 2, 2)  # 选中第 2 个子图
plt.hist(errors_plot, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差直方图（50 个柱子）
plt.xlabel('Prediction Error', fontsize=11)  # 设置横轴标签
plt.ylabel('Frequency', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考竖线
plt.grid(True, alpha=0.3, axis='y')  # 开启纵轴网格

# 子图3: 误差时序图（前300个点）
plt.subplot(2, 2, 3)  # 选中第 3 个子图
error_steps = np.arange(len(errors_plot))  # 生成误差时序的横轴索引
plt.plot(error_steps, errors_plot, alpha=0.7, linewidth=1)  # 绘制误差随时间变化曲线
plt.xlabel('Timestep', fontsize=11)  # 设置横轴标签
plt.ylabel('Prediction Error', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Time Series (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考横线
plt.grid(True, alpha=0.3)  # 开启网格

# 子图4: 误差箱线图（前300个点）
plt.subplot(2, 2, 4)  # 选中第 4 个子图
plt.boxplot([errors_plot], tick_labels=['GRU'])  # 绘制误差箱线图
plt.ylabel('Prediction Error', fontsize=11)  # 设置纵轴标签
plt.title(f'Error Statistics Distribution (First {max_points} Points)', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 绘制零误差参考横线
plt.grid(True, alpha=0.3, axis='y')  # 开启纵轴网格

plt.tight_layout()  # 紧凑布局

error_filename = 'test_GRU_error_analysis.png'  # 误差分析图保存文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 保存误差分析图，600 dpi
print(f"误差分析图已保存: {error_filename}")  # 打印保存提示
plt.close()  # 关闭画布释放内存

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("误差统计信息")  # 打印统计标题
print("=" * 60)  # 再次打印分隔线
print(f"误差均值:     {np.mean(errors):.4f} ℃")  # 打印误差均值
print(f"误差标准差:   {np.std(errors):.4f} ℃")  # 打印误差标准差
print(f"误差最大值:   {np.max(errors):.4f} ℃")  # 打印误差最大值
print(f"误差最小值:   {np.min(errors):.4f} ℃")  # 打印误差最小值
print(f"误差中位数:   {np.median(errors):.4f} ℃")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\n测试完成！")  # 打印全部完成提示
