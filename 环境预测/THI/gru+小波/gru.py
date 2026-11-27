"""
GRU + 小波变换模型训练与评估脚本（THI 温湿度指数预测）

本脚本实现基于 PyTorch 的「滑动窗口小波分解 + GRU」时序预测的完整流程：
1. 尝试多种编码读取温湿度 CSV 训练数据；
2. 按连续变量（插值+前后填充）与离散变量（前后填充）分别处理缺失值；
3. 用「仅对 THI 做滑动窗口小波分解（db4，3 层）、其余特征保持不变」的方式构造特征，
   滑动窗口只使用当前及历史数据，避免未来信息泄漏；
4. 按 6:4 划分训练/测试集（在小波分解之后），仅在训练集上拟合归一化器；
5. 按滑动时间窗口构造序列样本（输入为历史窗口，标签为若干步之后的 THI）；
6. 定义 GRU 预测模型，从训练集中切出 20% 作为验证集进行带早停的训练；
7. 绘制损失曲线，在测试集与新数据上评估 R²/RMSE/MAE；
8. 保存模型检查点（含小波参数与归一化器）与各类可视化图片。

数据输入: 温湿度数据.csv（训练/测试）、部分温湿度数据.csv.csv（新数据）
产物输出: training_loss/test_prediction/new_data_prediction 等 PNG 图片、Model_GRUWavelet_pytorch.pth
"""
import pandas as pd  # 导入 pandas 库，用于 CSV 读取与 DataFrame 数据处理
import numpy as np  # 导入 numpy 库，用于数组与数值计算
import random  # 导入 random 库（随机种子设置当前已注释，保留备用）
import os  # 导入 os 库（环境变量设置当前已注释，保留备用）
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，用于提前指定绘图后端
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片，无需图形界面
import matplotlib.pyplot as plt  # 导入 pyplot 子模块，提供绘图接口
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标的归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入回归评估指标：R²、MSE、MAE
from sklearn.model_selection import train_test_split, KFold  # 导入数据集划分与 K 折交叉验证工具（当前用手写划分）
import copy  # 导入 copy 库，用于深拷贝模型权重（早停时恢复最佳权重）
import torch  # 导入 PyTorch 主库，提供张量计算与设备支持
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU、全连接层等
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入数据集与数据加载器，用于批处理训练
import pywt  # 小波变换库  # 导入 PyWavelets 库，用于离散小波分解（wavedec）
# def set_seed(seed=10):
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
#
# set_seed(42)
# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU（可用时）或 CPU 作为训练设备
print(f"使用设备: {device}")  # 打印本次运行使用的计算设备
if torch.cuda.is_available():  # 若检测到可用 GPU，进一步打印其详细信息
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号，避免坐标轴负号显示为方块
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，逐一尝试直到成功读取
data = None  # 初始化数据变量为空，稍后由成功读取的编码赋值
for enc in encodings:  # 遍历每种候选编码尝试读取 CSV
    try:
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据文件
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功后跳出循环，不再尝试后续编码
    except (UnicodeDecodeError, UnicodeError):  # 若出现解码错误则尝试下一种编码
        continue  # 继续尝试下一个候选编码

if data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔为每行构造时间戳
features = ['日龄', 'THI', '当前温度', '目标温度']  # 预定义的候选特征列名称
target = 'THI'  # 指定预测目标为 THI
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列，避免缺列报错
# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印分隔线，标识缺失值处理阶段开始
print("缺失值处理")  # 打印阶段标题
print(f"{'=' * 60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前数据的行列数
print(f"\n各列缺失值统计:")  # 打印缺失值统计表标题
for col in features + [target]:  # 遍历所有特征列与目标列
    if col in data.columns:  # 仅统计数据中存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算缺失值占比（百分比）
        if nan_count > 0:  # 仅打印确实存在缺失值的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与占比

# 定义连续变量和离散变量
continuous_vars = ['THI','目标温度', '当前温度']  # 连续型变量列表，采用插值方式填充
discrete_vars = ['日龄']  # 离散型变量列表，采用前后向填充


# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值填充（双向）
        data[col] = data[col].fillna(method='ffill')  # 用前一个有效值填充剩余缺失（主要填补开头）
        data[col] = data[col].fillna(method='bfill')  # 用后一个有效值填充剩余缺失（主要填补结尾）
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理存在且含缺失值的列
        print(f"\n处理 {col}:")  # 打印当前处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        data[col] = data[col].fillna(method='ffill')  # 用前一个有效值填充离散变量缺失值
        data[col] = data[col].fillna(method='bfill')  # 用后一个有效值填充开头残留的缺失值
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计所有特征与目标列残留的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 提示将删除含缺失值的行
    data = data.dropna(subset=features + [target])  # 删除特征或目标中含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据规模
print(f"{'='*60}\n")  # 打印分隔线，标识缺失值处理阶段结束

print(f"\n【GRU + 小波变换模型】")  # 打印所用模型名称
print(f"使用特征: {features}")  # 打印最终使用的特征列表
print(f"特征数量: {len(features)}")  # 打印特征总数

# ===================== 小波变换特征提取（仅对THI） =====================
def wavelet_decompose_sliding(data, target_col='THI', wavelet='db4', level=3, window_size=None):
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（THI）进行小波分解，其他列保持原样
    每个时间点只使用历史数据
    
    参数:
        data: 输入数据 (DataFrame)，行代表时间步。
        target_col: 需要进行小波分解的列名（默认'THI'）。
        wavelet: 小波基函数 ('db4', 'sym5', 'coif3' 等)。
        level: 小波分解层数。
        window_size: 滑动窗口大小，None 则自动计算。
    返回:
        分解后的特征 (DataFrame)，含目标列的近似/细节系数与其他原始列。
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印分解开始提示与目标列
    print(f"小波基函数: {wavelet}")  # 打印所用的小波基函数
    print(f"分解层数: {level}")  # 打印分解层数
    
    # 自动计算窗口大小：确保足够大以进行有效的小波分解
    if window_size is None:  # 若未显式指定窗口大小
        window_size = min(2 ** (level + 6), len(data))  # 根据分解层数自适应  # 取 2^(层数+6) 与数据长度较小者
    
    print(f"滑动窗口大小: {window_size}")  # 打印实际使用的窗口大小
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建空结果表，索引与原数据一致
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 跳过时间戳列，不参与特征构造
            continue  # 直接进入下一列
        
        signal = data[col].values  # 取出当前列信号为一维 numpy 数组
        
        # 只对目标列进行小波分解
        if col == target_col:  # 若当前列是分解目标列（THI）
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 收集各时间点的近似系数
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数创建列表
            
            for i in range(len(signal)):  # 逐时间点做滑动窗口分解
                # 只使用当前时间点之前的数据（包括当前点）
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，保证不越界到未来
                window_signal = signal[start_idx:i+1]  # 截取截止当前的窗口信号（仅历史+当前）
                
                # 如果窗口太小，用边界值填充到window_size
                if len(window_signal) < window_size:  # 若序列开头窗口不足长度
                    pad_width = window_size - len(window_signal)  # 计算需要填充的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值在左侧补齐
                
                # 小波分解
                try:
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层离散小波分解
                    
                    # 取最后一个值作为当前时间点的特征
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为当前点近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层细节系数末值，空则补 0
                except Exception as e:  # 若分解过程出错
                    # 如果分解失败，使用原始值
                    approx_list.append(window_signal[-1])  # 用窗口末值近似代替近似系数
                    for j in range(level):  # 遍历每一层
                        detail_lists[j].append(0.0)  # 细节系数用 0 填充
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数特征列
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入第 j+1 层细节系数特征列
        else:
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印未分解的列名
            wavelet_features[col] = signal  # 其他列直接原样复制
    
    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印生成的特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印小波特征数量（1 近似 + level 细节）
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印保留的原始特征数量
    return wavelet_features  # 返回构造好的小波特征表

# ===================== 数据集划分 =====================
time_step = 36  # GRU时间窗口：48步 × 5分钟 = 4小时
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印输入窗口长度及对应分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移及对应分钟数

# 划分数据集
split_index = int(len(data) * 0.6)  # 按 60% 位置计算切分索引
train_data_raw = data.iloc[:split_index].copy()  # 取前 60% 作为原始训练数据副本
test_data_raw = data.iloc[split_index:].copy()  # 取后 40% 作为原始测试数据副本

print(f"\n数据集划分:")  # 打印数据集划分结果标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw)/len(data)*100:.1f}%)")  # 打印训练集样本数及占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw)/len(data)*100:.1f}%)")  # 打印测试集样本数及占比

# ===================== 小波分解 =====================
# 使用与GRU相同的时间窗口
wavelet_window = time_step  # 让小波窗口和GRU窗口一致  # 将小波滑动窗口设为与 GRU 时间窗口一致
print(f"\n小波窗口大小设置为: {wavelet_window} 步 (与GRU时间窗口一致)")  # 打印小波窗口设置

print("\n对训练集进行小波分解（仅分解THI）...")  # 打印训练集分解提示
train_wavelet_features = wavelet_decompose_sliding(train_data_raw[features], target_col='THI', wavelet='db4', level=3, window_size=wavelet_window)  # 对训练集特征做小波分解

print("\n对测试集进行小波分解（仅分解THI）...")  # 打印测试集分解提示
test_wavelet_features = wavelet_decompose_sliding(test_data_raw[features], target_col='THI', wavelet='db4', level=3, window_size=wavelet_window)  # 对测试集特征做小波分解

# 直接使用小波分解后的特征（已包含原始特征和小波特征）
train_features = train_wavelet_features  # 训练集特征取小波分解结果
test_features = test_wavelet_features  # 测试集特征取小波分解结果
feature_names = list(train_wavelet_features.columns)  # 记录小波特征列名列表

print(f"\n使用模式: THI小波分解 + 其他原始特征")  # 打印特征使用模式
print(f"总特征数: {len(feature_names)}")  # 打印特征总维度
print(f"  - THI的小波特征: 4 个（1个近似系数 + 3个细节系数）")  # 说明 THI 小波特征构成
print(f"  - 其他原始特征: {len(features) - 1} 个")  # 说明保留的原始特征数量
print(f"特征列表: {feature_names}")  # 打印全部特征名称

# ===================== 归一化 =====================
print("\n归一化处理...")  # 打印归一化阶段提示
feature_scaler = MinMaxScaler()  # 创建特征归一化器（缩放到 0~1）
target_scaler = MinMaxScaler()  # 创建目标归一化器（缩放到 0~1）

# 只在训练集上fit
feature_scaler.fit(train_features)  # 仅用训练特征拟合归一化器，避免数据泄漏
target_scaler.fit(train_data_raw[[target]])  # 仅用训练目标拟合目标归一化器

# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 归一化训练特征并重建 DataFrame
    feature_scaler.transform(train_features),  # 缩放训练特征
    columns=feature_names  # 保持特征列名
)
test_features = pd.DataFrame(  # 归一化测试特征并重建 DataFrame
    feature_scaler.transform(test_features),  # 缩放测试特征
    columns=feature_names  # 保持特征列名
)

train_target = target_scaler.transform(train_data_raw[[target]]).flatten()  # 归一化训练目标并展平
test_target = target_scaler.transform(test_data_raw[[target]]).flatten()  # 归一化测试目标并展平

# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):
    """按滑动窗口把二维特征表转换为三维时序样本。

    参数:
        data: 归一化后的特征 DataFrame。
        time_step: 每个样本包含的历史时间步数。
    返回:
        np.ndarray，形状为 (样本数, time_step, 特征数)。
    """
    sequences = []  # 收集各滑动窗口样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，保证窗口与偏移不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取第 i 个历史窗口并转数组
    return np.array(sequences)  # 堆叠成三维数组返回

print("\n创建训练集序列...")  # 打印训练集序列构造提示
X_train = create_sequences(train_features, time_step)  # 构造训练集输入序列
y_train = train_target[time_step + pred_offset - 1:]  # 取与窗口对应的目标（偏移后的时刻）
y_train = y_train[:len(X_train)]  # 裁剪标签使其与样本数一致

print("\n创建测试集序列...")  # 打印测试集序列构造提示
X_test = create_sequences(test_features, time_step)  # 构造测试集输入序列
y_test = test_target[time_step + pred_offset - 1:]  # 取测试集对应目标（偏移后的时刻）
y_test = y_test[:len(X_test)]  # 裁剪标签使其与样本数一致

print(f"\n序列数据形状:")  # 打印序列形状信息标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集输入与标签形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集输入与标签形状

# ===================== GRU+小波模型定义 =====================
class GRUWaveletModel(nn.Module):
    """
    GRU + 小波变换模型（仅小波特征）
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):
        """初始化 GRU+小波预测模型。

        参数:
            input_dim: 每步输入特征维度（含小波特征）。
            time_step: 输入序列时间步数。
            gru_hidden: GRU 隐藏层维度，默认 128。
            gru_layers: GRU 层数，默认 1（单层时关闭 dropout）。
            dropout: 多层 GRU 层间丢弃率，默认 0.2。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类初始化方法
        self.time_step = time_step  # 保存时间步长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层维度
        
        # GRU层
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
        """前向传播：GRU 编码 -> 取末时刻 -> 全连接预测。

        参数:
            x: (batch, time_step, input_dim) 输入序列。
        返回:
            (batch, 1) 的预测张量。
        """
        # x: (batch, time_step, input_dim)
        # GRU编码
        out, h = self.gru(x)  # out: (batch, time_step, hidden)  # GRU 输出各时刻状态与最终隐状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)  # 取最后时间步输出作为整体表征
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 经全连接层得到预测值并返回


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
        self.mode = mode  # 保存优化方向
        self.restore_best_weights = restore_best_weights  # 保存是否恢复最佳权重开关
        self.verbose = verbose  # 保存是否打印日志开关

        self.counter = 0  # 初始化未提升计数器
        self.best_score = None  # 初始化最佳指标值
        self.best_epoch = 0  # 记录最佳指标出现轮次
        self.early_stop = False  # 早停标志
        self.best_model = None  # 存放最佳模型权重副本

    def _is_improvement(self, score):
        """判断本次指标是否优于历史最佳。

        参数:
            score: 当前轮次验证指标。
        返回:
            bool，True 表示有提升。
        """
        if self.best_score is None:  # 首次调用视为提升
            return True  # 直接返回 True

        if self.mode == 'min':  # 最小化模式
            return score < self.best_score - self.min_delta  # 需小于最佳值一定幅度
        elif self.mode == 'max':  # 最大化模式
            return score > self.best_score + self.min_delta  # 需大于最佳值一定幅度
        else:  # 非法模式
            raise ValueError("mode must be 'min' or 'max'")  # 抛出异常

    def __call__(self, score, model, epoch=None):
        """每轮训练后调用，更新最佳指标并判断是否早停。

        参数:
            score: 当前轮验证指标。
            model: 当前模型（提升时深拷贝其权重）。
            epoch: 当前轮次编号，可选。
        返回:
            None。
        """
        # 防止 nan / inf 干扰
        if score is None or np.isnan(score) or np.isinf(score):  # 指标非法时
            if self.verbose:  # 若开启日志
                print(f"EarlyStopping warning: invalid score = {score}")  # 打印警告
            self.counter += 1  # 计入未提升次数
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停
            return  # 返回

        if self._is_improvement(score):  # 若有提升
            self.best_score = score  # 更新最佳指标
            self.best_epoch = epoch if epoch is not None else self.best_epoch  # 记录最佳轮次
            self.best_model = copy.deepcopy(model.state_dict())  # 深拷贝保存最佳权重
            self.counter = 0  # 重置计数器
            if self.verbose:  # 若开启日志
                print(f"验证指标提升，最佳值更新为: {self.best_score:.6f}")  # 打印提升信息
        else:  # 无提升
            self.counter += 1  # 计数加一
            if self.verbose:  # 若开启日志
                print(f"验证指标未提升，计数器: {self.counter}/{self.patience}")  # 打印计数
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停

    def restore(self, model):
        """在开启开关且存在最佳权重时，将模型恢复到最佳状态。

        参数:
            model: 待恢复权重的模型。
        返回:
            None。
        """
        if self.restore_best_weights and self.best_model is not None:  # 若需恢复且有最佳权重
            model.load_state_dict(self.best_model)  # 加载最佳权重
            if self.verbose:  # 若开启日志
                print(f"已恢复到最佳权重（epoch={self.best_epoch}, best_score={self.best_score:.6f}）")  # 打印恢复信息


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):
    """训练模型并在验证集上早停。

    参数:
        model: 待训练模型。
        train_loader: 训练集 DataLoader。
        val_loader: 验证集 DataLoader。
        epochs: 最大训练轮数，默认 200。
        lr: Adam 学习率，默认 0.001。
        patience: 早停容忍轮数，默认 10。
    返回:
        (train_losses, val_losses)：每轮训练/验证损失列表。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器

    train_losses = []  # 记录训练损失
    val_losses = []  # 记录验证损失

    for epoch in range(epochs):  # 按轮次循环
        # 训练阶段
        model.train()  # 切换训练模式
        train_loss = 0.0  # 初始化累计训练损失
        for batch_x, batch_y in train_loader:  # 遍历训练批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 数据移动到设备

            optimizer.zero_grad()  # 清空梯度
            outputs = model(batch_x)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item()  # 累加批次损失

        train_loss /= len(train_loader)  # 求平均训练损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换评估模式
        val_loss = 0.0  # 初始化累计验证损失
        with torch.no_grad():  # 关闭梯度计算
            for batch_x, batch_y in val_loader:  # 遍历验证批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 数据移动到设备
                outputs = model(batch_x)  # 前向传播
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()  # 累加批次验证损失

        val_loss /= len(val_loader)  # 求平均验证损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每 10 轮打印一次
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印损失进展

        # 早停检查
        early_stopping(val_loss, model, epoch=epoch + 1)  # 更新早停状态
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停轮次
            early_stopping.restore(model)  # 恢复最佳权重
            break  # 结束训练

    # 加载最佳模型
    early_stopping.restore(model)  # 确保最终为最佳权重

    return train_losses, val_losses  # 返回损失记录


# ===================== K折交叉验证训练 =====================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(feature_names)  # 使用小波特征数  # 输入维度取小波特征总数
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
#     # 构建GRU+小波模型
#     model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2).to(device)
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
# plt.savefig('kfold_training_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRUWavelet_pytorch.png")

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据，从训练集中划分验证集用于早停）")  # 打印阶段标题
print(f"{'=' * 50}")  # 打印分隔线

# 从训练集中划分10%作为最终模型训练时的验证集（早停用），不使用测试集
final_val_split = int(len(X_train) * 0.8)  # 按 80% 计算训练集内部切分点
X_final_train = X_train[:final_val_split]  # 实际训练部分
y_final_train = y_train[:final_val_split]  # 对应训练标签
X_final_val = X_train[final_val_split:]  # 验证部分（后 20%）
y_final_val = y_train[final_val_split:]  # 对应验证标签

print(f"最终训练集样本数: {len(X_final_train)}")  # 打印最终训练样本数
print(f"最终验证集样本数（早停用，来自训练集）: {len(X_final_val)}")  # 打印内部验证样本数
print(f"测试集样本数（仅用于最终评估）: {len(X_test)}")  # 打印测试样本数

X_train_tensor = torch.FloatTensor(X_final_train)  # 训练序列转张量
y_train_tensor = torch.FloatTensor(y_final_train).unsqueeze(1)  # 训练标签转张量并升维
X_val_tensor = torch.FloatTensor(X_final_val)  # 验证序列转张量
y_val_tensor = torch.FloatTensor(y_final_val).unsqueeze(1)  # 验证标签转张量并升维
X_test_tensor = torch.FloatTensor(X_test)  # 测试序列转张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试标签转张量并升维

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 构造训练数据集
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # 构造验证数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 构造测试数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练加载器，打乱
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试加载器

final_model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2).to(device)  # 实例化最终模型并移至设备
print(final_model)  # 打印模型结构
print("\n开始训练最终模型...")  # 打印训练开始提示
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印模型参数总数

# 使用从训练集划分的验证集进行早停，不使用测试集
train_losses, val_losses = train_model(final_model, train_loader, val_loader,
                                       epochs=200, lr=0.001, patience=10)  # 训练最终模型

print(f"\n训练完成！最终验证损失（训练集内验证）: {min(val_losses):.6f}")  # 打印最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置标题
plt.xlabel('训练轮数 (Epoch)')  # 设置 X 轴标签
plt.ylabel('损失值 (MSE)')  # 设置 Y 轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.tight_layout()  # 调整布局
plt.savefig('training_loss_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 保存损失曲线图
print("训练损失曲线已保存: training_loss_GRUWavelet_pytorch.png")  # 打印保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 打印测试预测提示

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将测试标签反归一化为真实 THI

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试序列转张量并移至设备

final_model.eval()  # 切换评估模式
with torch.no_grad():  # 关闭梯度
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 预测并取回 numpy 结果

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测值反归一化为真实 THI

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算 MSE
rmse_test = np.sqrt(mse_test)  # 计算 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果（GRU+小波）")  # 打印评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R^2:  {r2:.4f}")  # 打印 R²
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建画布绘制测试对比
y_test_rescaled_flat = y_test_rescaled.flatten()  # 展平真实值
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 展平预测值

test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset - 1:]  # 按偏移对齐测试时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 裁剪时间戳使其与绘图数据等长

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实THI', linewidth=2)  # 绘制真实 THI 曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+小波预测', linewidth=2)  # 绘制预测 THI 曲线
plt.xlabel('时间')  # 设置 X 轴标签
plt.ylabel('THI ')  # 设置 Y 轴标签
plt.title(f'测试集THI预测结果 (R^2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置含指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.savefig('test_prediction_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 保存测试预测图
print("测试集预测结果已保存: test_prediction_GRUWavelet_pytorch.png")  # 打印保存提示

# ===================== 模型保存 =====================
torch.save({  # 保存模型检查点字典
    'model_state_dict': final_model.state_dict(),  # 保存模型权重
    'input_dim': input_dim,  # 保存输入维度
    'time_step': time_step,  # 保存时间窗口
    'feature_names': feature_names,  # 保存特征名列表
    'original_features': features,  # 保存原始特征名列表
    'wavelet': 'db4',  # 保存小波基函数类型
    'level': 3,  # 保存分解层数
    'wavelet_window': wavelet_window,  # 保存小波窗口大小
    'wavelet_target_col': 'THI',  # 保存小波分解的目标列
    'use_wavelet_only': False,  # 使用THI小波特征 + 其他原始特征  # 标记是否只用小波特征
    'feature_scaler': feature_scaler,  # 保存特征归一化器
    'target_scaler': target_scaler,  # 保存目标归一化器
}, 'Model_GRUWavelet_pytorch.pth')  # 指定检查点文件名
print("模型已保存: Model_GRUWavelet_pytorch.pth")  # 打印保存提示
# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 打印新数据处理提示
new_data = pd.read_csv('部分温湿度数据.csv.csv')  # 读取待预测的新数据文件
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',
                                       origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟间隔生成时间戳
# 对新数据进行小波分解（仅分解THI）
print("\n对新数据进行滑动窗口小波分解（仅分解THI）...")  # 打印分解提示
new_wavelet_features = wavelet_decompose_sliding(new_data[features], target_col='THI', wavelet='db4', level=3, window_size=wavelet_window)  # 对新数据特征做小波分解
# 归一化
new_features = pd.DataFrame(  # 用训练集归一化器处理新特征
    feature_scaler.transform(new_wavelet_features),  # 缩放新特征
    columns=feature_names  # 保持特征列名
)

# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据 THI 的原始真实值（未归一化）

X_new = create_sequences(new_features, time_step)  # 构造新数据输入序列
y_new = y_new_original[time_step + pred_offset:]  # 按偏移对齐新数据标签
X_new = X_new[:len(y_new)]  # 裁剪输入序列使其与标签等长

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 新数据序列转张量并移至设备

final_model.eval()  # 切换评估模式
with torch.no_grad():  # 关闭梯度
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 预测并取回 numpy 结果

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 预测值反归一化为真实 THI
y_new_rescaled = y_new.reshape(-1, 1)  # 真实值重塑为二维列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MSE
rmse_new = np.sqrt(mse_new)  # 计算新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果（GRU+小波）")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R²:   {r2_new:.4f}")  # 打印新数据 R²
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线
timestamps = new_data['timestamp'][time_step + pred_offset:]  # 按偏移截取新数据时间戳
plt.figure(figsize=(15, 5))  # 创建画布绘制新数据对比
y_new_rescaled_flat = y_new_rescaled.flatten()  # 展平真实值
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 展平预测值
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实THI', linewidth=2)  # 绘制真实曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='GRU+小波预测', linewidth=2)  # 绘制预测曲线
plt.xlabel('时间')  # 设置 X 轴标签
plt.ylabel('THI ')  # 设置 Y 轴标签
plt.title(f'新数据集THI预测结果 (R²: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置含指标标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.savefig('new_data_prediction_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 保存新数据预测图
print("新数据集预测结果已保存: new_data_prediction_GRUWavelet_pytorch.png")  # 打印保存提示
print("\n========== 所有任务完成（GRU+小波模型） ==========")  # 打印全部任务完成提示
