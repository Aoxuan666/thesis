"""
GRU+小波变换模型测试脚本
用于加载训练好的GRU+小波模型并进行预测

本脚本是 Model_GRUWavelet_pytorch.pth 检查点的推理/评估端，主要流程：
1. 定义与训练端一致的 GRUWaveletModel 以及「仅对 THI 做滑动窗口小波分解」的函数；
2. 多编码尝试读取测试数据，处理时间戳、缺失值与索引；
3. 从检查点读取输入维度、时间窗口、小波参数（基函数/层数/窗口/目标列）和归一化器；
4. 兼容新旧两类模型：新版仅分解 THI，旧版对所有特征做小波分解（内联兼容函数）；
5. 对训练/测试数据做小波分解与归一化，按滑动窗口构造序列；
6. 加载权重进行推理，计算 R²/MSE/RMSE/MAE；
7. 输出预测对比图、误差分析四子图及误差统计信息。
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
import pywt  # 小波变换库  # 导入 PyWavelets 库，用于离散小波分解

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择 GPU（可用时）或 CPU 作为推理设备
print(f"使用设备: {device}")  # 打印本次运行使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号，避免坐标轴负号显示为方块
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
# ===================== 小波变换函数（仅对THI） =====================
def wavelet_decompose_sliding(data, target_col='THI', wavelet='db4', level=3, window_size=None):
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（THI）进行小波分解，其他列保持原样
    
    参数:
        data: 输入特征表 (DataFrame)。
        target_col: 需要分解的目标列名，默认 'THI'。
        wavelet: 小波基函数，默认 'db4'。
        level: 分解层数，默认 3。
        window_size: 滑动窗口大小，None 时自适应计算。
    返回:
        含小波系数与其他原始列的特征 DataFrame。
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印分解提示与目标列
    print(f"小波基函数: {wavelet}, 分解层数: {level}")  # 打印小波基与层数
    
    if window_size is None:  # 未指定窗口时
        window_size = min(2 ** (level + 6), len(data))  # 自适应取 2^(层数+6) 与数据长度较小者
    
    print(f"滑动窗口大小: {window_size}")  # 打印实际窗口大小
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建结果表，索引与原数据一致
    
    for col in data.columns:  # 遍历每一列
        if col == 'timestamp':  # 跳过时间戳列
            continue  # 进入下一列
        
        signal = data[col].values  # 取当前列为一维数组信号
        
        # 只对目标列进行小波分解
        if col == target_col:  # 若是目标列（THI）
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 收集近似系数
            detail_lists = [[] for _ in range(level)]  # 每层细节系数各建一个列表
            
            for i in range(len(signal)):  # 逐时间点滑动分解
                start_idx = max(0, i - window_size + 1)  # 窗口起点，避免越界到未来
                window_signal = signal[start_idx:i+1]  # 截止当前的窗口信号（仅历史+当前）
                
                if len(window_signal) < window_size:  # 开头窗口不足时
                    pad_width = window_size - len(window_signal)  # 计算需补齐长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值左侧补齐
                
                try:
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # level 层离散小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数末值作为当前点特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取末值，空则补 0
                except Exception as e:  # 分解失败时
                    approx_list.append(window_signal[-1])  # 用窗口末值代替近似系数
                    for j in range(level):  # 遍历各层
                        detail_lists[j].append(0.0)  # 细节系数补 0
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数特征列
            for j in range(level):  # 遍历各层
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入第 j+1 层细节特征列
        else:
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印未分解的列名
            wavelet_features[col] = signal  # 原样复制其他列
    
    print(f"小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印生成特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印小波特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印保留原始特征数量
    return wavelet_features  # 返回构造好的特征表
# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印测试数据加载提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
data = None  # 初始化数据变量为空
for enc in encodings:  # 遍历每种候选编码
    try:
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功编码
        break  # 成功则跳出
    except (UnicodeDecodeError, UnicodeError):  # 解码失败继续
        continue  # 尝试下一种编码
if data is None:  # 全部失败
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出异常
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 按 5 分钟间隔生成时间戳
# 特征配置
features = ['日龄','THI','目标温度','当前温度']  # 预定义候选特征列
target = 'THI'  # 指定预测目标
# 过滤特征列
features = [f for f in features if f in data.columns]  # 仅保留数据中存在的特征
print(f"使用特征: {features}")  # 打印最终特征
print(f"数据形状: {data.shape}")  # 打印数据规模
# 检查目标列是否存在
if target not in data.columns:  # 若缺少目标列
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带THI的CSV文件。")  # 抛出异常
# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印阶段标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前规模
print(f"\n各列缺失值统计:")  # 打印统计标题
for col in features + [target]:  # 遍历特征与目标列
    if col in data.columns:  # 仅统计存在的列
        nan_count = data[col].isna().sum()  # 缺失值个数
        nan_pct = nan_count / len(data) * 100  # 缺失占比
        if nan_count > 0:  # 仅打印含缺失的列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印缺失数与占比
# 定义连续变量和离散变量
continuous_vars = ['THI', '外界温度', 'CO2', '目标温度']  # 连续变量列表
discrete_vars = ['日龄']  # 离散变量列表
# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理含缺失的列
        print(f"\n处理 {col}:")  # 打印当前列名
        before_nan = data[col].isna().sum()  # 填充前缺失数
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 线性插值（双向）
        data[col] = data[col].fillna(method='ffill')  # 前向填充剩余缺失
        data[col] = data[col].fillna(method='bfill')  # 后向填充开头残留缺失
        after_nan = data[col].isna().sum()  # 填充后缺失数
        print(f"  插值前: {before_nan} 个缺失值")  # 打印填充前数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印填充后数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅处理含缺失的列
        print(f"\n处理 {col}:")  # 打印当前列名
        before_nan = data[col].isna().sum()  # 填充前缺失数
        data[col] = data[col].fillna(method='ffill')  # 前向填充
        data[col] = data[col].fillna(method='bfill')  # 后向填充
        after_nan = data[col].isna().sum()  # 填充后缺失数
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 残留缺失总数
if remaining_nan > 0:  # 若仍有缺失
    print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 提示删除行
    data = data.dropna(subset=features + [target])  # 删除含缺失的行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后规模
print(f"{'='*60}\n")  # 打印分隔线

# 重置索引
data = data.reset_index(drop=True)  # 重置索引使序号连续

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 打印加载提示

MODEL_PATH = 'Model_GRUWavelet_pytorch.pth'  # 定义检查点文件路径

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载检查点到设备（含配置与归一化器）
    print(f"成功加载模型: {MODEL_PATH}")  # 打印成功提示
except FileNotFoundError:  # 文件不存在
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛出异常

input_dim = checkpoint['input_dim']  # 读取输入维度
time_step = checkpoint['time_step']  # 读取时间窗口
feature_names = checkpoint.get('feature_names', [])  # 读取小波特征名列表（无则空列表）
original_features = checkpoint.get('original_features', features)  # 读取原始特征名列表
wavelet_type = checkpoint.get('wavelet', 'db4')  # 读取小波基函数类型
wavelet_level = checkpoint.get('level', 3)  # 读取分解层数
wavelet_window = checkpoint.get('wavelet_window', time_step)  # 读取小波窗口大小
wavelet_target_col = checkpoint.get('wavelet_target_col', 'THI')  # 获取小波分解的目标列
print(f"\n模型配置:")  # 打印模型配置标题
print(f"  输入维度: {input_dim}")  # 打印输入维度
print(f"  时间步长: {time_step}")  # 打印时间窗口
print(f"  小波基函数: {wavelet_type}")  # 打印小波基
print(f"  分解层数: {wavelet_level}")  # 打印分解层数
print(f"  小波窗口: {wavelet_window}")  # 打印小波窗口
print(f"  小波分解目标列: {wavelet_target_col}")  # 打印分解目标列

# 使用原始特征列表
features = original_features  # 统一使用模型保存的原始特征名与顺序

# 检查测试数据是否包含所需特征
missing_features = [f for f in features if f not in data.columns]  # 找出数据中缺失的必需特征
if missing_features:  # 若存在缺失
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出异常

print(f"\n使用原始特征: {features}")  # 打印最终使用的原始特征

# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
print("\n加载训练数据...")  # 打印训练数据加载提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表
train_data = None  # 初始化训练数据为空
for enc in encodings:  # 遍历编码
    try:
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 读取训练数据
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功编码
        break  # 跳出
    except (UnicodeDecodeError, UnicodeError):  # 解码失败
        continue  # 继续尝试
if train_data is None:  # 全部失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常

# 对训练数据和测试数据进行小波分解
# 检查模型是使用全部特征小波分解还是仅THI小波分解
use_partial_wavelet = checkpoint.get('wavelet_target_col') is not None  # 检查点含目标列则说明是新版（仅分解目标列）

if use_partial_wavelet:  # 新版：仅分解 THI
    # 新版本：仅对THI进行小波分解
    print("\n检测到新版本模型（仅对THI进行小波分解）")  # 打印版本提示
    print("\n对训练数据进行小波分解（仅分解THI）...")  # 打印训练分解提示
    train_wavelet_features = wavelet_decompose_sliding(train_data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 训练集特征小波分解
    
    print("\n对测试数据进行小波分解（仅分解THI）...")  # 打印测试分解提示
    test_wavelet_features = wavelet_decompose_sliding(data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 测试集特征小波分解
else:
    # 旧版本：对所有特征进行小波分解
    print("\n检测到旧版本模型（对所有特征进行小波分解）")  # 打印版本提示
    
    def wavelet_decompose_sliding_all(data, wavelet='db4', level=3, window_size=None):
        """对所有特征进行小波分解（旧版本兼容）"""
        print(f"\n开始滑动窗口小波分解（所有特征）...")  # 打印分解开始提示
        print(f"小波基函数: {wavelet}, 分解层数: {level}")  # 打印小波基与层数
        
        if window_size is None:  # 未指定窗口时
            window_size = min(2 ** (level + 6), len(data))  # 自适应计算窗口大小
        
        print(f"滑动窗口大小: {window_size}")  # 打印窗口大小
        
        wavelet_features = pd.DataFrame(index=data.index)  # 创建结果表
        
        for col in data.columns:  # 遍历每一列（全部特征）
            if col == 'timestamp':  # 跳过时间戳列
                continue  # 进入下一列
            
            signal = data[col].values  # 取当前列信号
            approx_list = []  # 收集近似系数
            detail_lists = [[] for _ in range(level)]  # 每层细节系数列表
            
            for i in range(len(signal)):  # 逐时间点滑动分解
                start_idx = max(0, i - window_size + 1)  # 窗口起点，避免越界
                window_signal = signal[start_idx:i+1]  # 截止当前的窗口信号
                
                if len(window_signal) < window_size:  # 开头窗口不足
                    pad_width = window_size - len(window_signal)  # 需补齐长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 边缘值补齐
                
                try:
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 离散小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数末值
                    for j, detail in enumerate(coeffs[1:]):  # 遍历细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取末值，空补 0
                except Exception as e:  # 分解失败
                    approx_list.append(window_signal[-1])  # 末值代替近似系数
                    for j in range(level):  # 遍历各层
                        detail_lists[j].append(0.0)  # 细节补 0
            
            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似特征列
            for j in range(level):  # 遍历各层
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入细节特征列
        
        print(f"小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印特征总数
        return wavelet_features  # 返回结果表
    
    print("\n对训练数据进行小波分解（所有特征）...")  # 打印训练分解提示
    train_wavelet_features = wavelet_decompose_sliding_all(train_data[features], wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 训练集全特征分解
    
    print("\n对测试数据进行小波分解（所有特征）...")  # 打印测试分解提示
    test_wavelet_features = wavelet_decompose_sliding_all(data[features], wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 测试集全特征分解

# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若检查点含归一化器
    print("\n使用模型保存的归一化器...")  # 打印提示
    feature_scaler = checkpoint['feature_scaler']  # 复用保存的特征归一化器
    target_scaler = checkpoint['target_scaler']  # 复用保存的目标归一化器
else:  # 否则重新拟合
    print("\n使用训练数据重新拟合归一化器...")  # 打印提示
    feature_scaler = MinMaxScaler()  # 新建特征归一化器
    target_scaler = MinMaxScaler()  # 新建目标归一化器
    feature_scaler.fit(train_wavelet_features)  # 用训练小波特征拟合
    target_scaler.fit(train_data[[target]])  # 用训练目标拟合

# 归一化测试数据
test_features = pd.DataFrame(  # 归一化测试小波特征并重建 DataFrame
    feature_scaler.transform(test_wavelet_features),  # 缩放测试特征
    columns=feature_names  # 使用模型保存的特征列名
)

print(f"  特征数量: {len(test_features.columns)}")  # 打印特征数量
print(f"  特征列表: {list(test_features.columns)}")  # 打印特征列表

# ===================== 序列构造 =====================
pred_offset = 6  # 预测偏移：用当前窗口预测 6 步之后的 THI

def create_sequences(data, time_step):
    """创建时间序列样本"""
    sequences = []  # 收集各滑动窗口样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动起点，保证不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取历史窗口
    return np.array(sequences)  # 堆叠为三维数组返回

print("\n创建序列数据...")  # 打印序列构造提示
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 缩放测试标签

# 创建序列
X_test = create_sequences(test_features, time_step)  # 构造测试输入序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 按偏移对齐目标并展平
y_test = y_test[:len(X_test)]  # 裁剪标签使其与样本数一致

print(f"\n测试数据形状:")  # 打印形状检查标题
print(f"  X_test: {X_test.shape}")  # 打印测试输入形状
print(f"  y_test: {y_test.shape}")  # 打印测试标签形状

# ===================== 构建模型 =====================
print("\n构建模型...")  # 打印模型构建提示
model = GRUWaveletModel(  # 实例化待加载权重的模型
    input_dim=input_dim,  # 输入维度来自检查点
    time_step=time_step,  # 时间步来自检查点
    gru_hidden=128,  # 隐藏层维度，与训练一致
    gru_layers=1,  # GRU 层数，与训练一致
    dropout=0.2  # dropout 比例，与训练一致
).to(device)  # 移至指定设备

model.load_state_dict(checkpoint['model_state_dict'])  # 加载训练好的权重
model.eval()  # 切换评估模式

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型参数总数

# ===================== 模型预测 =====================
print("\n开始预测...")  # 打印预测提示
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试序列转张量并移至设备

with torch.no_grad():  # 关闭梯度进行推理
    y_pred = model(X_test_tensor).cpu().numpy()  # 预测并取回 numpy 结果

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 真实标签反归一化为真实 THI
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 预测值反归一化为真实 THI

# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R²
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 计算均方根误差 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print("\n" + "=" * 60)  # 打印分隔线
print("GRU+小波模型测试结果")  # 打印评估标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R²
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线
# ===================== 结果可视化 =====================
plot_length = min(1600, len(y_test_rescaled))  # 限制绘图点数最多 1600
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的预测值

# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成 X 轴步骤索引

plt.figure(figsize=(15, 8))  # 创建画布
y_test_flat = y_test_plot.flatten()  # 展平真实值
y_pred_flat = y_pred_plot.flatten()  # 展平预测值
plt.tick_params(axis='both', labelsize=16)  # 设置刻度字号
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实曲线（蓝色）
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测曲线（橙色）
plt.xlabel('Timestep', fontsize=16)  # 设置 X 轴标签
plt.ylabel('THI', fontsize=16)  # 设置 Y 轴标签
# plt.title(f'GRU+Wavelet Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 显示网格线
plt.tight_layout()  # 调整布局
plt.text(0.5, 0.95, '(e)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', va='top', ha='center')  # 在图上添加子图编号 (e)
plt.tight_layout()  # 再次调整布局
output_filename = 'test_GRUWavelet_prediction.png'  # 定义输出文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 保存预测对比图（高分辨率）
print(f"\n预测结果图已保存: {output_filename}")  # 打印保存提示
plt.close()  # 关闭图形释放内存

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算全部预测误差（预测 - 真实）

# 仅用于绘图：取前300个点
max_points = min(300, len(errors))  # 限制绘图误差点数最多 300
y_test_plot_err = y_test_rescaled[:max_points].flatten()  # 取前 N 个真实值
y_pred_plot_err = y_pred_rescaled[:max_points].flatten()  # 取前 N 个预测值
errors_plot = y_pred_plot_err - y_test_plot_err  # 计算绘图用误差

print(f"\n误差分析图使用前 {max_points} 个数据点")  # 打印绘图点数

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
min_val = min(y_test_plot_err.min(), y_pred_plot_err.min())  # 取最小值作坐标下界
max_val = max(y_test_plot_err.max(), y_pred_plot_err.max())  # 取最大值作坐标上界
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')  # 绘制 y=x 参考线
plt.xlabel('True value', fontsize=11)  # 设置 X 轴标签
plt.ylabel('Predicted value', fontsize=11)  # 设置 Y 轴标签
plt.title('True value vs Predicted value', fontsize=12)  # 设置子图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.4)  # 显示网格线
margin = (max_val - min_val) * 0.05  # 计算坐标边距
plt.xlim(min_val - margin, max_val + margin)  # 设置 X 轴范围
plt.ylim(min_val - margin, max_val + margin)  # 设置 Y 轴范围

# 子图2: 误差分布直方图（前300个点）
plt.subplot(2, 2, 2)  # 选中第 2 个子图
plt.hist(errors_plot, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差直方图（50 箱）
plt.xlabel('Prediction Error', fontsize=11)  # 设置 X 轴标签
plt.ylabel('Frequency', fontsize=11)  # 设置 Y 轴标签
plt.title(f'Error Distribution (First {max_points} Points)', fontsize=12)  # 设置标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 0 误差处画参考虚线
plt.grid(True, alpha=0.3, axis='y')  # 显示 Y 方向网格线

# 子图3: 误差时序图（前300个点）
plt.subplot(2, 2, 3)  # 选中第 3 个子图
error_steps = np.arange(len(errors_plot))  # 生成误差时序 X 轴索引
plt.plot(error_steps, errors_plot, alpha=0.7, linewidth=1)  # 绘制误差随步骤变化曲线
plt.xlabel('Timestep', fontsize=11)  # 设置 X 轴标签
plt.ylabel('Prediction Error', fontsize=11)  # 设置 Y 轴标签
plt.title(f'Error Time Series (First {max_points} Points)', fontsize=12)  # 设置标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 0 误差处画参考虚线
plt.grid(True, alpha=0.3)  # 显示网格线

# 子图4: 误差箱线图（前300个点）
plt.subplot(2, 2, 4)  # 选中第 4 个子图
plt.boxplot([errors_plot], tick_labels=['GRU+Wavelet'])  # 绘制误差箱线图
plt.ylabel('Prediction Error', fontsize=11)  # 设置 Y 轴标签
plt.title(f'Error Statistics Distribution (First {max_points} Points)', fontsize=12)  # 设置标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 0 误差处画参考虚线
plt.grid(True, alpha=0.3, axis='y')  # 显示 Y 方向网格线

plt.tight_layout()  # 调整子图间距

error_filename = 'test_GRUWavelet_error_analysis.png'  # 定义误差分析图文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 保存误差分析图（高分辨率）
print(f"误差分析图已保存: {error_filename}")  # 打印保存提示
plt.close()  # 关闭图形释放内存

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("Error Statistics")  # 打印统计标题
print("=" * 60)  # 打印分隔线
print(f"Mean Error:     {np.mean(errors):.4f}")  # 打印误差均值
print(f"Std Deviation:  {np.std(errors):.4f}")  # 打印误差标准差
print(f"Max Error:      {np.max(errors):.4f}")  # 打印最大误差
print(f"Min Error:      {np.min(errors):.4f}")  # 打印最小误差
print(f"Median Error:   {np.median(errors):.4f}")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\nTest Completed!")  # 打印测试完成提示
