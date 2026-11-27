"""
GRU + 小波分解模型训练与评估脚本（温度偏差预测）。

以"温度偏差"为预测目标，读取温湿度时序 CSV 并完成缺失值处理与特征相关性分析；
使用滑动窗口小波分解（db4 基函数、3 层）提取温度偏差的近似/细节特征，避免未来信息泄漏；
在 8:2 划分的训练/测试集上构造时序样本，训练带早停机制的 GRU 模型；
随后在测试集与新增数据上评估 R2、RMSE、MAE，绘制损失曲线、预测对比图与相关性图，
最后将模型权重、结构参数与归一化器保存为 .pth 文件，供推理脚本加载使用。
"""

import pandas as pd  # 导入 pandas，用于 CSV 读取与 DataFrame 数据处理
import numpy as np  # 导入 numpy，用于数组运算与数值统计
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础模块，为设置后端做准备
matplotlib.use('Agg')  # 使用非交互式后端 Agg，直接保存图片而不弹出窗口
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制训练过程与预测结果图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、MSE、MAE 三个回归评估指标函数
from sklearn.model_selection import train_test_split, KFold  # 导入 train_test_split 与 KFold，用于数据集划分与交叉验证
import torch  # 导入 PyTorch 主包，用于张量计算与设备选择
import torch.nn as nn  # 导入神经网络模块，用于构建 GRU 与全连接层
import torch.optim as optim  # 导入优化器模块，使用 Adam 更新模型参数
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入 Dataset、DataLoader 与 TensorDataset，用于批量加载数据
import pywt  # 小波变换库，提供 wavedec 等离散小波分解函数
import seaborn as sns  # 用于绘制相关性热力图；文件开头已导入，此处重复导入保持原样
# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选用 CUDA（若可用）否则使用 CPU 作为计算设备
print(f"使用设备: {device}")  # 打印当前使用的计算设备
if torch.cuda.is_available():  # 若检测到可用的 GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示负号
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 候选编码列表，依次尝试直到成功读取文件
data = None  # 初始化数据变量为空
for enc in encodings:  # 遍历候选编码列表
    try:  # 尝试用当前编码读取
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 以当前编码读取训练数据 CSV
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获解码异常
        continue  # 尝试下一种编码

if data is None:  # 若所有编码均读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止脚本

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按起始时间与 5 分钟频率构造 timestamp 时间列
features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 定义候选特征列表
target = '温度偏差'  # 设定预测目标列为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列
# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印缺失值处理标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据形状
print(f"\n各列缺失值统计:")  # 打印缺失值统计标题
for col in features + [target]:  # 遍历特征与目标列
    if col in data.columns:  # 仅处理数据中存在的列
        nan_count = data[col].isna().sum()  # 统计该列缺失值数量
        nan_pct = nan_count / len(data) * 100  # 计算缺失值所占百分比
        if nan_count > 0:  # 仅当存在缺失值时打印
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失值数量与比例

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 定义需做线性插值的连续变量列表
discrete_vars = ['日龄']  # 定义需做前后向填充的离散变量列表

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印当前处理提示
        before_nan = data[col].isna().sum()  # 记录处理前的缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 对连续变量做线性插值
        data[col] = data[col].ffill()  # 前向填充开头的缺失值
        data[col] = data[col].bfill()  # 后向填充结尾的缺失值
        after_nan = data[col].isna().sum()  # 记录处理后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前的缺失值数量
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后的缺失值数量

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当列存在且有缺失值时处理
        print(f"\n处理 {col}:")  # 打印当前处理提示
        before_nan = data[col].isna().sum()  # 记录处理前的缺失值数量
        data[col] = data[col].ffill()  # 对离散变量做前向填充
        data[col] = data[col].bfill()  # 对离散变量做后向填充
        after_nan = data[col].isna().sum()  # 记录处理后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前的缺失值数量
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后的缺失值数量

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征与目标列剩余的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告信息
    data = data.dropna(subset=features + [target])  # 删除仍含缺失值的行

print(f"\n处理后数据形状: {data.shape}")  # 打印处理后的数据形状
print(f"{'='*60}\n")  # 打印分隔线

print(f"\n【GRU + 小波变换模型】")  # 打印当前模型标题
print(f"使用特征: {features}")  # 打印使用的特征列表
print(f"特征数量: {len(features)}")  # 打印特征数量

# ===================== 小波变换特征提取（仅对温度偏差） =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):  # 定义无数据泄漏的滑动窗口小波分解函数，仅分解目标列
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（温度偏差）进行小波分解，其他列保持原样
    每个时间点只使用历史数据
    
    参数:
        data: 输入数据 (DataFrame)
        target_col: 需要进行小波分解的列名（默认'温度偏差'）
        wavelet: 小波基函数 ('db4', 'sym5', 'coif3' 等)
        level: 分解层数
        window_size: 滑动窗口大小，None则自动计算
    返回:
        分解后的特征 (DataFrame)
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印开始小波分解的提示
    print(f"小波基函数: {wavelet}")  # 打印所使用的小波基函数
    print(f"分解层数: {level}")  # 打印分解层数
    
    # 自动计算窗口大小：确保足够大以进行有效的小波分解
    if window_size is None:  # 当未指定窗口大小时
        window_size = min(2 ** (level + 6), len(data))  # 根据分解层数自适应：取 2 的 (level+6) 次方与数据长度的较小值
    
    print(f"滑动窗口大小: {window_size}")  # 打印滑动窗口大小
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空结果表
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 跳过时间列
            continue  # 不参与分解
        
        signal = data[col].values  # 取出当前列的 numpy 数组信号
        
        # 只对目标列进行小波分解
        if col == target_col:  # 仅对目标列执行小波分解
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 初始化近似系数序列列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数各建一个空列表
            
            for i in range(len(signal)):  # 逐时间点滑动窗口
                # 只使用当前时间点之前的数据（包括当前点）
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点，只使用当前及历史数据
                window_signal = signal[start_idx:i+1]  # 截取从起点到当前点的窗口信号
                
                # 如果窗口太小，用边界值填充到window_size
                if len(window_signal) < window_size:  # 当窗口长度不足时
                    pad_width = window_size - len(window_signal)  # 计算需要填充的长度
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边界值在前端填充至指定窗口长度
                
                # 小波分解
                try:  # 尝试对窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 wavedec 做 level 层离散小波分解
                    
                    # 取最后一个值作为当前时间点的特征
                    approx_list.append(coeffs[0][-1])  # 取近似系数末值作为当前点近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取该层细节系数末值，空则记为 0
                except Exception as e:  # 捕获分解异常（异常对象 e 未使用，与原文件一致）
                    # 如果分解失败，使用原始值
                    approx_list.append(window_signal[-1])  # 用窗口末值作为近似特征回退
                    for j in range(level):  # 遍历各层细节特征
                        detail_lists[j].append(0.0)  # 将细节特征回退为 0.0
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 将近似系数序列写入结果表
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 将每层细节系数写入对应特征列
        else:  # 对于其他列
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列保持原样的提示
            wavelet_features[col] = signal  # 将该列原样保留到结果表
    
    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印分解完成提示与生成特征数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印小波分解特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印其他原始特征数量
    return wavelet_features  # 返回小波分解后的特征表

# ===================== 数据集划分 =====================
time_step = 36  # GRU 时间窗口为 36 步（180 分钟；原注释写 48 步，保持原样）
pred_offset = 6  # 预测偏移为 6 步，即提前 30 分钟预测

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印 GRU 时间窗口的步数与分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移的步数与分钟数

# 划分数据集
split_index = int(len(data) * 0.8)  # 按 8 比 2 计算训练集与测试集的划分下标
train_data_raw = data.iloc[:split_index].copy()  # 取前 80% 数据作为训练集
test_data_raw = data.iloc[split_index:].copy()  # 取后 20% 数据作为测试集

print(f"\n数据集划分:")  # 打印数据集划分标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw)/len(data)*100:.1f}%)")  # 打印训练集样本数及占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw)/len(data)*100:.1f}%)")  # 打印测试集样本数及占比

# ===================== 小波分解 =====================
# 使用与GRU相同的时间窗口
wavelet_window = time_step  # 让小波窗口与 GRU 时间窗口保持一致
print(f"\n小波窗口大小设置为: {wavelet_window} 步 (与GRU时间窗口一致)")  # 打印小波窗口大小设置，与 GRU 时间窗口一致

print("\n对训练集进行小波分解（仅分解温度偏差）...")  # 提示即将对训练集进行小波分解
train_wavelet_features = wavelet_decompose_sliding(train_data_raw[features], target_col='温度偏差', wavelet='db4', level=3, window_size=wavelet_window)  # 对训练集特征执行滑动窗口小波分解

print("\n对测试集进行小波分解（仅分解温度偏差）...")  # 提示对测试集进行小波分解
test_wavelet_features = wavelet_decompose_sliding(test_data_raw[features], target_col='温度偏差', wavelet='db4', level=3, window_size=wavelet_window)  # 对测试集特征执行滑动窗口小波分解

# 直接使用小波分解后的特征（已包含原始特征和小波特征）
train_features = train_wavelet_features  # 训练特征直接采用小波分解结果（已含原始特征）
test_features = test_wavelet_features  # 测试特征直接采用小波分解结果（已含原始特征）
feature_names = list(train_wavelet_features.columns)  # 保存训练特征的所有列名

print(f"\n使用模式: 温度偏差小波分解 + 其他原始特征")  # 打印当前使用的特征模式
print(f"总特征数: {len(feature_names)}")  # 打印总特征数
print(f"  - 温度偏差的小波特征: 4 个（1个近似系数 + 3个细节系数）")  # 打印温度偏差小波特征数量说明
print(f"  - 其他原始特征: {len(features) - 1} 个")  # 打印其他原始特征数量说明
print(f"特征列表: {feature_names}")  # 打印完整特征列表

# ===================== 相关性分析 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("相关性分析：小波分解特征 + 原始变量")  # 打印相关性分析标题
print(f"{'='*60}")  # 打印分隔线

# 合并训练集的小波特征和目标变量
correlation_data = train_wavelet_features.copy()  # 复制训练集小波特征用于相关性分析
correlation_data['温度偏差_目标'] = train_data_raw[target].values  # 向相关性分析数据中追加真实目标列

# 计算相关性矩阵
correlation_matrix = correlation_data.corr()  # 计算各特征与目标的相关性矩阵

# 提取与目标变量的相关性
target_correlation = correlation_matrix['温度偏差_目标'].drop('温度偏差_目标').sort_values(ascending=False)  # 提取与目标变量的相关性，丢弃自身并降序排列

print("\n各特征与目标变量（温度偏差）的相关性：")  # 打印各特征与目标相关性的提示
print("-" * 60)  # 打印分隔线
for feature, corr in target_correlation.items():  # 遍历每个特征及其相关系数
    print(f"{feature:30s}: {corr:7.4f}")  # 打印特征名与相关系数

# 绘制相关性热力图
import seaborn as sns  # 再次导入 seaborn（原文件如此，保持原样）

plt.figure(figsize=(16, 14))  # 创建相关性热力图画布
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',   # 用 seaborn 绘制相关性热力图
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('特征相关性热力图（小波分解特征 + 原始变量）', fontsize=16, pad=20)  # 设置热力图标题
plt.xticks(rotation=45, ha='right')  # 旋转横轴标签以避免重叠
plt.yticks(rotation=0)  # 保持纵轴标签水平
plt.tight_layout()  # 调整布局使之紧凑
plt.savefig('correlation_heatmap_wavelet.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存相关性热力图
print("\n相关性热力图已保存: correlation_heatmap_wavelet.png")  # 打印热力图保存路径

# 绘制与目标变量的相关性条形图
plt.figure(figsize=(12, 8))  # 创建与目标相关性条形图画布
colors = ['green' if x > 0 else 'red' for x in target_correlation.values]  # 按相关系数的正负设定条形颜色
plt.barh(range(len(target_correlation)), target_correlation.values, color=colors, alpha=0.7)  # 绘制水平条形图展示各特征与温度偏差的相关性
plt.yticks(range(len(target_correlation)), target_correlation.index)  # 设置条形图纵轴标签为特征名
plt.xlabel('相关系数', fontsize=12)  # 设置横轴标签
plt.title('各特征与温度偏差的相关性', fontsize=14, pad=15)  # 设置条形图标题
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)  # 绘制相关系数为 0 的参考线
plt.grid(True, alpha=0.3, axis='x')  # 仅为横轴添加网格
plt.tight_layout()  # 调整布局使之紧凑
plt.savefig('correlation_barplot_target.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存相关性条形图
print("目标变量相关性条形图已保存: correlation_barplot_target.png")  # 打印条形图保存路径

print(f"{'='*60}\n")  # 打印分隔线

# ===================== 归一化 =====================
print("\n归一化处理...")  # 提示开始归一化处理
feature_scaler = MinMaxScaler()  # 创建特征归一化器
target_scaler = MinMaxScaler()  # 创建目标归一化器

# 只在训练集上fit
feature_scaler.fit(train_features)  # 只在训练特征上拟合归一化器，避免数据泄漏
target_scaler.fit(train_data_raw[[target]])  # 只在训练目标上拟合归一化器

# 分别transform训练集和测试集
train_features = pd.DataFrame(  # 对训练特征归一化并重建为 DataFrame
    feature_scaler.transform(train_features),
    columns=feature_names
)
test_features = pd.DataFrame(  # 对测试特征归一化并重建为 DataFrame
    feature_scaler.transform(test_features),
    columns=feature_names
)

train_target = target_scaler.transform(train_data_raw[[target]]).flatten()  # 归一化训练目标并展平为一维数组
test_target = target_scaler.transform(test_data_raw[[target]]).flatten()  # 归一化测试目标并展平为一维数组

# ===================== 序列数据构造 =====================
def create_sequences(data, time_step):  # 定义序列构造函数，按时间步与预测偏移切分样本
    """
    将二维特征表滑窗切分为三维序列样本 (样本数, time_step, 特征数)。
    参数: data 为归一化后的特征表；time_step 为输入窗口长度。
    返回: numpy 三维数组形式的序列样本。
    """
    sequences = []  # 初始化序列样本列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点范围，尾部留出预测偏移
        sequences.append(data.iloc[i:i + time_step].values)  # 取窗口内数据作为一个样本
    return np.array(sequences)  # 将样本列表转为 numpy 数组返回

print("\n创建训练集序列...")  # 提示开始创建训练集序列
X_train = create_sequences(train_features, time_step)  # 由归一化训练特征构造时序样本
y_train = train_target[time_step + pred_offset:]  # 按预测偏移对齐训练目标
y_train = y_train[:len(X_train)]  # 截断训练目标使其与样本数一致

print("\n创建测试集序列...")  # 提示开始创建测试集序列
X_test = create_sequences(test_features, time_step)  # 由归一化测试特征构造时序样本
y_test = test_target[time_step + pred_offset:]  # 按预测偏移对齐测试目标
y_test = y_test[:len(X_test)]  # 截断测试目标使其与样本数一致

print(f"\n序列数据形状:")  # 打印序列数据形状的标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集序列与目标形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集序列与目标形状

# ===================== GRU+小波模型定义 =====================
class GRUWaveletModel(nn.Module):  # 定义 GRU + 小波变换预测模型类
    """
    GRU + 小波变换模型（仅小波特征）
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)
    由 GRU 编码各时间步，并取最后时刻的隐藏状态经全连接层给出单点预测。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3):  # 初始化模型结构参数
        """
        构建单向 GRU 编码层与输出全连接层。
        参数: input_dim 输入特征数；time_step 时间步长；gru_hidden 隐藏维度；
              gru_layers GRU 层数；dropout 丢弃率（仅多层时生效）。
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类初始化
        self.time_step = time_step  # 保存时间步长
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏维度
        
        # GRU层
        self.gru = nn.GRU(  # 构建单向 GRU 编码层，batch_first 表示批次维在前
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        
        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 定义输出全连接层，将隐藏状态映射为单个预测值

    def forward(self, x):  # 前向传播：GRU 编码后取末时刻输出做单点预测
        """
        前向传播：GRU 编码输入序列并取末时刻输出做预测。
        参数: x 形状为 (batch, time_step, input_dim)；返回形状为 (batch, 1)。
        """
        # x: (batch, time_step, input_dim)
        # GRU编码
        out, h = self.gru(x)  # GRU 编码，out 形状为 (batch, time_step, hidden)
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # 取最后时刻输出，形状为 (batch, hidden)
        
        # 预测输出
        return self.fc(out_last)  # 全连接层输出预测，形状为 (batch, 1)


class EarlyStopping:  # 定义早停策略类，验证损失长期不降时终止训练
    """
    早停策略：记录最佳验证损失与模型权重，
    当验证损失连续 patience 轮未改善超过 min_delta 时触发停止。
    """
    def __init__(self, patience=10, min_delta=1e-4):  # 初始化早停参数
        """
        初始化早停器。
        参数: patience 容忍无改善的轮数；min_delta 认定改善所需的最小损失下降量。
        """
        self.patience = patience  # 记录容忍无改善的轮数
        self.min_delta = min_delta  # 记录认定改善所需的最小损失下降量
        self.counter = 0  # 无改善计数器清零
        self.best_loss = None  # 最佳损失初始为空
        self.early_stop = False  # 早停标志初始为假
        self.best_model = None  # 最佳模型权重初始为空

    def __call__(self, val_loss, model):  # 每轮验证后调用，更新最佳权重或累计无改善次数
        """
        根据本次验证损失更新最佳模型与停止状态。
        参数: val_loss 本轮验证损失；model 当前模型（用于保存最佳权重副本）。
        """
        if self.best_loss is None:  # 首次调用时
            self.best_loss = val_loss  # 将当前验证损失记为最佳损失
            self.best_model = model.state_dict().copy()  # 保存当前模型权重副本
        elif val_loss > self.best_loss - self.min_delta:  # 若验证损失未比最佳损失改善超过 min_delta
            self.counter += 1  # 无改善计数加一
            if self.counter >= self.patience:  # 无改善轮数达到耐心阈值时
                self.early_stop = True  # 置位早停标志
        else:  # 若验证损失显著改善
            self.best_loss = val_loss  # 更新最佳损失
            self.best_model = model.state_dict().copy()  # 更新最佳模型权重副本
            self.counter = 0  # 重置无改善计数器


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 定义训练函数，使用早停并返回训练与验证损失曲线
    """
    在给定加载器上训练模型，使用 MSE 损失与 Adam 优化器，并配合早停机制。
    参数: model 待训练模型；train_loader/val_loader 训练与验证数据加载器；
          epochs 最大轮数；lr 学习率；patience 早停耐心值。
    返回: (train_losses, val_losses) 每轮平均训练与验证损失列表。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新全部可训练参数
    early_stopping = EarlyStopping(patience=patience)  # 创建早停器实例

    train_losses = []  # 初始化训练损失记录列表
    val_losses = []  # 初始化验证损失记录列表

    for epoch in range(epochs):  # 遍历训练轮次
        # 训练阶段
        model.train()  # 切换为训练模式，启用 dropout 等训练专属行为
        train_loss = 0.0  # 累计本轮训练损失
        for batch_x, batch_y in train_loader:  # 按批次读取训练数据
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将训练批次数据搬运到计算设备

            optimizer.zero_grad()  # 清空上一轮累积的梯度
            outputs = model(batch_x)  # 前向计算得到预测输出
            loss = criterion(outputs, batch_y)  # 计算预测值与真值的均方误差损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新模型参数

            train_loss += loss.item()  # 累加当前批次的损失值

        train_loss /= len(train_loader)  # 求本轮平均训练损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        # 验证阶段
        model.eval()  # 切换为评估模式，关闭 dropout
        val_loss = 0.0  # 累计本轮验证损失
        with torch.no_grad():  # 关闭梯度计算以节省显存与算力
            for batch_x, batch_y in val_loader:  # 按批次读取验证数据
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将验证批次数据搬运到计算设备
                outputs = model(batch_x)  # 前向计算得到验证预测
                loss = criterion(outputs, batch_y)  # 计算验证批次的损失
                val_loss += loss.item()  # 累加当前批次的验证损失

        val_loss /= len(val_loader)  # 求本轮平均验证损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        if (epoch + 1) % 10 == 0:  # 每隔 10 轮打印一次训练进度
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前轮次的训练损失与验证损失

        # 早停检查
        early_stopping(val_loss, model)  # 调用早停器更新最佳模型与停止状态
        if early_stopping.early_stop:  # 若触发早停
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停触发的轮次
            model.load_state_dict(early_stopping.best_model)  # 载入训练过程中保存的最佳权重
            break  # 跳出训练轮次循环

    # 加载最佳模型
    if early_stopping.best_model is not None:  # 训练结束后若曾保存最佳权重
        model.load_state_dict(early_stopping.best_model)  # 确保模型最终保留最佳状态

    return train_losses, val_losses  # 返回训练与验证损失曲线


# # ===================== K折交叉验证训练 =====================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# val_losses_all = []
# history_list = []
input_dim = len(feature_names)  # 使用小波特征数  # 输入维度等于小波特征数（原注释保留并补充）
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
#     model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)
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
# plt.savefig('kfold_training_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')
# print("\nK折交叉验证损失曲线已保存: kfold_training_GRUWavelet_pytorch.png")

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印正在训练最终模型的标题
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练集样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试集样本数
X_train_tensor = torch.FloatTensor(X_train)  # 将全量训练特征转换为浮点张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 将全量训练目标转换为张量并增加特征维度
X_test_tensor = torch.FloatTensor(X_test)  # 将测试特征转换为浮点张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 将测试目标转换为张量并增加特征维度

# 从训练集中划出20%作为验证集（用于早停，避免使用测试集）
train_val_split = int(len(X_train) * 0.8)  # 从训练集中划出 20% 作为验证集，用于早停而不使用测试集
X_train_split = X_train_tensor[:train_val_split]  # 取划分后的训练特征
y_train_split = y_train_tensor[:train_val_split]  # 取划分后的训练目标
X_val_split = X_train_tensor[train_val_split:]  # 取划分后的验证特征
y_val_split = y_train_tensor[train_val_split:]  # 取划分后的验证目标

train_dataset = TensorDataset(X_train_split, y_train_split)  # 封装训练集为 TensorDataset
val_dataset = TensorDataset(X_val_split, y_val_split)  # 封装验证集为 TensorDataset
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 封装测试集为 TensorDataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构建训练数据加载器并打乱
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 构建验证数据加载器不打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 构建测试数据加载器不打乱
final_model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.3).to(device)  # 构建最终 GRU+小波模型并部署到设备
print(final_model)  # 打印模型结构
print("\n开始训练最终模型...")  # 提示开始训练最终模型
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印模型参数量
print(f"训练集样本数: {len(X_train_split)}")  # 打印训练集样本数
print(f"验证集样本数: {len(X_val_split)}")  # 打印验证集样本数

train_losses, val_losses = train_model(final_model, train_loader, val_loader,  # 在划分出的训练集上训练最终模型，用验证集早停
                                       epochs=200, lr=0.001, patience=10)

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练完成提示与最小验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12×6 的损失曲线画布
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses, label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置损失曲线标题
plt.xlabel('训练轮数 (Epoch)')  # 设置横轴标签（训练轮数）
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签（损失值 MSE）
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.tight_layout()  # 调整布局使之紧凑
plt.savefig('training_loss_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存训练损失曲线
print("训练损失曲线已保存: training_loss_GRUWavelet_pytorch.png")  # 打印损失曲线保存路径

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始测试集预测

y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将归一化测试目标反归一化还原为原始量纲

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 将测试特征转为张量并部署到设备

final_model.eval()  # 切换最终模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 前向预测并将结果取回为 numpy 数组

y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将归一化预测值反归一化还原为原始量纲

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse_test = np.sqrt(mse_test)  # 由 MSE 开方得到 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果（GRU+小波）")  # 打印测试集评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印测试集 R2
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印测试集 MAE
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建 15×5 的测试集预测对比画布
y_test_rescaled_flat = y_test_rescaled.flatten()  # 将反归一化测试目标展平
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 将反归一化预测值展平

test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset:]  # 取出与预测结果对应的测试集时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳使其与预测长度一致

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制真实温度偏差曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+小波预测', linewidth=2)  # 绘制模型预测温度偏差曲线
plt.xlabel('时间')  # 设置横轴标签（时间）
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签（温度偏差，摄氏度）
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置图标题并显示 R2 与 RMSE
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.savefig('test_prediction_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存测试集预测图
print("测试集预测结果已保存: test_prediction_GRUWavelet_pytorch.png")  # 打印测试集预测图保存路径

# ===================== 模型保存 =====================
torch.save({  # 开始构造要保存的模型检查点字典
    'model_state_dict': final_model.state_dict(),  # 保存模型权重状态字典
    'input_dim': input_dim,  # 保存输入特征维度
    'time_step': time_step,  # 保存时间步长
    'feature_names': feature_names,  # 保存全部特征名列表
    'original_features': features,  # 保存原始特征列表
    'wavelet': 'db4',  # 保存所使用的小波基函数名称
    'level': 3,  # 保存小波分解层数
    'wavelet_window': wavelet_window,  # 保存小波窗口大小
    'wavelet_target_col': '温度偏差',  # 保存小波分解的目标列，供推理脚本使用
    'use_wavelet_only': False,  # 标志是否仅用小波特征；False 表示同时使用小波特征与其他原始特征
    'feature_scaler': feature_scaler,  # 保存特征归一化器对象
    'target_scaler': target_scaler,  # 保存目标归一化器对象
}, 'Model_GRUWavelet_pytorch.pth')  # 将检查点保存到 .pth 文件
print("模型已保存: Model_GRUWavelet_pytorch.pth")  # 打印模型保存路径

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 提示开始处理新数据
new_data = pd.read_csv('部分温湿度数据.csv')  # 读取新采集的温湿度数据 CSV
new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',  # 按索引乘以 5 分钟并给定起始时间构造新数据时间列
                                       origin=pd.Timestamp('2025/6/21 0:04'))

# 对新数据进行小波分解（仅分解温度偏差）
print("\n对新数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 提示对新数据进行滑动窗口小波分解
new_wavelet_features = wavelet_decompose_sliding(new_data[features], target_col='温度偏差', wavelet='db4', level=3, window_size=wavelet_window)  # 对新数据特征执行滑动窗口小波分解（仅分解目标列）

# 归一化
new_features = pd.DataFrame(  # 对新数据小波特征归一化并重建为 DataFrame
    feature_scaler.transform(new_wavelet_features),
    columns=feature_names
)

# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据的原始温度偏差目标值

X_new = create_sequences(new_features, time_step)  # 由归一化新特征构造时序样本
y_new = y_new_original[time_step + pred_offset:]  # 按预测偏移对齐新数据目标值

X_new = X_new[:len(y_new)]  # 截断新数据序列使其与目标长度一致

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 将新数据特征转为张量并部署到设备

final_model.eval()  # 切换最终模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据前向预测并将结果取回为 numpy 数组

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 将新数据预测值反归一化还原量纲
y_new_rescaled = y_new.reshape(-1, 1)  # 将新数据真实目标 reshape 为二维数组

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据决定系数 R2
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据均方误差 MSE
rmse_new = np.sqrt(mse_new)  # 由 MSE 开方得到新数据 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果（GRU+小波）")  # 打印新数据评估标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 打印新数据 R2
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线
timestamps = new_data['timestamp'][time_step + pred_offset:]  # 取出新数据中与预测结果对应的时间戳
plt.figure(figsize=(15, 5))  # 创建 15×5 的新数据预测对比画布
y_new_rescaled_flat = y_new_rescaled.flatten()  # 将新数据真实目标展平
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 将新数据预测值展平
plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat, label='真实温度偏差', linewidth=2)  # 绘制新数据真实温度偏差曲线
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat, label='GRU+小波预测', linewidth=2)  # 绘制新数据模型预测曲线
plt.xlabel('时间')  # 设置横轴标签（时间）
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签（温度偏差，摄氏度）
plt.title(f'新数据集温度偏差结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置图标题并显示 R2 与 RMSE
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.savefig('new_data_prediction_GRUWavelet_pytorch.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存新数据预测图
print("新数据集预测结果已保存: new_data_prediction_GRUWavelet_pytorch.png")  # 打印新数据预测图保存路径
print("\n========== 所有任务完成（GRU+小波模型） ==========")  # 打印所有任务完成提示
