"""
GRU+小波变换模型测试脚本
用于加载训练好的GRU+小波模型并进行预测

在原有说明基础上补充：
本脚本负责加载已训练并保存的 GRU + 小波变换模型（Model_GRUWavelet_pytorch.pth），
对测试集数据执行与训练阶段一致的小波分解和归一化，构造时间序列样本后进行预测，
输出 R2/MSE/RMSE/MAE 等评价指标，并绘制预测曲线与误差分析图。
脚本同时兼容“仅对温度偏差做小波分解”的新版模型与“对所有特征做小波分解”的旧版模型。
"""
import pandas as pd  # 导入 pandas，用于读取 CSV 并组织特征表
import numpy as np  # 导入 numpy，用于数组变换与统计计算
import matplotlib  # 导入 matplotlib 基础模块以配置绘图后端
matplotlib.use('Agg')  # 设置 Agg 后端，支持无界面环境保存图片
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制预测与误差图表
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征与目标值归一化
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、均方误差、平均绝对误差三个评价指标
import torch  # 导入 PyTorch 主模块，用于张量与设备操作
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU 模型
import pywt  # 小波变换库  # 补充：提供 wavedec 等离散小波分解函数

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有 GPU 则使用 CUDA，否则回退到 CPU
print(f"使用设备: {device}")  # 打印当前使用的计算设备

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，保证图表中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，避免坐标轴负号乱码


# ===================== GRU+小波模型定义 =====================
class GRUWaveletModel(nn.Module):  # 定义 GRU 与小波结合的模型类
    """
    GRU + 小波变换模型（仅小波特征）
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)

    参数：
        input_dim: 输入特征维度。
        time_step: 序列时间步长，保存为实例属性供参考。
        gru_hidden: GRU 隐藏单元数，默认 128。
        gru_layers: GRU 堆叠层数，默认 1。
        dropout: 多层时的丢弃比例，默认 0.1。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 初始化 GRU 层与全连接输出层
        """初始化 GRU 层、全连接输出层，并保存模型的基本配置参数。"""
        super(GRUWaveletModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间步长配置到实例属性
        self.input_dim = input_dim  # 保存输入特征维度到实例属性
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏单元数到实例属性
        
        # GRU层
        self.gru = nn.GRU(  # 定义单向 GRU 层，具体参数见下面数行
            input_size=input_dim,  # GRU 输入维度
            hidden_size=gru_hidden,  # GRU 隐藏层单元数
            num_layers=gru_layers,  # GRU 堆叠层数
            dropout=dropout if gru_layers > 1 else 0.0,  # 仅当多层时启用 dropout，单层时置 0
            batch_first=True,  # 输入张量按 batch 优先排列
            bidirectional=False  # 使用单向 GRU（非双向）
        )  # GRU 层定义结束
        
        # 输出层
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层把隐藏状态映射为单一预测值

    def forward(self, x):  # 定义前向传播方法
        """前向传播：取序列最后一个时间步的隐藏状态，输出形状 (batch, 1) 的预测值。"""
        # x: (batch, time_step, input_dim)
        # GRU编码
        out, h = self.gru(x)  # out: (batch, time_step, hidden)  # 将序列送入 GRU，得到各时间步输出；h 为最终隐状态
        
        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)  # 取最后一个时间步的输出作为整段序列的表征
        
        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 经全连接层得到形状 (batch, 1) 的预测值


# ===================== 小波变换函数（仅对温度偏差） =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):  # 定义滑动窗口小波分解函数
    """
    使用滑动窗口进行小波分解，避免未来信息泄漏
    仅对指定列（温度偏差）进行小波分解，其他列保持原样

    参数：
        data: 待分解的特征 DataFrame。
        target_col: 需要分解的目标列名。
        wavelet: 使用的小波基函数名称。
        level: 小波分解层数。
        window_size: 滑动窗口长度；为 None 时按 2**(level+6) 与样本数取较小值。
    返回：
        DataFrame，目标列展开成近似与细节特征，其他列原样保留。
    """
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印分解开始提示及当前目标列
    print(f"小波基函数: {wavelet}, 分解层数: {level}")  # 打印所用小波基函数与分解层数
    
    if window_size is None:  # 窗口大小未显式指定时进入自动推算
        window_size = min(2 ** (level + 6), len(data))  # 取 2 的 (level+6) 次方与样本数的较小值作为窗口长度
    
    print(f"滑动窗口大小: {window_size}")  # 打印最终采用的滑动窗口大小
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空表，用于存放小波分解特征
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 跳过时间戳列，不参与分解
            continue  # 继续处理下一列
        
        signal = data[col].values  # 取出当前列的值作为一维时间序列信号
        
        # 只对目标列进行小波分解
        if col == target_col:  # 仅目标列执行小波分解
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在分解的列名
            approx_list = []  # 收集每个时间点的近似系数（低频分量）
            detail_lists = [[] for _ in range(level)]  # 为每个分解层各准备一个细节系数列表
            
            for i in range(len(signal)):  # 逐个时间点向前滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点并保证不越界
                window_signal = signal[start_idx:i+1]  # 截取以当前点为结尾的窗口信号
                
                if len(window_signal) < window_size:  # 序列开头处窗口长度不足时先补齐
                    pad_width = window_size - len(window_signal)  # 计算左侧需要补齐的元素个数
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值在左侧补齐至窗口长度
                
                try:  # 进入 try 块，捕获小波分解过程中的异常
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层离散小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数的末值作为当前点低频特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取该层细节系数末值，若为空则补 0
                except Exception as e:  # 捕获分解异常
                    approx_list.append(window_signal[-1])  # 分解失败时退化为取窗口最后一个原始值
                    for j in range(level):  # 遍历所有分解层
                        detail_lists[j].append(0.0)  # 分解失败时该层细节特征统一补 0
            
            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 将近似系数序列写入特征列
            for j in range(level):  # 遍历所有分解层
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 将各层细节系数写入对应特征列
        else:  # 非目标列走此分支
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列保持原样、不做分解的提示
            wavelet_features[col] = signal  # 非目标列原样复制到结果表
    
    print(f"小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印分解完成提示与生成特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 说明来自目标列的小波特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 说明其余原始特征的个数
    return wavelet_features  # 返回组装好的小波特征表


# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印加载测试数据提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 按优先级排列的候选文件编码列表
data = None  # 初始化数据变量为空，用于记录读取是否成功
for enc in encodings:  # 遍历候选编码逐一尝试
    try:  # 进入 try 块捕获解码错误
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试集 CSV 文件
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 读取成功则跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码相关异常
        continue  # 当前编码失败则尝试下一种

if data is None:  # 全部编码失败时 data 仍为 None
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出 ValueError 提示测试数据读取失败
data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 用行索引乘 5 分钟构造时间戳，起点为 2025/6/21 0:04
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 时间戳构造参数：单位为分钟并指定起始时刻

# 特征配置
features = ['日龄','温度偏差','外界温度','CO2','目标温度']  # 候选输入特征名称列表
target = '温度偏差'  # 指定预测目标列为温度偏差

# 过滤特征列
features = [f for f in features if f in data.columns]  # 仅保留测试数据中实际存在的特征列

print(f"使用特征: {features}")  # 打印实际使用的特征列表
print(f"数据形状: {data.shape}")  # 打印测试数据的行列数

# 检查目标列是否存在
if target not in data.columns:  # 检查目标列是否存在于测试数据中
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'。请确保使用带温度偏差的CSV文件。")  # 缺少目标列时抛出 KeyError 并给出明确提示

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印分隔线
print("缺失值处理")  # 打印缺失值处理标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印处理前的数据形状
print(f"\n各列缺失值统计:")  # 提示即将统计各列缺失值
for col in features + [target]:  # 遍历特征列与目标列
    if col in data.columns:  # 仅统计数据集中实际存在的列
        nan_count = data[col].isna().sum()  # 统计当前列缺失值数量
        nan_pct = nan_count / len(data) * 100  # 计算该列缺失值占比（百分比）
        if nan_count > 0:  # 仅对确有缺失的列输出统计信息
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失数量与百分比

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 连续型变量列表，采用插值方式填补
discrete_vars = ['日龄']  # 离散型变量列表，采用前后向填充

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历每个连续型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录插值前的缺失值数量
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 用线性插值填补内部缺失，limit_direction=both 兼顾两端
        data[col] = data[col].fillna(method='ffill')  # 再用前值填充开头残留的缺失
        data[col] = data[col].fillna(method='bfill')  # 最后用后值填充结尾残留的缺失
        after_nan = data[col].isna().sum()  # 记录插值后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前缺失数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后缺失数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历每个离散型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且有缺失值才处理
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量
        data[col] = data[col].fillna(method='ffill')  # 用前值向后填充缺失
        data[col] = data[col].fillna(method='bfill')  # 再用后值反向填充开头缺失
        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前缺失数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后缺失数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征列与目标列中剩余缺失值总数
if remaining_nan > 0:  # 若仍有缺失则进入删除行分支
    print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 警告仍有缺失值并提示将删除相关行
    data = data.dropna(subset=features + [target])  # 删除特征或目标列中含缺失的行
print(f"\n处理后数据形状: {data.shape}")  # 打印处理后的数据形状
print(f"{'='*60}\n")  # 打印分隔线并换行
# 重置索引
data = data.reset_index(drop=True)  # 重置索引，保持索引连续从 0 开始
# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 打印加载模型的提示
MODEL_PATH = 'Model_GRUWavelet_pytorch.pth'  # 定义已训练模型权重文件路径
try:  # 进入 try 块捕获模型文件读取异常
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载模型检查点并映射到指定设备
    print(f"成功加载模型: {MODEL_PATH}")  # 打印模型加载成功提示
except FileNotFoundError:  # 捕获文件不存在的异常
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 抛出信息更明确的 FileNotFoundError

input_dim = checkpoint['input_dim']  # 从检查点中读取输入特征维度
time_step = checkpoint['time_step']  # 从检查点中读取时间步长
feature_names = checkpoint.get('feature_names', [])  # 读取特征名列表，若未保存则使用空列表
original_features = checkpoint.get('original_features', features)  # 读取原始特征列表，若未保存则回退到当前 features
wavelet_type = checkpoint.get('wavelet', 'db4')  # 读取小波基函数名称，默认 db4
wavelet_level = checkpoint.get('level', 3)  # 读取小波分解层数，默认 3
wavelet_window = checkpoint.get('wavelet_window', 36)  # 读取小波滑动窗口长度，默认 36
wavelet_target_col = checkpoint.get('wavelet_target_col', '温度偏差')  # 获取小波分解的目标列  # 读取小波分解目标列，默认温度偏差
print(f"\n模型配置:")  # 打印模型配置标题
print(f"  输入维度: {input_dim}")  # 打印输入维度
print(f"  时间步长: {time_step}")  # 打印时间步长
print(f"  小波基函数: {wavelet_type}")  # 打印小波基函数
print(f"  分解层数: {wavelet_level}")  # 打印分解层数
print(f"  小波窗口: {wavelet_window}")  # 打印小波窗口长度
print(f"  小波分解目标列: {wavelet_target_col}")  # 打印小波分解目标列
# 使用原始特征列表
features = original_features  # 改为使用模型保存时记录的原始特征列表
# 检查测试数据是否包含所需特征
missing_features = [f for f in features if f not in data.columns]  # 找出测试数据中缺失的特征名
if missing_features:  # 若存在缺失特征则进入报错分支
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出 ValueError 并列出缺失的特征
print(f"\n使用原始特征: {features}")  # 打印实际使用的原始特征列表
# ===================== 数据预处理 =====================
# 加载训练数据以获取归一化参数
print("\n加载训练数据...")  # 打印加载训练数据的提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 再次准备候选编码列表用于读取训练数据
train_data = None  # 初始化训练数据变量为空
for enc in encodings:  # 遍历候选编码读取训练数据
    try:  # 进入 try 块捕获解码错误
        train_data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据 CSV，用于获取归一化参数
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码
        break  # 读取成功则跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码异常
        continue  # 继续尝试下一种编码
if train_data is None:  # 若所有编码均失败则 train_data 仍为 None
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出训练数据文件读取失败异常

# 对训练数据和测试数据进行小波分解
# 检查模型是使用全部特征小波分解还是仅温度偏差小波分解
use_partial_wavelet = checkpoint.get('wavelet_target_col') is not None  # 根据检查点是否记录目标列来区分新版/旧版分解方式

if use_partial_wavelet:  # 新版本：仅对温度偏差做小波分解
    # 新版本：仅对温度偏差进行小波分解
    print("\n检测到新版本模型（仅对温度偏差进行小波分解）")  # 打印检测到新版模型的提示
    print("\n对训练数据进行小波分解（仅分解温度偏差）...")  # 提示即将对训练数据做小波分解
    train_wavelet_features = wavelet_decompose_sliding(train_data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 对训练特征做滑动窗口小波分解
    
    print("\n对测试数据进行小波分解（仅分解温度偏差）...")  # 提示即将对测试数据做小波分解
    test_wavelet_features = wavelet_decompose_sliding(data[features], target_col=wavelet_target_col, wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 对测试特征做相同参数的小波分解
else:  # 旧版本模型走此分支
    # 旧版本：对所有特征进行小波分解
    print("\n检测到旧版本模型（对所有特征进行小波分解）")  # 打印检测到旧版模型的提示
    
    def wavelet_decompose_sliding_all(data, wavelet='db4', level=3, window_size=None):  # 定义兼容旧版的对所有特征做小波分解的函数
        """对所有特征进行小波分解（旧版本兼容）：每列展开为近似与各层细节系数特征。"""
        print(f"\n开始滑动窗口小波分解（所有特征）...")  # 打印开始对所有特征做小波分解的提示
        print(f"小波基函数: {wavelet}, 分解层数: {level}")  # 打印小波基函数与分解层数
        
        if window_size is None:  # 窗口长度未指定时自动推算
            window_size = min(2 ** (level + 6), len(data))  # 取 2 的 (level+6) 次方与样本数的较小值
        
        print(f"滑动窗口大小: {window_size}")  # 打印滑动窗口大小
        
        wavelet_features = pd.DataFrame(index=data.index)  # 创建与输入同索引的空结果表
        
        for col in data.columns:  # 遍历数据的每一列
            if col == 'timestamp':  # 跳过时间戳列
                continue  # 继续下一列
            
            signal = data[col].values  # 取出当前列的一维信号
            approx_list = []  # 收集近似系数
            detail_lists = [[] for _ in range(level)]  # 为每层准备细节系数列表
            
            for i in range(len(signal)):  # 逐时间点滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点并保证不越界
                window_signal = signal[start_idx:i+1]  # 截取以当前点为结尾的窗口片段
                
                if len(window_signal) < window_size:  # 窗口长度不足时先补齐
                    pad_width = window_size - len(window_signal)  # 计算需要补齐的元素个数
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值在左侧补齐
                
                try:  # 进入 try 块捕获异常
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层离散小波分解
                    approx_list.append(coeffs[0][-1])  # 记录近似系数末值
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 记录该层细节系数末值
                except Exception as e:  # 捕获分解过程中的异常
                    approx_list.append(window_signal[-1])  # 失败时退化为窗口最后一个原始值
                    for j in range(level):  # 遍历所有分解层
                        detail_lists[j].append(0.0)  # 失败时该层细节系数补 0
            
            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数特征列
            for j in range(level):  # 遍历所有分解层
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入对应层的细节系数特征列
        
        print(f"小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印分解完成提示与特征总数
        return wavelet_features  # 返回所有特征的小波分解结果
    
    print("\n对训练数据进行小波分解（所有特征）...")  # 提示对训练数据做全特征小波分解
    train_wavelet_features = wavelet_decompose_sliding_all(train_data[features], wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 调用兼容函数对训练特征做分解
    
    print("\n对测试数据进行小波分解（所有特征）...")  # 提示对测试数据做全特征小波分解
    test_wavelet_features = wavelet_decompose_sliding_all(data[features], wavelet=wavelet_type, level=wavelet_level, window_size=wavelet_window)  # 调用兼容函数对测试特征做分解

# 归一化（如果模型保存了scaler，直接使用）
if 'feature_scaler' in checkpoint and 'target_scaler' in checkpoint:  # 若检查点中保存了归一化器则直接复用
    print("\n使用模型保存的归一化器...")  # 打印使用已保存归一化器的提示
    feature_scaler = checkpoint['feature_scaler']  # 取出模型中保存的特征归一化器
    target_scaler = checkpoint['target_scaler']  # 取出模型中保存的目标归一化器
else:  # 否则用训练数据重新拟合归一化器
    print("\n使用训练数据重新拟合归一化器...")  # 打印重新拟合归一化器的提示
    feature_scaler = MinMaxScaler()  # 创建 MinMax 特征归一化器
    target_scaler = MinMaxScaler()  # 创建 MinMax 目标归一化器
    feature_scaler.fit(train_wavelet_features)  # 用训练小波特征拟合特征归一化器
    target_scaler.fit(train_data[[target]])  # 用训练集目标列拟合目标归一化器

# 归一化测试数据
test_features = pd.DataFrame(  # 将归一化后的测试特征重建为 DataFrame
    feature_scaler.transform(test_wavelet_features),  # 对测试小波特征执行归一化变换
    columns=feature_names  # 使用模型保存的特征名作为列名
)  # DataFrame 构造结束

print(f"  特征数量: {len(test_features.columns)}")  # 打印特征数量
print(f"  特征列表: {list(test_features.columns)}")  # 打印特征名列表

# ===================== 序列构造 =====================
pred_offset = 6  # 定义预测偏移为 6 个采样步

def create_sequences(data, time_step):  # 定义滑动窗口序列构造函数
    """创建时间序列样本：按 time_step 滑动窗口截取特征片段。

    返回形状为 (样本数, time_step, 特征数) 的 numpy 数组。
    """
    sequences = []  # 初始化序列样本列表
    for i in range(len(data) - time_step - pred_offset + 1):  # 滑动窗口起点，保证窗口与偏移均不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 截取长度为 time_step 的连续片段作为一个样本
    return np.array(sequences)  # 转换为 numpy 数组后返回

print("\n创建序列数据...")  # 打印创建序列数据的提示
# 归一化目标值
data_target_normalized = target_scaler.transform(data[[target]])  # 用目标归一化器归一化测试集目标值

# 创建序列
X_test = create_sequences(test_features, time_step)  # 由归一化测试特征构造滑动窗口序列
y_test = data_target_normalized[time_step + pred_offset - 1:].flatten()  # 按时间步与偏移对齐测试标签并展平为一维
X_test = X_test[:len(y_test)]  # 截断 X_test 使其长度与 y_test 一致

print(f"\n测试数据形状:")  # 提示即将打印测试数据形状
print(f"  X_test: {X_test.shape}")  # 打印 X_test 的形状
print(f"  y_test: {y_test.shape}")  # 打印 y_test 的形状

# ===================== 构建模型 =====================
print("\n构建模型...")  # 打印构建模型的提示
model = GRUWaveletModel(  # 实例化 GRUWaveletModel 并移至设备，参数见下方
    input_dim=input_dim,  # 输入维度取自模型检查点
    time_step=time_step,  # 时间步长取自模型检查点
    gru_hidden=128,  # 隐藏单元数固定为 128
    gru_layers=1,  # GRU 层数固定为 1
    dropout=0.1  # dropout 比例固定为 0.1
).to(device)  # 模型实例化结束并移至设备

model.load_state_dict(checkpoint['model_state_dict'])  # 载入检查点里的模型权重
model.eval()  # 切换模型为评估模式（关闭 dropout）

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型参数总量，使用千分位格式

# ===================== 模型预测 =====================
print("\n开始预测...")  # 打印开始预测的提示
X_test_tensor = torch.FloatTensor(X_test).to(device)  # 测试特征转为浮点张量并移至设备

with torch.no_grad():  # 关闭梯度计算以加速预测并节省显存
    y_pred = model(X_test_tensor).cpu().numpy()  # 执行预测并把结果转换回 numpy 数组

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将测试真值反归一化到原始量纲
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测值反归一化到原始量纲

# ===================== 评估指标 =====================
r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 对 MSE 开平方得到 RMSE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print("\n" + "=" * 60)  # 打印分隔线
print("GRU+小波模型测试结果")  # 打印结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R2 指标
print(f"MSE:       {mse:.4f}")  # 打印 MSE 指标
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE 指标
print(f"MAE:       {mae:.4f}")  # 打印 MAE 指标
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
# 只绘制前1600个时间步长
plot_length = min(1600, len(y_test_rescaled))  # 绘图长度取 1600 与样本数的较小值
y_test_plot = y_test_rescaled[:plot_length]  # 截取用于绘图的前若干真实值
y_pred_plot = y_pred_rescaled[:plot_length]  # 截取用于绘图的前若干预测值
# 使用步长索引作为X轴
steps = np.arange(plot_length)  # 生成与绘图长度对应的步数索引作为横轴
plt.figure(figsize=(15, 8))  # 创建 15x8 英寸的预测结果画布
y_test_flat = y_test_plot.flatten()  # 将真实值展平成一维数组
y_pred_flat = y_pred_plot.flatten()  # 将预测值展平成一维数组
plt.tick_params(axis='both', labelsize=16)  # 设置坐标轴刻度标签字号为 16
plt.plot(steps, y_test_flat, label='Real', linewidth=2, alpha=1, color='#1f77b4')  # 绘制真实值曲线，蓝色实线
plt.plot(steps, y_pred_flat, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 绘制预测值曲线，橙色实线
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签为 Timestep
plt.ylabel('TD (℃)', fontsize=16)  # 设置纵轴标签为温度偏差（摄氏度）
# plt.title(f'GRU Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=16)
plt.legend(fontsize=16)  # 显示图例并设置字号 16
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.tight_layout()  # 自动调整子图间距避免标签重叠
plt.text(0.5, 0.95, '(e)', transform=plt.gca().transAxes,  # 在图上添加子图编号文字 (e)
         fontsize=16, fontweight='bold', va='top', ha='center')  # 文本参数：字号 16、加粗、顶端对齐、水平居中
plt.tight_layout()  # 再次自动调整布局
output_filename = 'test_GRUWavelet_prediction.png'  # 定义预测结果图片文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存预测结果图
print(f"\n预测结果图已保存: {output_filename}")  # 打印图片保存路径提示

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算预测误差：预测值减真实值

plt.figure(figsize=(15, 10))  # 创建 15x10 英寸的误差分析画布

# 子图1: 预测值 vs 真实值散点图
plt.subplot(2, 2, 1)  # 选中第 1 个子图：预测值对真实值散点图
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, s=20)  # 绘制真实值与预测值的散点图
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],  # 绘制理想预测对角线的起点
         [y_test_rescaled.min(), y_test_rescaled.max()],  # 理想对角线的终点
         'r--', lw=2, label='理想预测线')  # 用红色虚线绘制对角线，线宽 2 并添加图例
plt.xlabel('真实温度偏差 (℃)', fontsize=11)  # 设置横轴标签为真实温度偏差
plt.ylabel('预测温度偏差 (℃)', fontsize=11)  # 设置纵轴标签为预测温度偏差
plt.title(f'预测值 vs 真实值 (R2={r2:.4f})', fontsize=12)  # 设置标题并显示 R2 数值
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线

# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 选中第 2 个子图：误差分布直方图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 绘制误差直方图，50 个分箱
plt.xlabel('预测误差 (℃)', fontsize=11)  # 设置横轴标签为预测误差
plt.ylabel('频数', fontsize=11)  # 设置纵轴标签为频数
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 设置标题并显示 MAE 数值
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 在 0 处绘制红色竖直虚线作为参考
plt.grid(True, alpha=0.3, axis='y')  # 仅在纵轴方向添加网格线

# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 选中第 3 个子图：误差时序变化图
error_timestamps = timestamps[:len(errors)]  # 取与误差等长的时间戳（注意：原脚本中未定义 timestamps 变量）
plt.plot(error_timestamps, errors, alpha=0.7, linewidth=1)  # 绘制误差随时间变化的折线
plt.xlabel('时间', fontsize=11)  # 设置横轴标签为时间
plt.ylabel('预测误差 (℃)', fontsize=11)  # 设置纵轴标签为预测误差
plt.title('误差时序变化', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 在 0 处绘制红色水平虚线作为参考
plt.grid(True, alpha=0.3)  # 添加半透明网格线

# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 选中第 4 个子图：误差箱线图
box_data = [errors]  # 把误差组织为箱线图的输入列表
plt.boxplot(box_data, tick_labels=['GRU+小波'])  # 绘制箱线图并标注模型名称为 GRU+小波
plt.ylabel('预测误差 (℃)', fontsize=11)  # 设置纵轴标签为预测误差
plt.title('误差统计分布', fontsize=12)  # 设置子图标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 在 0 处绘制红色水平虚线
plt.grid(True, alpha=0.3, axis='y')  # 仅在纵轴方向添加网格线

plt.tight_layout()  # 自动调整子图间距

error_filename = 'test_GRUWavelet_error_analysis.png'  # 定义误差分析图的文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印误差分析图保存提示

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印分隔线
print("误差统计信息")  # 打印误差统计信息标题
print("=" * 60)  # 打印分隔线
print(f"误差均值:     {np.mean(errors):.4f} ℃")  # 打印误差均值
print(f"误差标准差:   {np.std(errors):.4f} ℃")  # 打印误差标准差
print(f"误差最大值:   {np.max(errors):.4f} ℃")  # 打印误差最大值
print(f"误差最小值:   {np.min(errors):.4f} ℃")  # 打印误差最小值
print(f"误差中位数:   {np.median(errors):.4f} ℃")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\n测试完成！")  # 打印测试完成提示
