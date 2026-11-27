"""
GRU + 小波变换 + ECA 通道注意力的温度偏差预测实验脚本

本脚本读取温湿度历史数据，先做缺失值处理；为避免未来信息泄漏，先按时间顺序划分
训练集与测试集，再分别对温度偏差列做滑动窗口小波分解以提取多尺度特征，并用训练集
拟合的归一化器处理特征与目标值；随后构建 GRU + ECA 模型，先在训练集上做 5 折交叉
验证，再用全部训练数据训练最终模型，在测试集与一份新数据上完成预测、评估与可视化，
最后把模型状态及相关配置保存为 .pth 文件。

主要流程:
    1. 多编码尝试读取 CSV，构造 5 分钟间隔的时间戳；
    2. 连续变量线性插值、离散变量前后填充，必要时删除残留缺失行；
    3. 按 8:2 划分数据集，分别做滑动窗口小波分解，仅分解温度偏差列；
    4. 只在训练集上拟合 MinMaxScaler，构造时间窗口输入序列与标签；
    5. 定义 ECA 注意力层和 GRU 小波模型，用带早停的训练函数训练；
    6. 5 折交叉验证绘图、最终模型训练、测试集/新数据评估绘图与模型保存。
"""
import pandas as pd  # 导入 pandas 库，用于表格数据的读取与处理
import numpy as np  # 导入 numpy 库，用于数值计算与数组操作
import time  # 导入 time 库，用于统计模型训练耗时
# matplotlib后端配置：必须在导入pyplot之前设置
import matplotlib  # 导入 matplotlib 基础库，为后续切换绘图后端做准备
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片  # 在非交互环境下直接渲染保存图片，避免弹出图形窗口
import matplotlib.pyplot as plt  # 导入 pyplot 模块，用于绘制各类曲线和柱状图
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于把特征归一化到 0-1 区间
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入评估指标：R² 决定系数、MSE 均方误差、MAE 平均绝对误差
from sklearn.model_selection import train_test_split, KFold  # 导入数据集划分工具 train_test_split 与 K 折交叉验证迭代器 KFold
import torch  # 导入 PyTorch 主库，用于张量计算与深度学习训练
import torch.nn as nn  # 导入 torch.nn 模块，用于构建神经网络层
import torch.optim as optim  # 导入优化器模块，用于模型参数更新
from torch.utils.data import Dataset, DataLoader, TensorDataset  # 导入 Dataset/DataLoader/TensorDataset，实现批量化训练
import pywt  # 小波变换库  # 小波变换库，用于对每个时间窗口做多尺度分解

# ===================== 辅助函数 =====================
def get_model_size(model):  # 定义计算模型占用内存大小（MB）的辅助函数
    """
    计算模型大小（MB）

    参数:
        model: 待统计的 PyTorch 模型（nn.Module）
    返回:
        参数与缓冲区占用内存之和，单位为 MB
    """
    param_size = 0  # 初始化参数占用字节数计数器
    for param in model.parameters():  # 遍历模型的所有可学习参数
        param_size += param.nelement() * param.element_size()  # 累加当前参数的元素个数乘以单个元素所占字节数
    buffer_size = 0  # 初始化缓冲区占用字节数计数器
    for buffer in model.buffers():  # 遍历模型的缓冲区（如归一化层的运行统计量）
        buffer_size += buffer.nelement() * buffer.element_size()  # 累加缓冲区的总字节数
    size_all_mb = (param_size + buffer_size) / 1024**2  # 把总字节数换算为 MB（1MB = 1024² 字节）
    return size_all_mb  # 返回模型参数与缓冲区占用的总内存大小

def format_time(seconds):  # 定义把秒数格式化为易读时间字符串的函数
    """
    把秒数格式化为易读的时间字符串

    参数:
        seconds: 时长，单位为秒
    返回:
        按秒/分钟/小时自适应选择的格式化字符串
    """
    if seconds < 60:  # 若时长小于 60 秒，按秒显示
        return f"{seconds:.2f}秒"  # 返回保留两位小数的秒数字符串
    elif seconds < 3600:  # 否则若时长小于 1 小时，按分钟显示
        minutes = seconds / 60  # 把秒换算为分钟
        return f"{minutes:.2f}分钟"  # 返回保留两位小数的分钟字符串
    else:  # 否则按小时显示
        hours = seconds / 3600  # 把秒换算为小时
        return f"{hours:.2f}小时"  # 返回保留两位小数的小时字符串

# ===================== 环境配置与初始化 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择训练设备：有可用 GPU 则用 CUDA，否则用 CPU
print(f"使用设备: {device}")  # 打印当前使用的计算设备
if torch.cuda.is_available():  # 若检测到可用的 GPU
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称
    print(f"GPU数量: {torch.cuda.device_count()}")  # 打印可用 GPU 的数量
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，避免图中中文显示为方块
plt.rcParams['axes.unicode_minus'] = False  # 关闭 Unicode 负号显示，保证负号正常呈现
# ===================== 数据加载与预处理 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 定义读取 CSV 时依次尝试的编码列表
data = None  # 初始化数据变量为空，后续用于保存读取结果
for enc in encodings:  # 依次尝试每种编码读取文件
    try:  # 捕获编码解码异常
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取温湿度训练数据 CSV
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功读取所用的编码
        break  # 读取成功后跳出循环，不再尝试其它编码
    except (UnicodeDecodeError, UnicodeError):  # 若出现解码错误则尝试下一种编码
        continue  # 继续尝试下一种编码

if data is None:  # 若所有编码都读取失败
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出异常终止程序，提示无法读取训练数据
data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔生成时间戳列，起始时间为 2025/5/26 8:22
features = ['日龄','温度偏差','外界温度','CO2','目标温度','相对湿度']  # 定义候选特征列表
target = '温度偏差'  # 设定预测目标列为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列
# ===================== 缺失值处理 =====================
print(f"\n{'=' * 60}")  # 打印由等号组成的分隔线
print("缺失值处理")  # 打印标题：缺失值处理
print(f"{'=' * 60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据的行数与列数
print(f"\n各列缺失值统计:")  # 打印提示：开始统计各列缺失值
for col in features + [target]:  # 遍历所有特征列与目标列
    if col in data.columns:  # 仅处理数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(data) * 100  # 计算该列缺失值占全部样本的百分比
        if nan_count > 0:  # 只打印存在缺失值的列，跳过完整列
            print(f"  {col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失值数量与所占比例

# 定义连续变量和离散变量
continuous_vars = ['温度偏差', 'CO2', '目标温度', '外界温度','相对湿度']  # 连续变量  # 定义需要进行线性插值的连续变量列表
discrete_vars = ['日龄']  # 离散变量  # 定义只做前后填充的离散变量列表

# 对连续变量使用插值+前后填充
for col in continuous_vars:  # 遍历连续变量列表
    if col in data.columns and data[col].isna().sum() > 0:  # 若该列存在且含有缺失值
        print(f"\n处理 {col}:")  # 打印正在处理的列名
        before_nan = data[col].isna().sum()  # 记录插值前的缺失值数量

        # 1. 线性插值（适合时间序列的连续变化）
        data[col] = data[col].interpolate(method='linear', limit_direction='both')  # 用线性插值填补缺失值，前后两个方向都填充

        # 2. 前向填充（填充开头的缺失值）
        data[col] = data[col].fillna(method='ffill')  # 前向填充开头仍然缺失的值

        # 3. 后向填充（填充结尾的缺失值）
        data[col] = data[col].fillna(method='bfill')  # 后向填充结尾仍然缺失的值

        after_nan = data[col].isna().sum()  # 记录插值后的缺失值数量
        print(f"  插值前: {before_nan} 个缺失值")  # 打印插值前的缺失值个数
        print(f"  插值后: {after_nan} 个缺失值")  # 打印插值后的缺失值个数

# 对离散变量使用前后填充
for col in discrete_vars:  # 遍历离散变量列表
    if col in data.columns and data[col].isna().sum() > 0:  # 若该离散列存在且含有缺失值
        print(f"\n处理 {col}:")  # 打印正在处理的离散列名
        before_nan = data[col].isna().sum()  # 记录填充前的缺失值数量

        # 前向填充
        data[col] = data[col].fillna(method='ffill')  # 对离散变量执行前向填充
        # 后向填充
        data[col] = data[col].fillna(method='bfill')  # 对离散变量执行后向填充

        after_nan = data[col].isna().sum()  # 记录填充后的缺失值数量
        print(f"  填充前: {before_nan} 个缺失值")  # 打印填充前的缺失值个数
        print(f"  填充后: {after_nan} 个缺失值")  # 打印填充后的缺失值个数

# 检查是否还有缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计所有特征与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 若处理后仍有缺失值
    print(f"\n⚠️  警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印警告，并说明将删除含缺失值的行
    data = data.dropna(subset=features + [target])  # 删除特征列或目标列中含缺失值的样本行

print(f"\n处理后数据形状: {data.shape}")  # 打印缺失值处理后的数据规模
print(f"{'=' * 60}\n")  # 打印分隔线并换行

# 保存原始数据副本（用于后续测试集预测）
data_original = data.copy()  # 保存一份缺失值处理后的数据副本，供后续使用

print(f"\n【GRU + 小波变换模型】")  # 打印模型方案标题
print(f"使用特征: {features}")  # 打印实际参与建模的特征列表
print(f"特征数量: {len(features)}")  # 打印特征数量


# ===================== 小波变换特征提取（仅对温度偏差） =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):  # 定义滑动窗口小波分解函数，仅对目标列分解以避免未来信息泄漏
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
    print(f"\n开始滑动窗口小波分解（仅对'{target_col}'进行分解）...")  # 打印开始分解的提示信息
    print(f"小波基函数: {wavelet}")  # 打印所选的小波基函数
    print(f"分解层数: {level}")  # 打印小波分解的层数

    # 自动计算窗口大小：确保足够大以进行有效的小波分解
    if window_size is None:  # 若调用时没有指定滑动窗口大小
        window_size = min(2 ** (level + 6), len(data))  # 根据分解层数自适应  # 取 2 的 (level+6) 次方与数据长度中的较小值作为窗口大小

    print(f"滑动窗口大小: {window_size}")  # 打印实际使用的滑动窗口大小

    wavelet_features = pd.DataFrame(index=data.index)  # 创建空的结果表，索引与原数据保持一致

    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 跳过时间戳列，时间戳不参与小波分解
            continue  # 进入下一列的处理

        signal = data[col].values  # 取出当前列的数值数组

        # 只对目标列进行小波分解
        if col == target_col:  # 若当前列是需要做小波分解的目标列
            print(f"  对 '{col}' 进行小波分解...")  # 打印正在做小波分解的列名
            approx_list = []  # 初始化近似系数列表
            detail_lists = [[] for _ in range(level)]  # 为每一层细节系数各初始化一个空列表

            for i in range(len(signal)):  # 按时间点逐格滑动窗口
                # 只使用当前时间点之前的数据（包括当前点）
                start_idx = max(0, i - window_size + 1)  # 计算当前窗口的起始下标，保证不小于 0
                window_signal = signal[start_idx:i + 1]  # 截取当前时间点及其之前的窗口数据，不使用未来信息

                # 如果窗口太小，用边界值填充到window_size
                if len(window_signal) < window_size:  # 若窗口长度不足以做一次完整分解
                    pad_width = window_size - len(window_signal)  # 计算需要在窗口前端补充的点数
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边界值在窗口前端填充至指定长度

                # 小波分解
                try:  # 尝试对当前窗口信号做小波分解
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 调用 wavedec 进行 level 层分解，得到近似与各层细节系数

                    # 取最后一个值作为当前时间点的特征
                    approx_list.append(coeffs[0][-1])  # 取近似系数的最后一个值作为当前时刻的近似特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取每层细节系数最后一个值，系数为空则补 0
                except Exception as e:  # 若分解过程出现异常
                    # 如果分解失败，使用原始值
                    approx_list.append(window_signal[-1])  # 降级处理：用窗口最后一个原始值作为近似特征
                    for j in range(level):  # 遍历每一层细节系数
                        detail_lists[j].append(0.0)  # 异常时把该层细节系数补为 0

            # 添加小波分解特征
            wavelet_features[f'{col}_approx'] = approx_list  # 把近似系数序列添加为新特征列
            for j in range(level):  # 遍历各层细节系数
                wavelet_features[f'{col}_detail{j + 1}'] = detail_lists[j]  # 把每一层细节系数添加为新特征列
        else:  # 若当前列不是分解目标列
            # 其他列保持原样，不进行小波分解
            print(f"  '{col}' 保持原样（不分解）")  # 打印该列不做分解的提示
            wavelet_features[col] = signal  # 把该列原始值直接复制到结果表中

    print(f"滑动窗口小波分解完成！生成特征数: {len(wavelet_features.columns)}")  # 打印分解完成提示与生成的特征总数
    print(f"  - 小波分解特征: {level + 1} 个（来自'{target_col}'）")  # 打印由温度偏差分解产生的小波特征数量
    print(f"  - 原始特征: {len(data.columns) - 1} 个（其他变量）")  # 打印保留下来的其他原始特征数量
    return wavelet_features  # 返回包含原始特征与小波特征的 DataFrame


# ===================== 数据集划分（在小波分解之前！） =====================
# 先定义时间窗口参数
time_step = 36  # GRU时间窗口：36步 × 5分钟 = 3小时  # 定义 GRU 输入序列的历史时间步长（36 步 × 5 分钟 = 3 小时）
pred_offset = 6  # 预测偏移：6步 × 5分钟 = 30分钟  # 定义标签相对输入序列的预测偏移（6 步 × 5 分钟 = 30 分钟）

print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  GRU时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 打印 GRU 时间窗口对应的分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 打印预测偏移对应的分钟数

# 先划分数据集，避免数据泄漏
split_index = int(len(data) * 0.8)  # 按 8:2 的比例计算训练集与测试集的划分下标
train_data_raw = data.iloc[:split_index].copy()  # 取前 80% 的数据作为训练集原始数据
test_data_raw = data.iloc[split_index:].copy()  # 取后 20% 的数据作为测试集原始数据

print(f"\n数据集划分:")  # 打印数据集划分标题
print(f"训练集大小: {len(train_data_raw)} ({len(train_data_raw) / len(data) * 100:.1f}%)")  # 打印训练集样本数及其占比
print(f"测试集大小: {len(test_data_raw)} ({len(test_data_raw) / len(data) * 100:.1f}%)")  # 打印测试集样本数及其占比

# 分别对训练集和测试集进行小波分解（避免数据泄漏）
# 只对温度偏差进行小波分解，其他特征保持原样
print("\n对训练集进行小波分解（仅分解温度偏差）...")  # 提示开始对训练集做小波分解
# 使用与GRU相同的时间窗口
wavelet_window = time_step  # 让小波窗口和GRU窗口一致（3小时）  # 让小波窗口大小与 GRU 时间步长保持一致
print(f"小波窗口大小设置为: {wavelet_window} 步 (与GRU时间窗口一致)")  # 打印设置的小波窗口大小
train_wavelet_features = wavelet_decompose_sliding(train_data_raw[features], target_col='温度偏差', wavelet='db4',  # 对训练集特征执行滑动窗口小波分解
                                                   level=3, window_size=wavelet_window)

print("\n对测试集进行小波分解（仅分解温度偏差）...")  # 提示开始对测试集做小波分解
test_wavelet_features = wavelet_decompose_sliding(test_data_raw[features], target_col='温度偏差', wavelet='db4',  # 对测试集特征执行滑动窗口小波分解
                                                  level=3, window_size=wavelet_window)


# 可视化小波分解效果（以第一个特征为例）
def visualize_wavelet_decomposition(original_signal, wavelet_features, feature_name,
                                    save_path='wavelet_decomposition.png'):
    """
    可视化小波分解的结果

    参数:
        original_signal: 原始信号的一维数组
        wavelet_features: 含近似系数与细节系数的特征表
        feature_name: 被分解的特征名，用于拼接系数列名
        save_path: 图片保存路径
    返回:
        无返回值；图片保存到本地并关闭画布
    """  # 定义小波分解结果的可视化函数
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))  # 创建 5 行 1 列的子图画布：原始信号 + 4 个分解系数

    # 原始信号
    axes[0].plot(original_signal, linewidth=1, color='black')  # 在第一个子图绘制原始信号（黑色曲线）
    axes[0].set_title(f'原始信号: {feature_name}', fontsize=12, fontweight='bold')  # 设置原始信号子图的标题
    axes[0].set_ylabel('幅值')  # 设置纵轴标签为幅值
    axes[0].grid(True, alpha=0.3)  # 为该子图添加半透明网格线

    # 近似系数（低频 - 趋势）
    axes[1].plot(wavelet_features[f'{feature_name}_approx'].values, linewidth=1, color='blue')  # 在第二个子图绘制近似系数（低频趋势成分）
    axes[1].set_title('近似系数 (低频成分 - 趋势)', fontsize=12)  # 设置近似系数子图的标题
    axes[1].set_ylabel('幅值')  # 设置近似系数子图的纵轴标签
    axes[1].grid(True, alpha=0.3)  # 为近似系数子图添加网格线

    # 细节系数（高频 - 波动）
    colors = ['green', 'orange', 'red']  # 定义三层细节系数对应的绘图颜色
    for i in range(1, 4):  # 循环绘制三层细节系数
        axes[i + 1].plot(wavelet_features[f'{feature_name}_detail{i}'].values,  # 绘制第 i 层细节系数（高频波动成分）
                         linewidth=1, color=colors[i - 1])
        axes[i + 1].set_title(f'细节系数 Level {i} (高频成分 - 波动)', fontsize=12)  # 设置细节系数子图的标题
        axes[i + 1].set_ylabel('幅值')  # 设置细节系数子图的纵轴标签
        axes[i + 1].grid(True, alpha=0.3)  # 为细节系数子图添加网格线

    axes[-1].set_xlabel('时间步', fontsize=11)  # 为最后一个子图设置横轴标签为时间步
    plt.tight_layout()  # 自动调整各子图之间的间距
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 把图片保存到指定路径，分辨率为 150dpi
    print(f"小波分解可视化已保存: {save_path}")  # 打印图片保存路径提示
    plt.close()  # 关闭画布，释放内存


# 可视化训练集温度偏差的小波分解
first_feature = '温度偏差'  # 指定要可视化的特征为温度偏差
visualize_wavelet_decomposition(  # 调用可视化函数绘制温度偏差的小波分解结果
    train_data_raw[first_feature].values,
    train_wavelet_features,
    first_feature,
    'wavelet_decomposition_temperature_deviation.png'
)

# 合并特征说明
print(f"\n原始特征数: {len(features)}")  # 打印原始特征数量
print(f"小波分解后特征数: {len(train_wavelet_features.columns)}")  # 打印小波分解后的总特征数量
print(f"  - 温度偏差的小波特征: 4 个（1个近似系数 + 3个细节系数）")  # 说明温度偏差产生的 4 个小波特征构成
print(f"  - 其他原始特征: {len(features) - 1} 个")  # 说明保留下来的其他原始特征数量

# 直接使用小波分解后的特征（已包含原始特征和小波特征）
train_combined_features = train_wavelet_features  # 训练集直接使用小波分解后的特征表
test_combined_features = test_wavelet_features  # 测试集直接使用小波分解后的特征表
feature_names = list(train_wavelet_features.columns)  # 记录最终使用的特征名称列表

print(f"\n使用模式: 温度偏差小波分解 + 其他原始特征")  # 打印当前的特征组合方案
print(f"总特征数: {len(feature_names)}")  # 打印参与建模的总特征数
print(f"特征列表: {feature_names}")  # 打印最终的特征名称列表

# ===================== 归一化（只在训练集上fit！） =====================
print("\n归一化处理（避免数据泄漏）...")  # 提示开始归一化处理，强调避免数据泄漏
feature_scaler = MinMaxScaler()  # 创建用于特征归一化的 MinMaxScaler
target_scaler = MinMaxScaler()  # 创建用于目标值归一化的 MinMaxScaler

# 只在训练集上fit
feature_scaler.fit(train_combined_features)  # 只在训练集上拟合特征归一化器
target_scaler.fit(train_data_raw[[target]])  # 只在训练集上拟合目标值归一化器

# 分别transform训练集和测试集
train_combined_features = pd.DataFrame(  # 用训练集拟合出的 scaler 变换训练集特征
    feature_scaler.transform(train_combined_features),
    columns=feature_names
)
test_combined_features = pd.DataFrame(  # 用训练集拟合出的 scaler 变换测试集特征，避免泄漏
    feature_scaler.transform(test_combined_features),
    columns=feature_names
)

train_data_raw[target] = target_scaler.transform(train_data_raw[[target]])  # 把训练集目标值归一化到 0-1 区间
test_data_raw[target] = target_scaler.transform(test_data_raw[[target]])  # 用同一个 scaler 把测试集目标值归一化


# ===================== 序列数据构造（分别对训练集和测试集） =====================
# time_step 和 pred_offset 已在前面定义

def create_sequences(data, time_step):  # 定义把二维特征表切成固定长度时间窗口序列的函数
    """
    将二维特征表切分为固定长度的时间窗口序列

    参数:
        data: 已归一化的二维特征表（DataFrame）
        time_step: 每个输入样本包含的历史时间步数
    返回:
        numpy 数组，形状为 (样本数, time_step, 特征数)
    """
    sequences = []  # 初始化序列列表
    for i in range(len(data) - time_step - pred_offset):  # 滑动切窗，并预留预测偏移以避免使用未来标签
        sequences.append(data.iloc[i:i + time_step].values)  # 截取 time_step 行作为一个输入样本
    return np.array(sequences)  # 把序列列表堆叠成三维 numpy 数组并返回


# 分别为训练集和测试集创建序列（避免跨边界）
print("\n创建训练集序列...")  # 提示开始创建训练集序列
X_train = create_sequences(train_combined_features, time_step)  # 生成训练集的输入序列
y_train = train_data_raw[target].values[time_step + pred_offset:]  # 取与输入序列时间窗口对应的训练集标签（偏移之后）
y_train = y_train[:len(X_train)]  # 确保长度匹配  # 截断标签，使其长度与输入序列数量一致

print("\n创建测试集序列...")  # 提示开始创建测试集序列
X_test = create_sequences(test_combined_features, time_step)  # 生成测试集的输入序列
y_test = test_data_raw[target].values[time_step + pred_offset:]  # 取与输入序列对应的测试集标签
y_test = y_test[:len(X_test)]  # 确保长度匹配  # 截断测试集标签，使其长度与输入序列数量一致

print(f"\n序列数据形状:")  # 打印序列数据形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集输入与标签的形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集输入与标签的形状


# ===================== PyTorch模型定义 =====================
class ECALayer(nn.Module):  # 定义 ECA 高效通道注意力层
    """
    ECA-Net: Efficient Channel Attention
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    通过1D卷积实现高效的通道注意力机制，避免降维带来的信息损失
    """

    def __init__(self, channels, gamma=2, b=1):  # 初始化 ECA 层：根据通道数自适应确定一维卷积核大小
        """
        初始化 ECA 通道注意力层

        参数:
            channels: 输入特征的通道数
            gamma: 计算卷积核尺寸时使用的缩放系数
            b: 计算卷积核尺寸时使用的偏置项
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 自适应计算卷积核大小
        t = int(abs((np.log2(channels) + b) / gamma))  # 按公式自适应计算卷积核尺寸参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数  # 若 t 为偶数则加 1，保证卷积核大小为奇数

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 定义全局平均池化，把时间维压缩成 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 定义一维卷积，用可学习参数实现跨通道信息交互且不降维
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活，把注意力权重映射到 0-1 之间

    def forward(self, x):  # 定义前向传播：对输入特征进行通道注意力加权
        """
        对输入特征执行通道注意力加权

        参数:
            x: 形状为 (batch, time_step, channels) 的输入张量
        返回:
            通道加权后的张量，形状保持不变
        """
        # x: (batch, time_step, channels)        # 转换为 (batch, channels, time_step) 以便进行池化
        x_permuted = x.permute(0, 2, 1)  # (batch, channels, time_step)  # 交换时间维与通道维，使张量满足池化的输入格式

        # 全局平均池化: (batch, channels, 1)
        y = self.avg_pool(x_permuted)  # 全局平均池化，得到每个通道的全局描述

        # 1D卷积: (batch, 1, channels) -> (batch, 1, channels)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 转置后做一维卷积再转回，实现通道间的信息交互

        # Sigmoid激活
        y = self.sigmoid(y)  # (batch, channels, 1)  # 用 Sigmoid 归一化得到每个通道的注意力权重

        # 通道注意力加权
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 把注意力权重按通道广播后乘回原始输入特征

        # 转换回 (batch, time_step, channels)
        return x_permuted.permute(0, 2, 1)  # 把维度顺序转换回原格式后返回加权结果


class GRUWaveletModel(nn.Module):  # 定义 GRU + 小波 + ECA 的组合预测模型
    """
    GRU + 小波变换 + ECA注意力机制
    结合小波分解的多尺度特征和GRU的时序建模能力
    输入:  (batch, time_step, input_dim)
    输出:  (batch, 1)
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 初始化 GRU 小波模型，配置 ECA、GRU 与输出层
        """
        初始化 GRU 小波预测模型

        参数:
            input_dim: 输入特征维度
            time_step: 时间窗口长度
            gru_hidden: GRU 隐藏层单元数
            gru_layers: GRU 的层数
            dropout: 多层 GRU 时的 dropout 比例
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存 GRU 隐藏层单元数

        # ECA-Net注意力模块：在GRU之前增强输入特征
        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 通道注意力模块

        # 单层GRU编码器：提取时序特征
        self.gru = nn.GRU(  # 定义单层单向 GRU 编码器，提取时序特征
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        # 输出层：直接从GRU隐藏状态映射到预测值
        self.fc = nn.Linear(gru_hidden, 1)  # 定义全连接输出层，把隐藏状态映射为单一预测值

    def forward(self, x):  # 定义前向传播：依次经过 ECA、GRU 与全连接层
        """
        前向传播：输入依次经过 ECA 注意力、GRU 编码与全连接层

        参数:
            x: 形状为 (batch, time_step, input_dim) 的输入序列
        返回:
            形状为 (batch, 1) 的预测值张量
        """
        # x: (batch, time_step, input_dim)
        # ECA注意力增强输入特征
        x = self.eca(x)  # (batch, time_step, input_dim)  # 用 ECA 模块增强输入特征的通道表达

        # GRU编码：提取时序特征
        out, h = self.gru(x)  # out: (batch, time_step, hidden), h: (layers, batch, hidden)  # GRU 编码，得到各时刻输出与最后时刻隐藏状态

        # 使用最后时刻的隐藏状态
        out_last = out[:, -1, :]  # (batch, hidden)  # 取最后一个时间步的输出作为整段序列的表示

        # 预测输出
        return self.fc(out_last)  # (batch, 1)  # 经全连接层输出最终预测值


class EarlyStopping:  # 定义早停策略类，在验证损失不再改善时停止训练
    """
    早停策略：验证损失长期不再改善时停止训练，并保留最佳模型权重

    参数:
        patience: 容忍验证损失不下降的 epoch 数
        min_delta: 视为一次改善所需的最小损失变化量
    """

    def __init__(self, patience=10, min_delta=0):  # 初始化早停器，设置容忍轮数与最小改善阈值
        """
        初始化早停器

        参数:
            patience: 允许验证损失连续不改善的 epoch 数
            min_delta: 判定为改善所需的最小损失下降量
        """
        self.patience = patience  # 保存容忍的 epoch 数
        self.min_delta = min_delta  # 保存被视为改善的最小损失变化量
        self.counter = 0  # 初始化未改善计数器为 0
        self.best_loss = None  # 初始化最佳损失为 None
        self.early_stop = False  # 初始化早停标志为 False
        self.best_model = None  # 初始化最佳模型权重为 None

    def __call__(self, val_loss, model):  # 定义每次 epoch 后的检查逻辑：更新最佳模型或触发早停
        """
        每个 epoch 调用一次，根据验证损失更新最佳模型或触发早停

        参数:
            val_loss: 当前 epoch 的验证损失
            model: 当前模型，用于保存其最佳状态字典
        返回:
            无返回值；结果通过 early_stop 与 best_model 属性对外传递
        """
        if self.best_loss is None:  # 若是首次调用，还没有最佳损失
            self.best_loss = val_loss  # 把当前验证损失设为初始最佳损失
            self.best_model = model.state_dict().copy()  # 保存当前模型权重作为最佳模型
        elif val_loss > self.best_loss - self.min_delta:  # 若当前验证损失没有比最佳损失明显下降
            self.counter += 1  # 未改善计数器加 1
            if self.counter >= self.patience:  # 若连续未改善次数达到容忍上限
                self.early_stop = True  # 置位早停标志
        else:  # 否则说明当前验证损失有改善
            self.best_loss = val_loss  # 更新最佳损失为当前值
            self.best_model = model.state_dict().copy()  # 更新保存的最佳模型权重
            self.counter = 0  # 重置未改善计数器


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 定义模型训练函数，含验证与早停逻辑
    """
    训练模型并在每轮结束后做验证，借助早停策略保留最佳权重

    参数:
        model: 待训练的 PyTorch 模型
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        epochs: 最大训练轮数
        lr: Adam 优化器的学习率
        patience: 早停容忍的轮数
    返回:
        train_losses, val_losses: 每轮的训练损失与验证损失列表
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 定义 Adam 优化器，按学习率更新参数
    early_stopping = EarlyStopping(patience=patience)  # 实例化早停器，传入容忍轮数

    train_losses = []  # 初始化每轮训练损失记录列表
    val_losses = []  # 初始化每轮验证损失记录列表

    for epoch in range(epochs):  # 按 epoch 循环进行训练
        # 训练阶段
        model.train()  # 把模型切换为训练模式，启用 dropout 等训练专属层
        train_loss = 0.0  # 初始化本轮累计训练损失
        for batch_x, batch_y in train_loader:  # 按批次遍历训练数据加载器
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把当前批次的输入与标签移动到指定设备

            optimizer.zero_grad()  # 清空上一个批次残留的梯度
            outputs = model(batch_x)  # 前向传播，得到模型预测输出
            loss = criterion(outputs, batch_y)  # 计算预测输出与真实标签之间的损失
            loss.backward()  # 反向传播，计算各参数的梯度
            optimizer.step()  # 优化器根据梯度更新模型参数

            train_loss += loss.item()  # 累加当前批次的损失值

        train_loss /= len(train_loader)  # 除以批次数，得到本轮平均训练损失
        train_losses.append(train_loss)  # 把本轮训练损失加入记录列表

        # 验证阶段
        model.eval()  # 把模型切换为评估模式，关闭 dropout 等层
        val_loss = 0.0  # 初始化本轮累计验证损失
        with torch.no_grad():  # 关闭梯度计算，节省显存并加速验证
            for batch_x, batch_y in val_loader:  # 按批次遍历验证数据加载器
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 把验证批次数据移动到指定设备
                outputs = model(batch_x)  # 前向传播，得到验证集预测
                loss = criterion(outputs, batch_y)  # 计算验证集上的损失
                val_loss += loss.item()  # 累加当前验证批次的损失值

        val_loss /= len(val_loader)  # 除以批次数得到本轮平均验证损失
        val_losses.append(val_loss)  # 把本轮验证损失加入记录列表

        if (epoch + 1) % 10 == 0:  # 每 10 个 epoch 打印一次训练状态
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')  # 打印当前 epoch 的训练损失与验证损失

        # 早停检查
        early_stopping(val_loss, model)  # 调用早停器，根据验证损失更新状态
        if early_stopping.early_stop:  # 若早停标志被触发
            print(f"早停触发于 epoch {epoch + 1}")  # 打印早停触发时的 epoch 编号
            model.load_state_dict(early_stopping.best_model)  # 加载早停器保存的最佳模型权重
            break  # 跳出 epoch 循环，提前结束训练

    # 加载最佳模型
    if early_stopping.best_model is not None:  # 若训练过程中保存过最佳模型权重
        model.load_state_dict(early_stopping.best_model)  # 训练结束后把最佳权重加载回模型

    return train_losses, val_losses  # 返回每轮的训练损失与验证损失列表


# ===================== K折交叉验证训练（仅在训练集上） =====================
# 注意：K折交叉验证只在训练集上进行，不包含测试集
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建 5 折交叉验证划分器，打乱数据并固定随机种子
val_losses_all = []  # 初始化各折最佳验证损失列表
history_list = []  # 初始化各折训练历史列表
input_dim = len(feature_names)  # 使用小波特征后的总特征数  # 输入特征维度等于最终特征数量

print("\n========== 开始5折交叉验证（仅训练集） ==========")  # 打印交叉验证开始提示
print(f"交叉验证数据量: {len(X_train)} 样本")  # 打印参与交叉验证的样本数量

for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):  # 按折循环，取出每折的训练下标与验证下标
    print(f"\n{'=' * 50}")  # 打印分隔线
    print(f"第 {fold}/5 折训练")  # 打印当前是第几折训练
    print(f"{'=' * 50}")  # 打印分隔线

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # 按当前折下标划分训练集与验证集的输入
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # 按当前折下标划分训练集与验证集的标签

    print(f"训练集样本数: {len(X_train_fold)}")  # 打印当前折训练集的样本数
    print(f"验证集样本数: {len(X_val_fold)}")  # 打印当前折验证集的样本数

    X_train_tensor = torch.FloatTensor(X_train_fold)  # 把当前折的训练输入转换为浮点张量
    y_train_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)  # 把训练标签转为张量，并增加一维以匹配输出形状
    X_val_tensor = torch.FloatTensor(X_val_fold)  # 把当前折的验证输入转换为浮点张量
    y_val_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)  # 把验证标签转为张量并增加一维

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 用训练张量构建 TensorDataset 数据集对象
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # 用验证张量构建 TensorDataset 数据集对象
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构建训练数据加载器，批量大小为 32 并打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 构建验证数据加载器，批量大小为 32 且不打乱

    # 构建GRU+小波模型
    model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1).to(device)  # 实例化当前折的 GRU 小波模型并移动到指定设备

    if fold == 1:  # 仅在第一折打印模型信息，避免重复输出
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")  # 打印模型可训练参数的总数

    train_losses, val_losses = train_model(model, train_loader, val_loader,  # 在当前折的数据上训练模型，返回损失记录
                                           epochs=200, lr=0.001, patience=10)
    history_list.append({  # 把当前折的训练与验证损失历史保存下来
        'train_losses': train_losses,
        'val_losses': val_losses
    })

    best_val_loss = min(val_losses)  # 取当前折出现过的最低验证损失
    val_losses_all.append(best_val_loss)  # 把该折最佳验证损失加入汇总列表
    print(f"\n第 {fold} 折最佳验证损失: {best_val_loss:.6f}")  # 打印当前折的最佳验证损失

print(f"\n{'=' * 50}")  # 打印分隔线
print("5折交叉验证结果汇总")  # 打印交叉验证结果汇总标题
print(f"{'=' * 50}")  # 打印分隔线
for i, loss in enumerate(val_losses_all, 1):  # 遍历汇总列表中的每一折最佳验证损失
    print(f"第 {i} 折验证损失: {loss:.6f}")  # 逐折打印验证损失
print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")  # 打印各折验证损失的平均值
print(f"验证损失标准差: {np.std(val_losses_all):.6f}")  # 打印各折验证损失的标准差

# ===================== 绘制K折交叉验证训练过程曲线 =====================
plt.figure(figsize=(15, 10))  # 创建用于绘制交叉验证结果的画布

plt.subplot(2, 2, 1)  # 选中 2×2 布局中的第 1 个子图
for i, history in enumerate(history_list, 1):  # 遍历每一折的训练历史
    plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制该折的训练损失曲线，设置透明度以区分折
plt.title('5-Fold Cross Validation - Training Loss', fontsize=14)  # 设置第 1 个子图的标题
plt.xlabel('Epoch')  # 设置横轴标签为 epoch
plt.ylabel('Loss (MSE)')  # 设置纵轴标签为 MSE 损失
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线

plt.subplot(2, 2, 2)  # 选中第 2 个子图
for i, history in enumerate(history_list, 1):  # 再次遍历每一折的训练历史
    plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制该折的验证损失曲线
plt.title('5-Fold Cross Validation - Validation Loss', fontsize=14)  # 设置第 2 个子图的标题
plt.xlabel('Epoch')  # 设置横轴标签为 epoch
plt.ylabel('Loss (MSE)')  # 设置纵轴标签为 MSE 损失
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线

plt.subplot(2, 2, 3)  # 选中第 3 个子图
folds = [f'Fold {i}' for i in range(1, 6)]  # 生成 5 个折的名称标签列表
plt.bar(folds, val_losses_all, color='steelblue', alpha=0.7)  # 用柱状图展示各折的最佳验证损失
plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',  # 用红色虚线标出所有折的平均验证损失
            label=f'Mean: {np.mean(val_losses_all):.6f}')
plt.title('Best Validation Loss per Fold', fontsize=14)  # 设置第 3 个子图的标题
plt.xlabel('Fold')  # 设置横轴标签为折编号
plt.ylabel('Validation Loss (MSE)')  # 设置纵轴标签为验证损失
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3, axis='y')  # 只沿 y 轴方向添加网格线

plt.subplot(2, 2, 4)  # 选中第 4 个子图
max_epochs = max(len(h['train_losses']) for h in history_list)  # 找出各折中最长的训练 epoch 数
avg_train_losses = []  # 初始化跨折平均训练损失列表
avg_val_losses = []  # 初始化跨折平均验证损失列表
for epoch in range(max_epochs):  # 逐个 epoch 计算跨折平均损失
    train_vals = [h['train_losses'][epoch] for h in history_list if epoch < len(h['train_losses'])]  # 收集当前 epoch 各折的训练损失，未训练到该轮的折跳过
    val_vals = [h['val_losses'][epoch] for h in history_list if epoch < len(h['val_losses'])]  # 收集当前 epoch 各折的验证损失，未训练到该轮的折跳过
    if train_vals:  # 若当前 epoch 存在可用的训练损失值
        avg_train_losses.append(np.mean(train_vals))  # 计算并保存当前 epoch 的平均训练损失
    if val_vals:  # 若当前 epoch 存在可用的验证损失值
        avg_val_losses.append(np.mean(val_vals))  # 计算并保存当前 epoch 的平均验证损失

plt.plot(avg_train_losses, label='Avg Train Loss', linewidth=2)  # 绘制跨折平均训练损失曲线
plt.plot(avg_val_losses, label='Avg Val Loss', linewidth=2)  # 绘制跨折平均验证损失曲线
plt.title('5-Fold Average Loss Curves', fontsize=14)  # 设置第 4 个子图的标题
plt.xlabel('Epoch')  # 设置横轴标签为 epoch
plt.ylabel('Loss (MSE)')  # 设置纵轴标签为 MSE 损失
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线

plt.tight_layout()  # 自动调整所有子图的布局
plt.savefig('kfold_training_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把交叉验证结果图保存到本地
print("\nK折交叉验证损失曲线已保存: kfold_training_GRUWavelet_pytorch.png")  # 打印交叉验证曲线保存提示

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印分隔线
print("训练最终模型（使用全部训练数据）")  # 打印开始训练最终模型的提示
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印用于最终训练的训练集样本数
print(f"测试集样本数: {len(X_test)}")  # 打印用于评估的测试集样本数

X_train_tensor = torch.FloatTensor(X_train)  # 把最终训练集输入转换为张量
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 把最终训练集标签转换为张量并增加一维
X_test_tensor = torch.FloatTensor(X_test)  # 把测试集输入转换为张量
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 把测试集标签转换为张量并增加一维

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 构建最终模型训练用的数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 构建最终评估用的测试数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构建训练集数据加载器，批量 32 并打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 构建测试集数据加载器，批量 32 且不打乱

final_model = GRUWaveletModel(input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1).to(device)  # 实例化最终模型并移动到指定设备

print(final_model)  # 打印最终模型的结构信息

print("\n开始训练最终模型...")  # 提示开始训练最终模型
print(f"模型参数量: {sum(p.numel() for p in final_model.parameters()):,}")  # 打印最终模型的参数总数
print(f"模型大小: {get_model_size(final_model):.2f} MB")  # 打印最终模型占用的内存大小

# 记录训练开始时间
training_start_time = time.time()  # 记录训练开始的时间戳

train_losses, val_losses = train_model(final_model, train_loader, test_loader,  # 在全部训练数据上训练最终模型，并把测试集作为验证集监控
                                       epochs=200, lr=0.001, patience=10)

# 计算训练总时长
training_end_time = time.time()  # 记录训练结束的时间戳
total_training_time = training_end_time - training_start_time  # 用结束时间减去开始时间得到训练总耗时（秒）

print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练完成提示及最低验证损失
print(f"训练总时长: {format_time(total_training_time)}")  # 以易读格式打印训练总时长
print(f"训练总时长（秒）: {total_training_time:.2f}秒")  # 以秒为单位打印训练总时长

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建用于绘制最终训练损失曲线的画布
plt.plot(train_losses, label='Train Loss', linewidth=2)  # 绘制最终模型的训练损失曲线
plt.plot(val_losses, label='Validation Loss', linewidth=2)  # 绘制最终模型的验证损失曲线
plt.title('Model Training - Loss Curves')  # 设置图表标题
plt.xlabel('Epoch')  # 设置横轴标签为 epoch
plt.ylabel('Loss (MSE)')  # 设置纵轴标签为 MSE 损失
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.tight_layout()  # 调整图表布局
plt.savefig('training_loss_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把训练损失曲线保存到本地
print("Training loss curve saved: training_loss_GRUWavelet_pytorch.png")  # 打印训练曲线保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 提示开始在测试集上进行预测

# 直接使用y_test（已经是归一化后的值）
y_test_normalized = y_test  # 测试集标签本身已是归一化后的数值

# 反归一化得到真实值
y_test_rescaled = target_scaler.inverse_transform(y_test_normalized.reshape(-1, 1))  # 对测试集真实标签做反归一化，还原到原始量纲

print(f"\n测试集数据检查:")  # 打印测试集数据检查标题
print(f"X_test shape: {X_test.shape}")  # 打印测试输入的形状
print(f"y_test shape: {y_test.shape}")  # 打印测试标签的形状
print(f"y_test_rescaled shape: {y_test_rescaled.shape}")  # 打印反归一化后标签的形状

X_test_tensor = torch.FloatTensor(X_test).to(device)  # 把测试输入转换为张量并移动到指定设备

final_model.eval()  # 把最终模型切换为评估模式
with torch.no_grad():  # 关闭梯度计算以节省显存并加速预测
    y_pred = final_model(X_test_tensor).cpu().numpy()  # 对测试输入做前向预测，并把结果从 GPU 转回 numpy 数组

# 反归一化预测值
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 对模型的预测值做反归一化，还原到原始量纲

print(f"y_pred_rescaled shape: {y_pred_rescaled.shape}")  # 打印预测结果的形状

r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算测试集决定系数 R²
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算测试集均方误差 MSE
rmse_test = np.sqrt(mse_test)  # 对 MSE 开平方得到均方根误差 RMSE
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算测试集平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("测试集评估结果（GRU+小波+ECA）")  # 打印测试集评估结果标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印测试集 R² 指标
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印测试集 RMSE 指标
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印测试集 MAE 指标
print(f"{'=' * 60}")  # 打印分隔线

plt.figure(figsize=(15, 5))  # 创建测试集预测对比图的画布
# 确保长度匹配
y_test_rescaled_flat = y_test_rescaled.flatten()  # 把测试集真实值展平成一维数组
y_pred_rescaled_flat = y_pred_rescaled.flatten()  # 把测试集预测值展平成一维数组

# 从test_data_raw获取时间戳（已经是测试集部分）
test_timestamps = test_data_raw['timestamp'].values[time_step + pred_offset:]  # 截取与预测结果时间范围对应的测试集时间戳
test_timestamps = test_timestamps[:len(y_test_rescaled_flat)]  # 截断时间戳，使其数量与真实值数量一致

print(f"\n绘图数据检查:")  # 打印绘图数据检查标题
print(f"test_timestamps length: {len(test_timestamps)}")  # 打印时间戳数组的长度
print(f"y_test_rescaled_flat length: {len(y_test_rescaled_flat)}")  # 打印真实值数组的长度
print(f"y_pred_rescaled_flat length: {len(y_pred_rescaled_flat)}")  # 打印预测值数组的长度

plt.plot(test_timestamps, y_test_rescaled_flat, label='真实温度偏差', linewidth=2)  # 按时间戳绘制真实温度偏差曲线
plt.plot(test_timestamps, y_pred_rescaled_flat, label='GRU+小波+ECA预测', linewidth=2)  # 按时间戳绘制模型预测的温度偏差曲线
plt.xlabel('时间')  # 设置横轴标签为时间
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签为温度偏差（摄氏度）
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置图表标题并展示 R² 与 RMSE 指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.savefig('test_prediction_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把测试集预测对比图保存到本地
print("测试集预测结果已保存: test_prediction_GRUWavelet_pytorch.png")  # 打印测试集预测图保存提示
plt.close()  # 关闭画布释放内存

# ===================== 模型保存 =====================
torch.save({  # 把模型状态字典及配置信息打包保存到 .pth 文件
    'model_state_dict': final_model.state_dict(),
    'input_dim': input_dim,
    'time_step': time_step,
    'feature_names': feature_names,
    'wavelet': 'db4',
    'level': 3,
    'wavelet_window': time_step,  # 保存小波窗口大小
    'wavelet_target_col': '温度偏差',  # 保存小波分解的目标列
    'feature_scaler': feature_scaler,
    'target_scaler': target_scaler,
    'no_data_leakage': True,  # 标记为无数据泄漏版本
    'original_features': features,  # 保存原始特征列表
}, 'Model_GRUWavelet_no_leakage_pytorch.pth')
print("\nGRU模型已保存: Model_GRUCO2Wavelet_no_leakage_pytorch.pth")  # 打印模型保存完成的提示
# ===================== 新数据预测 =====================
print("\n开始处理新数据")  # 提示开始处理新数据集
# 使用多编码尝试读取新数据
new_data = None  # 初始化新数据变量为空
for enc in encodings:  # 依次尝试用各种编码读取新数据
    try:  # 捕获解码异常
        new_data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取新数据 CSV
        print(f"成功使用 {enc} 编码加载新数据")  # 打印成功读取新数据所用的编码
        break  # 读取成功后跳出循环
    except (UnicodeDecodeError, UnicodeError):  # 若解码失败则继续尝试下一种编码
        continue  # 继续尝试下一种编码

if new_data is None:  # 若所有编码都读取失败
    raise ValueError("无法使用任何已知编码读取新数据文件")  # 抛出异常终止程序，提示无法读取新数据

new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',  # 按索引乘 5 分钟、以指定起点生成新数据的时间戳列
                                       origin=pd.Timestamp('2024/4/23 14:55'))
# 对新数据进行滑动窗口小波分解（避免数据泄漏）
# 只对温度偏差进行小波分解，其他特征保持原样
print("\n对新数据进行滑动窗口小波分解（仅分解温度偏差）...")  # 提示开始对新数据做滑动窗口小波分解
new_wavelet_features = wavelet_decompose_sliding(new_data[features], target_col='温度偏差', wavelet='db4', level=3,  # 对新数据特征执行滑动窗口小波分解
                                                 window_size=time_step)
# 直接使用小波分解后的特征
new_combined_features = new_wavelet_features  # 新数据直接使用小波分解后的特征表
# 使用训练集的scaler进行归一化
new_combined_features = pd.DataFrame(  # 用训练集拟合出的 scaler 对新数据特征做归一化
    feature_scaler.transform(new_combined_features),
    columns=feature_names
)
# 保存原始目标值
y_new_original = new_data[target].values  # 保存新数据的原始目标值

X_new = create_sequences(new_combined_features, time_step)  # 为新数据生成时间窗口输入序列
y_new = y_new_original[time_step + pred_offset:]  # 取新数据中与输入序列对应的标签（偏移之后）
X_new = X_new[:len(y_new)]  # 截断输入序列，使其数量与标签数量一致

X_new_tensor = torch.FloatTensor(X_new).to(device)  # 把新数据输入转换为张量并移动到指定设备

# GRU预测
print("\nGRU+小波+ECA模型预测新数据...")  # 提示开始用训练好的模型预测新数据
final_model.eval()  # 把模型切换为评估模式
with torch.no_grad():  # 关闭梯度计算以加速新数据预测
    y_new_pred = final_model(X_new_tensor).cpu().numpy()  # 对新数据做前向预测并把结果转回 numpy 数组

y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 对新数据的预测值做反归一化
y_new_rescaled = y_new.reshape(-1, 1)  # 原始值  # 把新数据真实标签整理成二维列向量

r2_new = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据的决定系数 R²
mse_new = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据的均方误差 MSE
rmse_new = np.sqrt(mse_new)  # 对新数据 MSE 开平方得到 RMSE
mae_new = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据的平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印分隔线
print("新数据评估结果（GRU+小波+ECA）")  # 打印新数据评估结果标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R²:   {r2_new:.4f}")  # 打印新数据的 R² 指标
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据的 RMSE 指标
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据的 MAE 指标
print(f"{'=' * 60}")  # 打印分隔线

# 可视化新数据预测结果
timestamps = new_data['timestamp'][time_step + pred_offset:]  # 截取与新数据预测结果对应的时间戳
plt.figure(figsize=(15, 5))  # 创建新数据预测结果可视化的画布
# 确保长度匹配
y_new_rescaled_flat = y_new_rescaled.flatten()  # 把新数据真实值展平成一维数组
y_new_pred_rescaled_flat = y_new_pred_rescaled.flatten()  # 把新数据预测值展平成一维数组

plt.plot(timestamps[:len(y_new_rescaled_flat)], y_new_rescaled_flat,  # 按时间戳绘制新数据的真实温度偏差
         label='真实温度偏差', linewidth=2, alpha=0.8)
plt.plot(timestamps[:len(y_new_pred_rescaled_flat)], y_new_pred_rescaled_flat,  # 按时间戳绘制新数据的模型预测温度偏差
         label='GRU+小波+ECA预测', linewidth=2, alpha=0.8)
plt.xlabel('时间')  # 设置横轴标签为时间
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签为温度偏差（摄氏度）
plt.title(f'新数据集温度偏差结果 (R²: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置标题并展示新数据的 R² 与 RMSE 指标
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格线
plt.savefig('new_data_prediction_GRUWavelet_pytorch.png', dpi=150, bbox_inches='tight')  # 把新数据预测结果图保存到本地
print("\n新数据预测结果已保存: new_data_prediction_GRUWavelet_pytorch.png")  # 打印新数据预测图保存提示
plt.close()  # 关闭画布释放内存

print("\n========== 所有任务完成（GRU+小波+ECA模型） ==========")  # 打印所有任务完成的提示

