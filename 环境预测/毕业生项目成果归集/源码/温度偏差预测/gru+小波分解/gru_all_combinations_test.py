"""
本文件是一个模型实验脚本：在“温度偏差预测”任务中，系统性地测试
“GRU 神经网络 + 小波分解”在各种特征组合下的预测效果。

主要流程：
1. 读取温湿度 CSV 数据（自动尝试多种编码），构造时间戳并进行缺失值填充；
2. 用 pywt 对目标列“温度偏差”做滑动窗口小波分解，生成近似系数与细节系数特征；
3. 枚举所有可能的小波特征子集（共 16 种组合），对每种组合：
   - 做 MinMax 归一化与滑窗序列构造；
   - 进行 5 折交叉验证评估验证损失；
   - 用全部训练数据训练最终模型，并在测试集上计算 R2/RMSE/MAE；
4. 将所有组合结果保存为 CSV，并绘制箱线图、柱状图、散点图等可视化图表；
5. 输出按 R2 排序的最佳组合分组统计摘要。

脚本借助 itertools.combinations 穷举组合，目的是找出最优的小波特征子集。
"""

import pandas as pd  # 导入 pandas，用于表格数据的读取、处理与结果保存
import numpy as np  # 导入 numpy，用于数组运算与统计指标计算
import time  # 导入 time，用于统计模型训练耗时
from itertools import combinations  # 导入 combinations，用于生成小波特征的所有子集组合
import matplotlib  # 导入 matplotlib 基础模块，用于设置绘图后端
matplotlib.use('Agg')  # 设置 Agg 非交互式后端，使无图形界面环境下也能保存图片
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制各类统计图表
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler，用于特征归一化（缩放到 0-1）
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、均方误差、平均绝对误差三个评价指标
from sklearn.model_selection import KFold  # 导入 KFold，用于 5 折交叉验证划分数据
import torch  # 导入 PyTorch 主模块，用于张量运算与设备选择
import torch.nn as nn  # 导入神经网络模块，用于搭建 GRU 模型与损失函数
import torch.optim as optim  # 导入优化器模块，用于 Adam 优化器
from torch.utils.data import DataLoader, TensorDataset  # 导入 DataLoader/TensorDataset，用于构建批量数据加载器
import pywt  # 导入 PyWavelets，用于对时间序列做小波分解

# ===================== 环境配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有 GPU 则用 CUDA，否则回退到 CPU
print(f"使用设备: {device}")  # 打印实际使用的计算设备，便于确认是否在 GPU 上运行
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，避免图表中文显示为方块
plt.rcParams['axes.unicode_minus'] = False  # 关闭负号的 Unicode 显示，避免坐标轴负号乱码

# ===================== 数据加载 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 依次尝试的 CSV 文件编码列表
data = None  # 初始化数据变量为空，用于记录是否成功读取
for enc in encodings:  # 遍历每种候选编码逐一尝试读取
    try:  # 进入 try 块，捕获编码解码异常
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取温湿度数据 CSV
        print(f"成功使用 {enc} 编码加载数据")  # 打印成功读取所用的编码
        break  # 读取成功后跳出循环，不再尝试后续编码
    except (UnicodeDecodeError, UnicodeError):  # 捕获编码不匹配导致的解码错误
        continue  # 当前编码失败，继续尝试下一种编码

if data is None:  # 若所有编码都失败，data 仍为空
    raise ValueError("无法读取数据文件")  # 抛出异常，终止程序并提示数据读取失败

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按 5 分钟间隔构造与数据等长的时间戳列
features = ['日龄','温度偏差','外界温度','CO2','目标温度','相对湿度']  # 预先定义的候选输入特征名称列表
target = '温度偏差'  # 指定预测目标列为“温度偏差”
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的候选特征，避免列不存在报错

# 缺失值处理
continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度','相对湿度']  # 定义需要进行插值的连续型变量列表
discrete_vars = ['日龄']  # 定义只需前后填充的离散型变量列表

for col in continuous_vars:  # 遍历每个连续型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且含有缺失值才处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').ffill().bfill()  # 线性插值后用前后值填充首尾残留缺失

for col in discrete_vars:  # 遍历每个离散型变量
    if col in data.columns and data[col].isna().sum() > 0:  # 仅当该列存在且含有缺失值才处理
        data[col] = data[col].ffill().bfill()  # 离散值不做插值，直接用前值/后值填充缺失

data = data.dropna(subset=features + [target])  # 丢弃特征列或目标列中仍有缺失的行，保证建模数据完整
print(f"数据形状: {data.shape}")  # 打印清洗后数据的行列数

# ===================== 小波分解函数 =====================
def wavelet_decompose_sliding(data, target_col='温度偏差', wavelet='db4', level=3, window_size=None):  # 定义滑动窗口小波分解函数
    """对目标列进行滑动窗口小波分解，提取近似系数和多层细节系数作为新特征。

    参数：
        data: 输入 DataFrame，通常为已筛选出的特征列。
        target_col: 需要做小波分解的目标列名，默认“温度偏差”。
        wavelet: 使用的小波基函数，默认 'db4'（Daubechies 4）。
        level: 小波分解层数，默认 3。
        window_size: 滑动窗口长度；为 None 时按 2**(level+6) 与样本数取较小值。
    返回：
        DataFrame，非目标列原样保留；目标列被替换为近似系数与各层细节系数特征。
    """
    if window_size is None:  # 未显式指定窗口长度时，按分解层数自动推算默认值
        window_size = min(2 ** (level + 6), len(data))  # 窗口取 2^(level+6) 与数据长度中较小者，避免超过样本数
    
    wavelet_features = pd.DataFrame(index=data.index)  # 创建与原数据同索引的空表，用于存放小波分解结果
    
    for col in data.columns:  # 遍历输入数据的每一列
        if col == 'timestamp':  # 时间戳列不参与小波分解
            continue  # 跳过该列
        
        signal = data[col].values  # 取出当前列的值作为一维时间序列信号
        
        if col == target_col:  # 仅目标列需要展开为多尺度小波特征
            approx_list = []  # 收集每个时间点的近似系数（低频分量）
            detail_lists = [[] for _ in range(level)]  # 为每一分解层各建一个列表，收集细节系数（高频分量）
            
            for i in range(len(signal)):  # 逐个时间点向前滑动窗口
                start_idx = max(0, i - window_size + 1)  # 计算窗口起点并保证不小于 0
                window_signal = signal[start_idx:i+1]  # 截取以当前点为结尾的窗口信号
                
                if len(window_signal) < window_size:  # 序列开头处窗口长度不足
                    pad_width = window_size - len(window_signal)  # 计算左侧需补齐的元素个数
                    window_signal = np.pad(window_signal, (pad_width, 0), mode='edge')  # 用边缘值在左侧补齐至窗口长度
                
                try:  # 进入 try 块，捕获小波分解异常
                    coeffs = pywt.wavedec(window_signal, wavelet, level=level)  # 对窗口信号做 level 层离散小波分解
                    approx_list.append(coeffs[0][-1])  # 取近似系数的末值作为当前点低频特征
                    for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                        detail_lists[j].append(detail[-1] if len(detail) > 0 else 0.0)  # 取该层细节系数末值，空则用 0 代替
                except:  # 捕获分解异常并回退为原始值
                    approx_list.append(window_signal[-1])  # 分解失败时退化为取窗口最后一个原始值
                    for j in range(level):  # 遍历所有分解层
                        detail_lists[j].append(0.0)  # 分解失败时该层细节特征补 0
            
            wavelet_features[f'{col}_approx'] = approx_list  # 写入近似系数特征列
            for j in range(level):  # 遍历所有分解层
                wavelet_features[f'{col}_detail{j+1}'] = detail_lists[j]  # 写入对应层的细节系数特征列
        else:  # 非目标列分支：原样保留该列
            wavelet_features[col] = signal  # 非目标列不做分解，原样复制到结果中
    
    return wavelet_features  # 返回组装好的小波特征 DataFrame

# ===================== GRU模型 =====================
class GRUModel(nn.Module):  # 定义基于 GRU 的回归模型类
    """基于 GRU 的回归模型：用最后一个时间步的隐状态预测单一目标值。

    参数：
        input_dim: 输入特征维度。
        time_step: 输入序列的时间步长（仅用于接口说明，网络本身自适应序列长度）。
        gru_hidden: GRU 隐藏层单元数，默认 128。
        gru_layers: GRU 堆叠层数，默认 1。
        dropout: 多层 GRU 时的 dropout 比例，默认 0.1。
    """

    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.1):  # 初始化 GRU 层与全连接输出层
        """初始化 GRU 层与全连接输出层。"""
        super(GRUModel, self).__init__()  # 调用父类 nn.Module 的初始化方法
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,  # 定义 GRU 层，具体参数见下一行
                         dropout=dropout if gru_layers > 1 else 0.0, batch_first=True)  # 定义 GRU：单层时不使用 dropout，输入按 batch 优先排列
        self.fc = nn.Linear(gru_hidden, 1)  # 全连接层把隐藏状态映射为 1 个预测值

    def forward(self, x):  # 定义前向传播方法
        """前向传播：输入形状 (batch, time_step, input_dim)，返回形状 (batch, 1) 的预测。"""
        out, _ = self.gru(x)  # 将序列送入 GRU，得到各时间步输出（忽略最终隐状态）
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出接全连接层，得到预测值

class EarlyStopping:  # 定义早停控制器类
    """早停机制：验证损失长期不下降时终止训练，并保留最佳模型参数。

    参数：
        patience: 允许验证损失不改善的最大轮数，默认 10。
    """

    def __init__(self, patience=10):  # 初始化早停的计数器与最佳参数
        """初始化计数器、最佳损失、停止标志与最佳模型权重。"""
        self.patience = patience  # 记录容忍轮数
        self.counter = 0  # 连续未改善的轮数计数
        self.best_loss = None  # 记录目前最小的验证损失
        self.early_stop = False  # 是否满足早停条件的标志
        self.best_model = None  # 保存验证损失最小时的模型参数副本

    def __call__(self, val_loss, model):  # 定义按验证损失更新早停状态的回调
        """根据当前验证损失更新最佳模型并判断是否触发早停。"""
        if self.best_loss is None:  # 首次调用，尚无最佳损失
            self.best_loss = val_loss  # 将当前验证损失设为初始最佳
            self.best_model = model.state_dict().copy()  # 保存当前模型参数副本
        elif val_loss > self.best_loss:  # 验证损失变差（未改善）
            self.counter += 1  # 未改善轮数加一
            if self.counter >= self.patience:  # 达到容忍上限
                self.early_stop = True  # 触发早停标志
        else:  # 验证损失持平或更优
            self.best_loss = val_loss  # 更新最佳损失
            self.best_model = model.state_dict().copy()  # 更新最佳模型参数
            self.counter = 0  # 重置未改善计数

def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=10):  # 定义模型训练与验证函数
    """训练模型并在每个 epoch 用验证集评估，返回训练/验证损失曲线。

    参数：
        model: 待训练的 GRU 模型。
        train_loader / val_loader: 训练集与验证集的数据加载器。
        epochs: 最大训练轮数，默认 200。
        lr: Adam 优化器学习率，默认 0.001。
        patience: 早停容忍轮数，默认 10。
    返回：
        (train_losses, val_losses) 两个列表，分别记录每轮平均训练损失与验证损失。
    """
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器更新模型参数
    early_stopping = EarlyStopping(patience=patience)  # 创建早停控制器
    train_losses, val_losses = [], []  # 初始化训练/验证损失记录列表

    for epoch in range(epochs):  # 按轮次进行训练
        model.train()  # 切换模型为训练模式（启用 dropout 等）
        train_loss = 0.0  # 累加本轮训练损失
        for batch_x, batch_y in train_loader:  # 遍历训练集的每个批次
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 将批次数据移动到指定设备
            optimizer.zero_grad()  # 清空上一轮梯度
            outputs = model(batch_x)  # 前向传播得到预测
            loss = criterion(outputs, batch_y)  # 计算预测与真值的均方误差
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新参数
            train_loss += loss.item()  # 累加当前批次的损失值
        train_loss /= len(train_loader)  # 求本轮平均训练损失
        train_losses.append(train_loss)  # 记录本轮训练损失

        model.eval()  # 切换为评估模式（关闭 dropout）
        val_loss = 0.0  # 累加本轮验证损失
        with torch.no_grad():  # 验证阶段不计算梯度，节省显存
            for batch_x, batch_y in val_loader:  # 遍历验证集各批次
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 数据移动到指定设备
                outputs = model(batch_x)  # 前向传播得到验证预测
                val_loss += criterion(outputs, batch_y).item()  # 累加验证损失
        val_loss /= len(val_loader)  # 求本轮平均验证损失
        val_losses.append(val_loss)  # 记录本轮验证损失

        early_stopping(val_loss, model)  # 调用早停控制器更新最佳模型
        if early_stopping.early_stop:  # 若触发早停
            model.load_state_dict(early_stopping.best_model)  # 恢复验证损失最小时的模型参数
            break  # 跳出训练循环

    if early_stopping.best_model is not None:  # 训练结束后若保存过最佳模型
        model.load_state_dict(early_stopping.best_model)  # 最终装入最佳参数，保证输出模型最优
    return train_losses, val_losses  # 返回两路损失曲线

def create_sequences(data, time_step, pred_offset):  # 定义滑动窗口序列构造函数
    """把二维特征表转换为监督学习所需的滑动窗口三维序列。

    参数：
        data: 归一化后的特征 DataFrame。
        time_step: 每个样本包含的历史时间步数。
        pred_offset: 预测偏移，决定标签相对窗口的起始位置。
    返回：
        ndarray，形状为 (样本数, time_step, 特征数)。
    """
    sequences = []  # 用于收集每个滑动窗口样本
    for i in range(len(data) - time_step - pred_offset):  # 滑动窗口起点，保证窗口与偏移都不越界
        sequences.append(data.iloc[i:i + time_step].values)  # 取出长度为 time_step 的连续片段作为一个样本
    return np.array(sequences)  # 转换为 numpy 数组返回

# ===================== 主测试流程 =====================
time_step = 36  # 定义输入序列的历史时间步长为 36（约 3 小时，按 5 分钟采样）
pred_offset = 6  # 定义预测偏移为 6（标签相对窗口末端向后 6 个采样点）

print(f"\n{'='*80}")  # 打印分隔线，标识测试开始
print("GRU + 小波分解 - 所有特征组合测试")  # 打印测试主题标题
print(f"{'='*80}")  # 再打印一条分隔线

# 划分数据集
split_index = int(len(data) * 0.8)  # 取前 80% 样本作为训练集切分点
train_data_raw = data.iloc[:split_index].copy()  # 切分出训练集（原始值，不做小波分解前）
test_data_raw = data.iloc[split_index:].copy()  # 切分出测试集（原始值）

print(f"训练集: {len(train_data_raw)}, 测试集: {len(test_data_raw)}")  # 打印训练集与测试集样本数

# 小波分解
print("\n进行小波分解...")  # 提示即将对训练/测试数据做小波分解
train_wavelet = wavelet_decompose_sliding(train_data_raw[features], target_col='温度偏差',  # 对训练集特征做小波分解（参数续见下一行）
                                         wavelet='db4', level=3, window_size=time_step)  # 对训练集特征做 3 层 db4 小波分解，窗口取时间步长
test_wavelet = wavelet_decompose_sliding(test_data_raw[features], target_col='温度偏差',  # 对测试集特征做小波分解（参数续见下一行）
                                        wavelet='db4', level=3, window_size=time_step)  # 对测试集特征做同样参数的小波分解

# 定义特征组
other_features = [f for f in features if f != '温度偏差']  # 非目标的原始特征集合
wavelet_features = ['温度偏差_approx', '温度偏差_detail1', '温度偏差_detail2', '温度偏差_detail3']  # 小波分解产生的 4 个候选特征名

print(f"\n其他特征: {other_features}")  # 打印参与组合的原始特征
print(f"小波特征: {wavelet_features}")  # 打印待枚举的小波特征

# 生成所有可能的小波特征组合（从0个到4个）
all_combinations = []  # 初始化组合列表，存放每个待测试方案的配置

# 组合0: 仅原始特征
all_combinations.append({  # 添加基线组合（仅原始特征）的配置
    'name': '仅原始特征',  # 组合名称
    'wavelet_subset': [],  # 空列表表示不使用任何小波特征
    'description': '基线模型（无小波）'  # 说明这是无小波的基线
})

# 组合1-15: 小波特征的所有子集
for r in range(1, len(wavelet_features) + 1):  # 枚举选取特征个数 r（1 到 4）
    for combo in combinations(wavelet_features, r):  # 生成从 4 个小波特征中取 r 个的所有组合
        all_combinations.append({  # 添加一个含小波特征组合的配置
            'name': f"原始+{list(combo)}",  # 以所选小波特征列表命名组合
            'wavelet_subset': list(combo),  # 记录该组合选用的小波特征
            'description': f"添加{len(combo)}个小波特征"  # 描述添加的小波特征数量
        })

print(f"\n总共测试 {len(all_combinations)} 种组合")  # 打印组合总数（1+4+6+4+1=16）
print(f"  - 0个小波特征: 1种")  # 说明选 0 个小波特征的组合数
print(f"  - 1个小波特征: 4种")  # 说明选 1 个小波特征的组合数
print(f"  - 2个小波特征: 6种")  # 说明选 2 个小波特征的组合数
print(f"  - 3个小波特征: 4种")  # 说明选 3 个小波特征的组合数
print(f"  - 4个小波特征: 1种")  # 说明选 4 个小波特征的组合数

# 存储所有结果
all_results = []  # 初始化结果列表，收集每个组合的评估指标

print(f"\n{'='*80}")  # 打印分隔线
print("开始测试所有组合（含5折交叉验证）...")  # 提示进入批量测试阶段
print(f"{'='*80}\n")  # 打印分隔线并换行

for idx, combo in enumerate(all_combinations, 1):  # 遍历每一种特征组合，idx 从 1 开始计数
    print(f"\n[{idx}/{len(all_combinations)}] 测试: {combo['name']}")  # 打印当前正在测试的组合进度与名称
    
    # 选择特征
    if len(combo['wavelet_subset']) == 0:  # 若该组合不含任何小波特征
        # 仅原始特征
        selected_features = other_features + ['温度偏差']  # 基线组合只使用原始特征（含目标列本身）
        train_features_selected = train_data_raw[selected_features]  # 从原始训练数据中取出所选特征列
        test_features_selected = test_data_raw[selected_features]  # 从原始测试数据中取出所选特征列
    else:  # 否则表示该组合包含小波特征
        # 原始特征 + 选定的小波特征
        selected_features = other_features + combo['wavelet_subset']  # 所选特征为原始特征加上选定的小波特征
        train_features_selected = train_wavelet[selected_features]  # 从小波分解后的训练数据中取出所选特征
        test_features_selected = test_wavelet[selected_features]  # 从小波分解后的测试数据中取出所选特征
    
    print(f"  特征数: {len(selected_features)}")  # 打印当前组合的特征总数
    
    try:  # 进入 try 块，捕获该组合测试过程中的异常
        # 归一化
        feature_scaler = MinMaxScaler()  # 创建特征归一化器
        target_scaler = MinMaxScaler()  # 创建目标值归一化器
        
        feature_scaler.fit(train_features_selected)  # 用训练集特征拟合归一化器
        target_scaler.fit(train_data_raw[[target]])  # 用训练集目标值拟合归一化器
        
        train_features_norm = pd.DataFrame(  # 将训练特征归一化后重建为 DataFrame
            feature_scaler.transform(train_features_selected),  # 对训练特征执行归一化变换
            columns=selected_features  # 指定新 DataFrame 的列名
        )
        test_features_norm = pd.DataFrame(  # 将测试特征归一化后重建为 DataFrame
            feature_scaler.transform(test_features_selected),  # 对测试特征执行归一化变换
            columns=selected_features  # 指定新 DataFrame 的列名
        )
        
        train_target_norm = target_scaler.transform(train_data_raw[[target]]).flatten()  # 归一化训练目标并展平为一维数组
        test_target_norm = target_scaler.transform(test_data_raw[[target]]).flatten()  # 归一化测试目标并展平为一维数组
        
        # 创建序列
        X_train = create_sequences(train_features_norm, time_step, pred_offset)  # 由归一化训练特征构造滑动窗口序列
        y_train = train_target_norm[time_step + pred_offset:][:len(X_train)]  # 按预测偏移对齐训练标签，使长度与样本数一致
        
        X_test = create_sequences(test_features_norm, time_step, pred_offset)  # 由归一化测试特征构造滑动窗口序列
        y_test = test_target_norm[time_step + pred_offset:][:len(X_test)]  # 按预测偏移对齐测试标签
        
        # ===================== 5折交叉验证 =====================
        print(f"  开始5折交叉验证...")  # 提示即将开始 5 折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 创建 5 折交叉验证划分器，打乱数据并固定随机种子
        cv_val_losses = []  # 收集各折的最佳验证损失
        cv_training_times = []  # 收集各折的训练耗时
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):  # 遍历 KFold 产生的每一折训练/验证索引
            X_train_fold = X_train[train_idx]  # 取当前折的训练特征
            y_train_fold = y_train[train_idx]  # 取当前折的训练标签
            X_val_fold = X_train[val_idx]  # 取当前折的验证特征
            y_val_fold = y_train[val_idx]  # 取当前折的验证标签
            
            # 转换为Tensor
            X_train_fold_tensor = torch.FloatTensor(X_train_fold)  # 训练特征转换为浮点张量
            y_train_fold_tensor = torch.FloatTensor(y_train_fold).unsqueeze(1)  # 训练标签转换为浮点张量并增加一维特征轴
            X_val_fold_tensor = torch.FloatTensor(X_val_fold)  # 验证特征转换为浮点张量
            y_val_fold_tensor = torch.FloatTensor(y_val_fold).unsqueeze(1)  # 验证标签转换为浮点张量并增加一维特征轴
            
            train_fold_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)  # 将训练张量封装为数据集
            val_fold_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)  # 将验证张量封装为数据集
            train_fold_loader = DataLoader(train_fold_dataset, batch_size=32, shuffle=True)  # 构建训练集加载器，批次大小 32 且打乱顺序
            val_fold_loader = DataLoader(val_fold_dataset, batch_size=32, shuffle=False)  # 构建验证集加载器，批次大小 32 且不打乱
            
            # 构建模型
            model_fold = GRUModel(len(selected_features), time_step, gru_hidden=128, gru_layers=1, dropout=0.1).to(device)  # 按当前特征数构建 GRU 模型并移至指定设备
            
            # 训练
            fold_start_time = time.time()  # 记录当前折训练开始时间
            _, fold_val_losses = train_model(model_fold, train_fold_loader, val_fold_loader,  # 训练当前折模型，丢弃训练损失只保留验证损失
                                            epochs=200, lr=0.001, patience=10)  # 训练参数：200 轮、学习率 0.001、早停容忍 10 轮
            fold_training_time = time.time() - fold_start_time  # 计算当前折训练耗时
            
            best_val_loss = min(fold_val_losses)  # 取当前折最小的验证损失作为该折最佳成绩
            cv_val_losses.append(best_val_loss)  # 记录该折最佳验证损失
            cv_training_times.append(fold_training_time)  # 记录该折训练耗时
        
        # 计算交叉验证统计
        cv_mean_loss = np.mean(cv_val_losses)  # 计算交叉验证平均验证损失
        cv_std_loss = np.std(cv_val_losses)  # 计算交叉验证验证损失的标准差
        cv_total_time = np.sum(cv_training_times)  # 汇总 5 折训练总耗时
        
        print(f"  5折交叉验证完成 - 平均验证损失: {cv_mean_loss:.6f} ± {cv_std_loss:.6f}")  # 打印交叉验证平均损失与波动情况
        
        # ===================== 训练最终模型（使用全部训练数据） =====================
        print(f"  训练最终模型...")  # 提示开始训练最终模型
        X_train_tensor = torch.FloatTensor(X_train)  # 全部训练特征转换为张量
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # 全部训练标签转换为张量并增加特征轴
        X_test_tensor = torch.FloatTensor(X_test)  # 测试特征转换为张量
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)  # 测试标签转换为张量并增加特征轴
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 将最终训练集张量封装为数据集
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 将最终测试集张量封装为数据集
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 构建最终模型的训练数据加载器
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 构建最终模型的测试数据加载器
        
        # 构建并训练最终模型
        final_model = GRUModel(len(selected_features), time_step, gru_hidden=128, gru_layers=1, dropout=0.1).to(device)  # 用全部训练特征数构建最终 GRU 模型并移至设备
        
        final_start_time = time.time()  # 记录最终模型训练开始时间
        train_losses, val_losses = train_model(final_model, train_loader, test_loader,  # 训练最终模型，同时记录训练与验证损失
                                              epochs=200, lr=0.001, patience=10)  # 训练参数：200 轮、学习率 0.001、早停容忍 10 轮
        final_training_time = time.time() - final_start_time  # 计算最终模型训练耗时
        
        total_training_time = cv_total_time + final_training_time  # 总训练时长等于交叉验证耗时加最终模型耗时
        
        # ===================== 测试集预测 =====================
        final_model.eval()  # 将最终模型切换为评估模式
        with torch.no_grad():  # 关闭梯度计算以节省显存并加速预测
            y_pred = final_model(X_test_tensor.to(device)).cpu().numpy()  # 用测试特征预测，结果转回 numpy 数组
        
        # 反归一化
        y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # 将测试真值反归一化，还原到原始量纲
        y_pred_rescaled = target_scaler.inverse_transform(y_pred).flatten()  # 将预测值反归一化，还原到原始量纲
        
        # 计算指标
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))  # 计算均方根误差 RMSE
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE
        
        print(f"  测试集结果 - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")  # 打印测试集上的三项评价指标
        print(f"  总训练时长: {total_training_time:.1f}s (交叉验证: {cv_total_time:.1f}s + 最终模型: {final_training_time:.1f}s)")  # 打印总训练时长及其构成
        
        # 保存结果
        all_results.append({  # 开始记录当前组合的成功结果
            'combo_id': idx,  # 组合编号
            'name': combo['name'],  # 组合名称
            'description': combo['description'],  # 组合描述
            'num_wavelet_features': len(combo['wavelet_subset']),  # 使用的小波特征个数
            'wavelet_features': ','.join(combo['wavelet_subset']) if combo['wavelet_subset'] else 'None',  # 使用的小波特征名，逗号连接，无则记为 None
            'total_features': len(selected_features),  # 总特征数
            'cv_mean_val_loss': cv_mean_loss,  # 交叉验证平均验证损失
            'cv_std_val_loss': cv_std_loss,  # 交叉验证验证损失标准差
            'cv_training_time': cv_total_time,  # 交叉验证阶段耗时
            'final_training_time': final_training_time,  # 最终模型训练耗时
            'total_training_time': total_training_time,  # 总训练时长
            'r2': r2,  # R2 指标
            'rmse': rmse,  # RMSE 指标
            'mae': mae,  # MAE 指标
            'num_epochs': len(train_losses),  # 实际训练轮数
            'final_train_loss': train_losses[-1],  # 最后一轮的训练损失
            'final_val_loss': val_losses[-1],  # 最后一轮的验证损失
            'status': 'success'  # 状态标记为成功
        })
        
    except Exception as e:  # 捕获当前组合测试过程中抛出的异常
        print(f"  错误: {str(e)}")  # 打印错误摘要
        import traceback  # 导入 traceback 模块以输出详细堆栈
        traceback.print_exc()  # 打印完整异常堆栈便于排查
        all_results.append({  # 异常时同样记录一条结果，便于统计失败组合
            'combo_id': idx,  # 组合编号
            'name': combo['name'],  # 组合名称
            'description': combo['description'],  # 组合描述
            'num_wavelet_features': len(combo['wavelet_subset']),  # 使用的小波特征个数
            'wavelet_features': ','.join(combo['wavelet_subset']) if combo['wavelet_subset'] else 'None',  # 使用的小波特征名
            'total_features': len(selected_features),  # 总特征数
            'cv_mean_val_loss': np.nan,  # 交叉验证平均损失以 NaN 占位表示失败
            'cv_std_val_loss': np.nan,  # 交叉验证损失标准差以 NaN 占位
            'cv_training_time': np.nan,  # 交叉验证耗时以 NaN 占位
            'final_training_time': np.nan,  # 最终模型训练耗时以 NaN 占位
            'total_training_time': np.nan,  # 总训练时长以 NaN 占位
            'r2': np.nan,  # R2 以 NaN 占位
            'rmse': np.nan,  # RMSE 以 NaN 占位
            'mae': np.nan,  # MAE 以 NaN 占位
            'num_epochs': np.nan,  # 训练轮数以 NaN 占位
            'final_train_loss': np.nan,  # 最终训练损失以 NaN 占位
            'final_val_loss': np.nan,  # 最终验证损失以 NaN 占位
            'status': f'failed: {str(e)}'  # 状态标记为失败并附带错误原因
        })

# ===================== 保存结果 =====================
print(f"\n{'='*80}")  # 打印分隔线
print("保存结果...")  # 提示正在保存结果
print(f"{'='*80}")  # 打印分隔线

results_df = pd.DataFrame(all_results)  # 将结果列表转换为 DataFrame 便于保存与排序
results_df.to_csv('all_wavelet_combinations_results.csv', index=False, encoding='utf-8-sig')  # 把完整结果写入 CSV，使用带 BOM 的 UTF-8 以便 Excel 打开
print("完整结果已保存: all_wavelet_combinations_results.csv")  # 提示完整结果已保存

# 只显示成功的结果
success_results = results_df[results_df['status'] == 'success'].copy()  # 筛选出测试成功的组合并复制一份
success_results = success_results.sort_values('r2', ascending=False)  # 按 R2 从高到低排序，便于找出最佳组合

print(f"\n成功测试: {len(success_results)}/{len(all_results)} 种组合")  # 打印成功与失败组合数量
print(f"\n前10名最佳组合 (按R2排序):")  # 提示将展示前 10 名组合
print(success_results[['combo_id', 'num_wavelet_features', 'r2', 'rmse', 'mae', 'cv_mean_val_loss', 'cv_std_val_loss', 'total_training_time']].head(10).to_string(index=False))  # 打印前 10 名组合的关键指标表格

# 保存排序后的结果
success_results.to_csv('all_wavelet_combinations_results_sorted.csv', index=False, encoding='utf-8-sig')  # 将排序后的成功结果另存为 CSV
print("\n排序结果已保存: all_wavelet_combinations_results_sorted.csv")  # 提示排序结果已保存

# ===================== 可视化 =====================
print("\n生成可视化图表...")  # 提示开始生成可视化图表

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 创建 2 行 3 列的子图画布，尺寸 18x10 英寸

# 按小波特征数量分组
grouped = success_results.groupby('num_wavelet_features')  # 按小波特征个数对成功结果分组，便于分组对比

# 1. R2 箱线图
ax = axes[0, 0]  # 选中左上角子图绘制 R2 箱线图
data_to_plot = [grouped.get_group(i)['r2'].values for i in sorted(grouped.groups.keys())]  # 按分组提取各组的 R2 数值序列
ax.boxplot(data_to_plot, labels=[f'{i}个' for i in sorted(grouped.groups.keys())])  # 绘制 R2 箱线图，横轴标签为小波特征个数
ax.set_xlabel('小波特征数量')  # 设置横轴标签
ax.set_ylabel('R2')  # 设置纵轴标签
ax.set_title('R2分布（按小波特征数量）')  # 设置子图标题
ax.grid(True, alpha=0.3)  # 添加半透明网格线

# 2. RMSE 箱线图
ax = axes[0, 1]  # 选中上中位置子图绘制 RMSE 箱线图
data_to_plot = [grouped.get_group(i)['rmse'].values for i in sorted(grouped.groups.keys())]  # 按分组提取各组的 RMSE 数值序列
ax.boxplot(data_to_plot, labels=[f'{i}个' for i in sorted(grouped.groups.keys())])  # 绘制 RMSE 箱线图
ax.set_xlabel('小波特征数量')  # 设置横轴标签
ax.set_ylabel('RMSE')  # 设置纵轴标签
ax.set_title('RMSE分布（按小波特征数量）')  # 设置子图标题
ax.grid(True, alpha=0.3)  # 添加半透明网格线

# 3. MAE 箱线图
ax = axes[0, 2]  # 选中右上角子图绘制 MAE 箱线图
data_to_plot = [grouped.get_group(i)['mae'].values for i in sorted(grouped.groups.keys())]  # 按分组提取各组的 MAE 数值序列
ax.boxplot(data_to_plot, labels=[f'{i}个' for i in sorted(grouped.groups.keys())])  # 绘制 MAE 箱线图
ax.set_xlabel('小波特征数量')  # 设置横轴标签
ax.set_ylabel('MAE')  # 设置纵轴标签
ax.set_title('MAE分布（按小波特征数量）')  # 设置子图标题
ax.grid(True, alpha=0.3)  # 添加半透明网格线

# 4. 训练时长对比
ax = axes[1, 0]  # 选中左下角子图绘制训练时长对比
avg_time = grouped['total_training_time'].mean()  # 按小波特征个数计算平均总训练时长
ax.bar(range(len(avg_time)), avg_time.values, alpha=0.7, color='steelblue')  # 绘制各分组平均训练时长柱状图
ax.set_xticks(range(len(avg_time)))  # 设置柱子的横轴刻度位置
ax.set_xticklabels([f'{i}个' for i in avg_time.index])  # 设置横轴刻度标签为小波特征个数
ax.set_xlabel('小波特征数量')  # 设置横轴标签
ax.set_ylabel('平均总训练时长 (秒)')  # 设置纵轴标签，单位为秒
ax.set_title('平均训练时长对比（含5折交叉验证）')  # 设置子图标题，说明含交叉验证
ax.grid(True, alpha=0.3, axis='y')  # 仅在纵轴方向添加网格线

# 5. Top10组合对比
ax = axes[1, 1]  # 选中下中位置子图绘制 Top10 组合对比
top10 = success_results.head(10)  # 取 R2 最高的前 10 个组合
x_pos = range(len(top10))  # 为每个组合生成一个水平位置坐标
ax.barh(x_pos, top10['r2'].values, alpha=0.7, color='green')  # 绘制水平柱状图，显示各组合 R2
ax.set_yticks(x_pos)  # 设置纵轴刻度位置
ax.set_yticklabels([f"组合{int(id)}" for id in top10['combo_id'].values], fontsize=8)  # 用组合编号作为纵轴刻度标签，字号 8
ax.set_xlabel('R2')  # 设置横轴标签
ax.set_title('Top 10 组合 R2对比')  # 设置子图标题
ax.grid(True, alpha=0.3, axis='x')  # 仅在横轴方向添加网格线
ax.invert_yaxis()  # 反转纵轴，使最佳组合显示在最上方

# 6. 特征数量 vs R2 散点图
ax = axes[1, 2]  # 选中右下角子图绘制总特征数与 R2 的散点图
colors = plt.cm.viridis(success_results['num_wavelet_features'] / success_results['num_wavelet_features'].max())  # 按小波特征数量计算 viridis 颜色映射值
scatter = ax.scatter(success_results['total_features'], success_results['r2'],   # 绘制散点图，横轴总特征数、纵轴 R2
                    c=success_results['num_wavelet_features'], cmap='viridis',   # 指定颜色按小波特征数映射，采用 viridis 配色
                    alpha=0.6, s=100)  # 设置点的透明度与大小
ax.set_xlabel('总特征数')  # 设置横轴标签
ax.set_ylabel('R2')  # 设置纵轴标签
ax.set_title('总特征数 vs R2')  # 设置子图标题
ax.grid(True, alpha=0.3)  # 添加半透明网格线
cbar = plt.colorbar(scatter, ax=ax)  # 为散点图添加颜色条
cbar.set_label('小波特征数')  # 给颜色条标注含义为小波特征数

plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.savefig('all_combinations_visualization.png', dpi=600, bbox_inches='tight')  # 以 600 dpi 保存可视化图片并紧贴内容边界
print("可视化图表已保存: all_combinations_visualization.png")  # 提示图表已保存

# 生成详细统计报告
print(f"\n{'='*80}")  # 打印分隔线
print("统计摘要")  # 打印统计摘要标题
print(f"{'='*80}")  # 打印分隔线
for num_wavelet in sorted(grouped.groups.keys()):  # 按小波特征个数分组遍历，逐组输出统计
    group_data = grouped.get_group(num_wavelet)  # 取出当前分组的所有组合结果
    print(f"\n{num_wavelet}个小波特征 ({len(group_data)}种组合):")  # 打印该组的组合数量
    print(f"  交叉验证损失 - 平均: {group_data['cv_mean_val_loss'].mean():.6f} ± {group_data['cv_std_val_loss'].mean():.6f}")  # 打印该组交叉验证损失的平均与波动
    print(f"  R2    - 平均: {group_data['r2'].mean():.4f}, 最佳: {group_data['r2'].max():.4f}, 最差: {group_data['r2'].min():.4f}")  # 打印该组 R2 的平均、最佳与最差
    print(f"  RMSE  - 平均: {group_data['rmse'].mean():.4f}, 最佳: {group_data['rmse'].min():.4f}, 最差: {group_data['rmse'].max():.4f}")  # 打印该组 RMSE 的平均、最佳与最差
    print(f"  MAE   - 平均: {group_data['mae'].mean():.4f}, 最佳: {group_data['mae'].min():.4f}, 最差: {group_data['mae'].max():.4f}")  # 打印该组 MAE 的平均、最佳与最差
    print(f"  总训练时长 - 平均: {group_data['total_training_time'].mean():.1f}秒 (交叉验证: {group_data['cv_training_time'].mean():.1f}秒 + 最终模型: {group_data['final_training_time'].mean():.1f}秒)")  # 打印该组总训练时长及交叉验证与最终模型耗时构成

print(f"\n{'='*80}")  # 打印分隔线
print("测试完成！")  # 打印测试完成提示
print(f"{'='*80}")  # 打印分隔线结束整体输出

