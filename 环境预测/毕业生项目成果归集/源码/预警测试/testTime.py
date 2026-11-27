# -*- coding: utf-8 -*-
"""
GRU + db4 小波 + ECA 模型的在线计算延时测试脚本

本脚本用于测量模型在单步在线滚动预测场景下的计算延时：从一个新的 5 分钟聚合样本
形成后，依次完成 db4 小波特征提取、特征归一化、输入窗口构造、模型推理、反归一化
以及 THI 预警等级判定的总耗时。脚本会先加载训练好的检查点并重建与之完全一致的模型
结构，读取测试数据并做与训练一致的缺失值处理；随后进行若干次预热，再对采样得到的
若干样本逐一高精度计时，最后输出平均/中位数/95%分位/最大/最小延时，并把每次预测的
明细结果保存为 CSV，便于整理成论文中的实验数值。

注意：这里统计的延时不包含 5 分钟聚合窗口本身的等待时间，也不包含无线传输时间。
"""

import time  # 导入 time 库，用于高精度计时
import warnings  # 导入 warnings 库，用于屏蔽无关警告
import numpy as np  # 导入 numpy 库，用于数组运算与随机数生成
import pandas as pd  # 导入 pandas 库，用于读取与保存表格数据
import torch  # 导入 PyTorch 主库，用于模型推理和设备管理
import torch.nn as nn  # 导入 torch.nn 模块，用于定义网络层
import pywt  # 导入 pywt 库，用于 db4 小波分解

warnings.filterwarnings("ignore")  # 忽略运行时的警告输出，保持日志简洁


# ============================================================
# 1. 路径配置：按你的实际文件名修改
# ============================================================

# 如果你是小波基对比代码保存的模型，一般是这个名字
MODEL_PATH = "gru_wavelet_eca_db4_level3_window56.pth"  # 指定要加载的模型权重文件路径

# 如果你是单一 db4 代码保存的模型，可能是这个名字
# MODEL_PATH = "Model_GRUWavelet_no_leakage_pytorch.pth"

# 用于测试计算延时的数据，要求包含：日龄、THI、当前温度、目标温度
TEST_FILE = "部分温湿度数据.csv"  # 指定用于延时测试的温湿度数据文件

# 延时测试次数
N_WARMUP = 50  # 设置预热次数，消除首次调用与初始化开销
N_RUNS = 500  # 设置正式计时测试的采样次数

# 结果保存文件
SAVE_PATH = "db4_online_prediction_warning_latency.csv"  # 指定延时明细结果的保存路径


# ============================================================
# 2. 基础工具函数
# ============================================================

def load_csv_with_multiple_encodings(file_path, encodings=None):  # 定义多编码尝试读取 CSV 的工具函数
    """尝试多种编码读取 CSV 文件，返回首个成功解析的 DataFrame。

    参数:
        file_path: CSV 文件路径
        encodings: 候选编码列表，为 None 时使用内置默认列表
    返回:
        成功读取到的 pandas DataFrame
    异常:
        所有编码都失败时抛出 ValueError
    """
    if encodings is None:  # 若调用时未指定编码列表
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin1"]  # 使用默认的多编码候选列表

    last_error = None  # 初始化最后一次解码异常为空

    for enc in encodings:  # 依次尝试每一种候选编码
        try:  # 捕获读取过程中的解码异常
            data = pd.read_csv(file_path, encoding=enc)  # 用当前编码读取 CSV 文件
            print(f"成功使用 {enc} 编码读取文件: {file_path}")  # 打印读取成功提示与所用编码
            return data  # 读取成功，直接返回数据框
        except UnicodeDecodeError as e:  # 若出现 Unicode 解码错误
            last_error = e  # 记录本次异常，便于最后统一报错
            continue  # 继续尝试下一种编码

    raise ValueError(f"无法读取文件: {file_path}，最后错误为: {last_error}")  # 所有编码都失败则抛出异常并附带最后的错误


def fill_missing_values(df, features, target, dataset_name="延时测试数据"):  # 定义缺失值处理函数，逻辑与训练阶段保持一致
    """对测试数据做缺失值处理，保持与训练代码一致的逻辑。

    连续变量用线性插值加前后填充，离散变量仅用前后填充，
    若仍有缺失则删除对应行。

    参数:
        df: 原始数据框
        features: 特征列名列表
        target: 目标列名
        dataset_name: 用于日志输出的数据集名称
    返回:
        缺失值处理后的数据框（副本）
    """
    df = df.copy()  # 复制数据框，避免修改调用方传入的原始数据

    print("\n" + "=" * 60)  # 打印分隔线
    print(f"{dataset_name}缺失值处理")  # 打印缺失值处理标题
    print("=" * 60)  # 打印分隔线
    print(f"原始数据形状: {df.shape}")  # 打印原始数据形状

    used_cols = [col for col in features + [target] if col in df.columns]  # 筛选出数据中实际存在的特征与目标列

    for col in used_cols:  # 遍历需要检查缺失值的列
        nan_count = df[col].isna().sum()  # 统计该列的缺失值个数
        nan_pct = nan_count / len(df) * 100  # 计算缺失值所占百分比
        if nan_count > 0:  # 只打印存在缺失值的列
            print(f"{col}: {nan_count} 个缺失值 ({nan_pct:.2f}%)")  # 打印该列缺失值数量与占比

    continuous_vars = ["THI", "当前温度", "目标温度"]  # 定义连续变量列表
    discrete_vars = ["日龄"]  # 定义离散变量列表

    for col in continuous_vars:  # 遍历连续变量
        if col in df.columns and df[col].isna().sum() > 0:  # 若该列存在且含有缺失值
            df[col] = df[col].interpolate(method="linear", limit_direction="both")  # 双向线性插值填补连续变量缺失值
            df[col] = df[col].ffill()  # 前向填充开头剩余的缺失值
            df[col] = df[col].bfill()  # 后向填充结尾剩余的缺失值

    for col in discrete_vars:  # 遍历离散变量
        if col in df.columns and df[col].isna().sum() > 0:  # 若该列存在且含有缺失值
            df[col] = df[col].ffill()  # 前向填充离散变量缺失值
            df[col] = df[col].bfill()  # 后向填充离散变量缺失值

    remaining_nan = df[used_cols].isna().sum().sum()  # 统计处理后剩余的缺失值总数
    if remaining_nan > 0:  # 若仍然有缺失值
        print(f"仍有 {remaining_nan} 个缺失值，将删除对应行")  # 打印将删除含缺失值行的提示
        df = df.dropna(subset=used_cols)  # 删除含缺失值的样本行

    print(f"处理后数据形状: {df.shape}")  # 打印处理后的数据形状
    print("=" * 60 + "\n")  # 打印分隔线并换行

    return df  # 返回缺失值处理后的数据框


def cuda_sync(device):  # 定义 GPU 同步函数，保证计时准确
    """GPU 计时同步函数。

    使用 CUDA 时调用 torch.cuda.synchronize，避免异步计算导致计时结果偏小；
    CPU 设备上不做任何操作。

    参数:
        device: 当前的 torch.device 对象
    返回:
        无
    """
    if device.type == "cuda":  # 仅当当前设备是 CUDA 时需要同步
        torch.cuda.synchronize()  # 同步 GPU，等待所有异步计算完成


# ============================================================
# 3. 模型结构：必须和训练时保持一致
# ============================================================

class ECALayer(nn.Module):  # 定义 ECA 高效通道注意力层
    """
    ECA-Net: Efficient Channel Attention
    使用一维卷积实现跨通道注意力，在几乎不增加参数量的前提下增强重要通道。
    """
    def __init__(self, channels, gamma=2, b=1):  # 初始化 ECA 层，自适应计算一维卷积核大小
        """
        初始化 ECA 通道注意力层

        参数:
            channels: 输入特征的通道数
            gamma: 计算卷积核尺寸时的缩放系数
            b: 计算卷积核尺寸时的偏置项
        """
        super(ECALayer, self).__init__()  # 调用父类 nn.Module 的初始化方法

        t = int(abs((np.log2(channels) + b) / gamma))  # 按公式计算卷积核尺寸参数 t
        k_size = t if t % 2 else t + 1  # 确保卷积核大小为奇数
        k_size = max(k_size, 1)  # 保证卷积核尺寸至少为 1，防止参数异常

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 定义全局平均池化，把时间维压缩为 1
        self.conv = nn.Conv1d(  # 定义一维卷积，用于通道间信息交互
            1,  # 输入通道数为 1
            1,  # 输出通道数为 1
            kernel_size=k_size,  # 使用自适应计算得到的卷积核大小
            padding=(k_size - 1) // 2,  # 设置填充量，保持输入输出长度一致
            bias=False  # 不使用偏置项
        )
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活，生成 0-1 的注意力权重

    def forward(self, x):  # 定义前向传播：对输入特征做通道注意力加权
        """
        前向传播：计算每个通道的注意力权重并加权输入

        参数:
            x: 形状为 [batch, seq_len, channels] 的输入张量
        返回:
            通道加权后的张量，形状不变
        """
        # x: [batch, seq_len, channels]
        x_permuted = x.permute(0, 2, 1)  # 交换时间维与通道维，适配池化输入格式
        y = self.avg_pool(x_permuted)  # 全局平均池化，得到每个通道的全局描述
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 一维卷积后再转回，实现通道间信息交互
        y = self.sigmoid(y)  # Sigmoid 激活得到各通道注意力权重
        x_permuted = x_permuted * y.expand_as(x_permuted)  # 把权重按通道广播后乘回原始输入特征
        return x_permuted.permute(0, 2, 1)  # 恢复原来的维度顺序并返回


class GRUWaveletModel(nn.Module):  # 定义 GRU + 小波 + ECA 预测模型
    """
    GRU + 小波变换 + ECA注意力机制
    依次经过 ECA 注意力与堆叠的多层 GRU，最后取末尾时刻输出经全连接层给出预测值；
    模型结构必须与训练时使用的检查点保持完全一致。
    """
    def __init__(self, input_dim, time_step, gru_hidden=128, gru_layers=1, dropout=0.2):  # 初始化 GRU 小波模型结构
        """
        初始化 GRU 小波模型

        参数:
            input_dim: 输入特征维度
            time_step: 时间窗口长度
            gru_hidden: 隐藏层单元数（可为整数或整数列表）
            gru_layers: GRU 层数
            dropout: dropout 比例
        """
        super(GRUWaveletModel, self).__init__()  # 调用父类 nn.Module 的初始化方法

        self.time_step = time_step  # 保存时间窗口长度
        self.input_dim = input_dim  # 保存输入特征维度
        self.gru_hidden = gru_hidden  # 保存隐藏层配置信息

        self.eca = ECALayer(channels=input_dim)  # 实例化 ECA 注意力模块

        if isinstance(gru_hidden, list):  # 若隐藏层大小以列表形式给出，表示多层各不同尺寸
            hidden_sizes = gru_hidden  # 直接采用传入的隐藏层尺寸列表
        else:  # 否则所有 GRU 层使用相同隐藏尺寸
            hidden_sizes = [gru_hidden] * gru_layers  # 按层数复制同一个隐藏层尺寸

        self.gru_layers_list = nn.ModuleList()  # 创建 ModuleList 容器以容纳多个 GRU 层
        in_size = input_dim  # 记录当前层的输入维度

        for h_size in hidden_sizes:  # 按隐藏层尺寸列表逐层构建 GRU
            self.gru_layers_list.append(  # 把构建好的 GRU 层加入列表
                nn.GRU(  # 定义当前这一层单向 GRU
                    input_size=in_size,  # 输入维度为上一层的输出维度
                    hidden_size=h_size,  # 隐藏层单元数为当前设定的尺寸
                    num_layers=1,  # 每层单独定义，因此 num_layers 固定为 1
                    dropout=0.0,  # 单层内部不使用 dropout
                    batch_first=True,  # 输入张量第一维为 batch
                    bidirectional=False  # 使用单向 GRU，不双向
                )
            )
            in_size = h_size  # 更新下一层的输入维度为当前隐藏尺寸

        self.dropout = nn.Dropout(dropout)  # 定义 dropout 层，防止过拟合
        self.fc = nn.Linear(hidden_sizes[-1], 1)  # 定义全连接输出层，输出单一预测值

    def forward(self, x):  # 定义前向传播：ECA、多层 GRU、dropout 与全连接
        """
        前向传播：ECA -> 多层 GRU -> 全连接输出

        参数:
            x: 形状为 [batch, time_step, input_dim] 的输入序列
        返回:
            形状为 [batch, 1] 的预测值
        """
        x = self.eca(x)  # 用 ECA 增强输入特征

        out = x  # 初始化序列输出为注意力增强后的输入
        for gru in self.gru_layers_list:  # 逐层通过 GRU 编码器
            out, _ = gru(out)  # 当前 GRU 层前向计算，忽略隐藏状态输出
            out = self.dropout(out)  # 对当前层输出施加 dropout

        out_last = out[:, -1, :]  # 取最后一个时间步的输出作为序列表示
        return self.fc(out_last)  # 经全连接层输出最终预测值


# ============================================================
# 4. THI预警阈值判定
# ============================================================

def thi_warning_level(thi_value):  # 定义 THI 预警等级判定函数
    """按论文表 6 的阈值判定 THI 预警等级。

    参数:
        thi_value: 待判定的 THI 数值
    返回:
        Comfort / Warning / Danger / Emergency 四个等级字符串之一
    """
    if thi_value < 70:  # THI 小于 70 判为舒适
        return "Comfort"  # 返回 Comfort 舒适等级
    elif thi_value < 76:  # THI 小于 76 判为警告
        return "Warning"  # 返回 Warning 警告等级
    elif thi_value < 81:  # THI 小于 81 判为危险
        return "Danger"  # 返回 Danger 危险等级
    else:  # 否则判为紧急
        return "Emergency"  # 返回 Emergency 紧急等级


# ============================================================
# 5. 在线单步小波特征构造
# ============================================================

def build_one_wavelet_row_db4(  # 定义单时刻在线小波特征行的构造函数
    raw_df,
    idx,
    features,
    target_col,
    feature_names,
    level=3,
    wavelet_window=56
):
    """
    构造某一时刻 idx 的在线特征行。

    与训练代码逻辑一致：
    1. 仅对 THI 做 db4 小波分解；
    2. 其他变量保持原始值；
    3. 每个时间点只使用当前及历史数据，不使用未来数据；
    4. 输出列顺序严格按照 feature_names。
    """

    row_dict = {}  # 初始化字典，保存当前时刻各特征值

    for col in features:  # 遍历原始特征列表
        if col == target_col:  # 若当前列是需要做小波分解的目标列
            signal = raw_df[target_col].values.astype(float)  # 取出目标列全部数值并转为浮点类型

            start_idx = max(0, idx - wavelet_window + 1)  # 计算滑动窗口的起始下标
            window_signal = signal[start_idx:idx + 1]  # 截取当前时刻及其历史的窗口数据，不使用未来信息

            if len(window_signal) < wavelet_window:  # 若窗口长度不足
                pad_width = wavelet_window - len(window_signal)  # 计算需要在前端补充的点数
                window_signal = np.pad(window_signal, (pad_width, 0), mode="edge")  # 用边界值在窗口前端填充至指定长度

            try:  # 尝试对窗口信号做 db4 小波分解
                coeffs = pywt.wavedec(  # 调用 wavedec 进行多层小波分解
                    data=window_signal,  # 传入当前窗口信号
                    wavelet="db4",  # 指定使用 db4 小波基
                    level=level  # 指定分解层数
                )

                # pywt.wavedec 返回顺序为 [cA3, cD3, cD2, cD1]
                # 这里保持与你原始代码一致：detail1 对应 coeffs[1]
                row_dict[f"{target_col}_approx"] = coeffs[0][-1]  # 取近似系数的最后一个值作为近似特征

                for j, detail in enumerate(coeffs[1:]):  # 遍历各层细节系数
                    row_dict[f"{target_col}_detail{j + 1}"] = (  # 取细节系数最后值，系数为空则补 0
                        detail[-1] if len(detail) > 0 else 0.0
                    )

            except Exception:  # 若分解过程出现异常
                row_dict[f"{target_col}_approx"] = window_signal[-1]  # 降级处理：用窗口最后一个原始值作为近似特征
                for j in range(level):  # 逐层处理细节系数
                    row_dict[f"{target_col}_detail{j + 1}"] = 0.0  # 异常时把该层细节系数填为 0

        else:  # 当前列不是小波分解目标列
            row_dict[col] = raw_df.iloc[idx][col]  # 直接取该列在当前时刻的原始值

    try:  # 尝试按指定顺序组装特征数组
        return np.array([row_dict[name] for name in feature_names], dtype=np.float32)  # 按 feature_names 顺序输出 float32 特征行
    except KeyError as e:  # 若组装时缺少某个特征列
        raise KeyError(  # 抛出更清晰的特征不匹配异常
            f"特征列不匹配，缺少 {e}。请检查 checkpoint 中的 feature_names "  # 异常信息前半部分：说明缺少的特征
            f"是否与当前数据列一致。当前可用列: {list(row_dict.keys())}"  # 异常信息后半部分：列出当前可用的列
        )


def build_one_input_sequence_db4(  # 定义在线输入序列构造函数
    raw_df,
    end_idx,
    features,
    target_col,
    feature_names,
    feature_scaler,
    time_step=36,
    level=3,
    wavelet_window=56,
    device=None
):
    """
    构造一个在线输入序列。
    输出形状：[1, time_step, input_dim]

    参数:
        raw_df: 原始数据框
        end_idx: 序列结束时刻的下标
        features: 原始特征列表
        target_col: 小波分解的目标列
        feature_names: 输出特征的列顺序
        feature_scaler: 训练时拟合的特征归一化器
        time_step: 时间窗口长度
        level: 小波分解层数
        wavelet_window: 滑动窗口大小
        device: 张量要移动到的计算设备
    返回:
        形状为 [1, time_step, input_dim] 的输入张量
    """

    start_idx = end_idx - time_step + 1  # 根据结束下标和时间窗口计算序列起始下标

    if start_idx < 0:  # 若起始下标为负，窗口不完整
        raise ValueError("end_idx 太小，无法构造完整输入窗口")  # 抛出异常提示结束下标过小

    rows = []  # 初始化特征行列表

    for i in range(start_idx, end_idx + 1):  # 逐个时刻构造特征行
        one_row = build_one_wavelet_row_db4(  # 调用单时刻小波特征构造函数
            raw_df=raw_df,  # 传入原始数据框
            idx=i,  # 传入当前时刻下标
            features=features,  # 传入原始特征列表
            target_col=target_col,  # 传入小波分解目标列
            feature_names=feature_names,  # 传入输出特征的列顺序
            level=level,  # 传入小波分解层数
            wavelet_window=wavelet_window  # 传入滑动窗口大小
        )
        rows.append(one_row)  # 收集当前时刻构造好的特征行

    x = np.vstack(rows)  # 把各行垂直堆叠成二维特征矩阵

    x_scaled = feature_scaler.transform(  # 用训练时的归一化器变换特征
        pd.DataFrame(x, columns=feature_names)  # 先组装成带列名的 DataFrame
    ).astype(np.float32)  # 结果转为 float32 以匹配模型输入类型

    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0)  # 转为张量并增加 batch 维度

    if device is not None:  # 若指定了计算设备
        x_tensor = x_tensor.to(device)  # 把输入张量移动到指定设备

    return x_tensor  # 返回形状为 [1, time_step, input_dim] 的输入张量


def predict_and_warn_once_db4(  # 定义单次在线滚动预测与预警函数
    model,
    raw_df,
    end_idx,
    features,
    target_col,
    feature_names,
    feature_scaler,
    target_scaler,
    time_step=36,
    level=3,
    wavelet_window=56,
    device=None
):
    """
    单次在线滚动预测与预警输出。

    包括：
    db4 小波特征提取 -> 归一化 -> 输入窗口构建 ->
    模型推理 -> 反归一化 -> 阈值判定

    参数:
        model: 训练好的预测模型
        raw_df: 原始数据框
        end_idx: 当前时刻的下标
        features: 原始特征列表
        target_col: 小波分解目标列
        feature_names: 特征列顺序
        feature_scaler: 特征归一化器
        target_scaler: 目标归一化器
        time_step: 时间窗口长度
        level: 小波分解层数
        wavelet_window: 滑动窗口大小
        device: 计算设备
    返回:
        (预测 THI 值, 预警等级字符串) 的二元组
    """

    x_tensor = build_one_input_sequence_db4(  # 构造当前时刻的模型输入序列
        raw_df=raw_df,  # 传入原始数据
        end_idx=end_idx,  # 传入当前结束下标
        features=features,  # 传入原始特征列表
        target_col=target_col,  # 传入目标列
        feature_names=feature_names,  # 传入特征列顺序
        feature_scaler=feature_scaler,  # 传入特征归一化器
        time_step=time_step,  # 传入时间窗口长度
        level=level,  # 传入小波分解层数
        wavelet_window=wavelet_window,  # 传入小波滑动窗口
        device=device  # 传入计算设备
    )

    model.eval()  # 把模型切换为评估模式

    with torch.no_grad():  # 关闭梯度计算以加速推理
        y_pred_norm = model(x_tensor).detach().cpu().numpy()  # 前向推理并把归一化预测值转为 numpy 数组

    y_pred = target_scaler.inverse_transform(  # 对预测结果做反归一化，还原到原始量纲
        y_pred_norm.reshape(-1, 1)  # 把预测值整理成二维列向量
    )[0, 0]  # 取出标量的预测 THI 值

    warning_state = thi_warning_level(y_pred)  # 根据预测 THI 判定预警等级

    return float(y_pred), warning_state  # 返回浮点预测值与预警等级


# ============================================================
# 6. 延时测试主函数
# ============================================================

def measure_db4_online_latency(  # 定义 db4 模型在线计算延时的测试函数
    model,
    raw_df,
    features,
    target_col,
    feature_names,
    feature_scaler,
    target_scaler,
    time_step=36,
    pred_offset=6,
    level=3,
    wavelet_window=56,
    n_warmup=50,
    n_runs=500,
    save_path="db4_online_prediction_warning_latency.csv",
    device=None
):
    """
    测试 db4 模型单次滚动预测与预警输出的计算延时。

    计算延时定义：
    新的 5 min 聚合样本形成后，完成
    db4 小波特征提取、特征归一化、输入窗口构造、模型推理、
    反归一化和阈值判定所需的时间。

    注意：
    该延时不包括 5 min 聚合窗口本身的等待时间，
    也不包括无线传输时间。
    """

    model.eval()  # 确保模型处于评估模式

    min_end_idx = max(time_step, wavelet_window) - 1  # 计算可用的最小结束下标，保证输入窗口和小波窗口完整
    max_end_idx = len(raw_df) - pred_offset - 1  # 计算可用的最大结束下标，为预测偏移留出余量

    if max_end_idx <= min_end_idx:  # 若可用区间为空
        raise ValueError(  # 抛出数据长度不足的异常
            f"数据长度不足，无法测试。当前数据长度={len(raw_df)}, "  # 异常信息：当前数据实际长度
            f"至少需要大于 {min_end_idx + pred_offset + 1}"  # 异常信息：所需的最小数据长度
        )

    all_indices = np.arange(min_end_idx, max_end_idx + 1)  # 生成所有可用的测试下标数组

    rng = np.random.default_rng(2026)  # 创建固定种子的随机数生成器，保证结果可复现

    if len(all_indices) > n_runs:  # 若可用样本数多于要求的测试次数
        test_indices = rng.choice(all_indices, size=n_runs, replace=False)  # 无放回随机抽取指定数量的测试下标
    else:  # 否则可用样本数不足
        test_indices = all_indices  # 直接使用全部可用样本作为测试集

    # 预热，避免首次调用、GPU初始化、缓存建立影响计时
    warmup_indices = test_indices[:min(n_warmup, len(test_indices))]  # 取前若干个样本用于预热

    for idx in warmup_indices:  # 依次执行预热调用
        _ = predict_and_warn_once_db4(  # 执行预热预测，不保存其返回值
            model=model,  # 传入模型
            raw_df=raw_df,  # 传入原始数据
            end_idx=int(idx),  # 传入下标并转为整数
            features=features,  # 传入特征列表
            target_col=target_col,  # 传入目标列
            feature_names=feature_names,  # 传入特征顺序
            feature_scaler=feature_scaler,  # 传入特征归一化器
            target_scaler=target_scaler,  # 传入目标归一化器
            time_step=time_step,  # 传入时间窗口
            level=level,  # 传入小波分解层数
            wavelet_window=wavelet_window,  # 传入小波窗口
            device=device  # 传入计算设备
        )

    cuda_sync(device)  # 预热结束后同步一次 GPU

    records = []  # 初始化结果记录列表
    latencies_ms = []  # 初始化延时记录列表

    for idx in test_indices:  # 对正式测试样本逐一计时
        idx = int(idx)  # 确保当前下标为整数

        cuda_sync(device)  # 计时开始前同步 GPU
        t0 = time.perf_counter()  # 记录高精度计时起点

        pred_value, warning_state = predict_and_warn_once_db4(  # 执行完整的单次预测与预警流程并接收结果
            model=model,  # 传入模型
            raw_df=raw_df,  # 传入原始数据
            end_idx=idx,  # 传入当前样本下标
            features=features,  # 传入特征列表
            target_col=target_col,  # 传入目标列
            feature_names=feature_names,  # 传入特征顺序
            feature_scaler=feature_scaler,  # 传入特征归一化器
            target_scaler=target_scaler,  # 传入目标归一化器
            time_step=time_step,  # 传入时间窗口
            level=level,  # 传入小波分解层数
            wavelet_window=wavelet_window,  # 传入小波窗口
            device=device  # 传入计算设备
        )

        cuda_sync(device)  # 计时终点前先同步 GPU
        t1 = time.perf_counter()  # 记录高精度计时终点

        latency_ms = (t1 - t0) * 1000.0  # 计算单次延时并换算成毫秒
        latencies_ms.append(latency_ms)  # 把本次延时加入统计列表

        records.append({  # 把预测结果与延时整理成一条记录
            "end_idx": idx,  # 记录当前样本下标
            "predicted_THI": pred_value,  # 记录预测得到的 THI 值
            "warning_state": warning_state,  # 记录判定出的预警等级
            "latency_ms": latency_ms  # 记录本次计算延时
        })

    latency_arr = np.array(latencies_ms)  # 把延时列表转成数组，便于统计计算

    result_df = pd.DataFrame(records)  # 把记录列表转成 DataFrame
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")  # 把延时明细保存为 CSV，使用带 BOM 的编码兼容 Excel

    print("\n" + "=" * 80)  # 打印分隔线
    print("db4 单次滚动预测与预警输出计算延时测试结果")  # 打印延时测试结果标题
    print("=" * 80)  # 打印分隔线
    print(f"设备: {device}")  # 打印测试所用设备
    print(f"测试次数: {len(latency_arr)}")  # 打印实际测试次数
    print(f"模型输入窗口: {time_step} 步 = {time_step * 5} min")  # 打印模型输入窗口对应的分钟数
    print(f"预测提前量: {pred_offset} 步 = {pred_offset * 5} min")  # 打印预测提前量对应的分钟数
    print("小波基: db4")  # 打印所用小波基
    print(f"小波分解层数: {level}")  # 打印小波分解层数
    print(f"小波窗口: {wavelet_window} 步 = {wavelet_window * 5} min")  # 打印小波窗口对应的分钟数
    print("-" * 80)  # 打印次级分隔线
    print(f"平均延时: {latency_arr.mean():.3f} ms ({latency_arr.mean() / 1000:.6f} s)")  # 打印平均延时（毫秒与秒）
    print(f"中位数延时: {np.median(latency_arr):.3f} ms")  # 打印延时的中位数
    print(f"95%分位延时: {np.percentile(latency_arr, 95):.3f} ms ({np.percentile(latency_arr, 95) / 1000:.6f} s)")  # 打印 95% 分位延时（毫秒与秒）
    print(f"最大延时: {latency_arr.max():.3f} ms ({latency_arr.max() / 1000:.6f} s)")  # 打印最大延时（毫秒与秒）
    print(f"最小延时: {latency_arr.min():.3f} ms")  # 打印最小延时
    print("-" * 80)  # 打印次级分隔线
    print(f"详细延时结果已保存: {save_path}")  # 打印明细结果的保存路径
    print("=" * 80 + "\n")  # 打印分隔线并换行

    return {  # 返回延时统计摘要与明细数据框
        "runs": len(latency_arr),  # 摘要：实际测试次数
        "mean_ms": float(latency_arr.mean()),  # 摘要：平均延时（毫秒）
        "median_ms": float(np.median(latency_arr)),  # 摘要：中位数延时（毫秒）
        "p95_ms": float(np.percentile(latency_arr, 95)),  # 摘要：95% 分位延时（毫秒）
        "max_ms": float(latency_arr.max()),  # 摘要：最大延时（毫秒）
        "min_ms": float(latency_arr.min()),  # 摘要：最小延时（毫秒）
        "mean_s": float(latency_arr.mean() / 1000.0),  # 摘要：平均延时（秒）
        "p95_s": float(np.percentile(latency_arr, 95) / 1000.0),  # 摘要：95% 分位延时（秒）
        "max_s": float(latency_arr.max() / 1000.0)  # 摘要：最大延时（秒）
    }, result_df


# ============================================================
# 7. 加载模型和数据，并开始测试
# ============================================================

def main():  # 定义主函数：加载模型与数据并执行延时测试
    """
    主函数：加载检查点、重建模型、读取测试数据并执行延时测试

    主要负责：选择设备、读取模型配置与权重、加载并清洗测试数据、
    调用 measure_db4_online_latency 完成测试，并打印可写入论文的指标。

    参数:
        无（使用文件顶部定义的全局配置常量）
    返回:
        无
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择推理设备：有 GPU 用 CUDA，否则用 CPU
    print(f"使用设备: {device}")  # 打印当前使用的设备

    if torch.cuda.is_available():  # 若检测到可用的 GPU
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 打印 GPU 名称

    # 加载 checkpoint
    try:  # 尝试用新接口加载检查点
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # 加载模型权重与配置到指定设备，关闭权重-only 校验
    except TypeError:  # 若当前 torch 版本不支持 weights_only 参数
        checkpoint = torch.load(MODEL_PATH, map_location=device)  # 退回旧接口加载检查点

    print(f"\n模型文件加载完成: {MODEL_PATH}")  # 打印模型文件加载完成提示

    input_dim = checkpoint["input_dim"]  # 从检查点读取输入特征维度
    time_step = checkpoint["time_step"]  # 从检查点读取时间窗口长度
    feature_names = checkpoint["feature_names"]  # 从检查点读取特征名称顺序
    feature_scaler = checkpoint["feature_scaler"]  # 从检查点读取特征归一化器
    target_scaler = checkpoint["target_scaler"]  # 从检查点读取目标归一化器

    wavelet_window = checkpoint.get("wavelet_window", time_step)  # 读取小波窗口，缺省时使用时间窗口长度
    wavelet_level = checkpoint.get("level", 3)  # 读取小波分解层数，默认 3
    target_col = checkpoint.get("wavelet_target_col", "THI")  # 读取小波分解的目标列，默认 THI
    features = checkpoint.get("original_features", ["日龄", "THI", "当前温度", "目标温度"])  # 读取原始特征列表，缺省为常用的四项

    pred_offset = 6  # 设定预测偏移步数

    print("\n模型参数信息:")  # 打印模型参数信息标题
    print(f"input_dim: {input_dim}")  # 打印输入维度
    print(f"time_step: {time_step}")  # 打印时间窗口长度
    print(f"pred_offset: {pred_offset}")  # 打印预测偏移步数
    print(f"feature_names: {feature_names}")  # 打印特征名称列表
    print(f"features: {features}")  # 打印原始特征列表
    print(f"target_col: {target_col}")  # 打印小波目标列
    print(f"wavelet_window: {wavelet_window}")  # 打印小波窗口大小
    print(f"wavelet_level: {wavelet_level}")  # 打印小波分解层数
    print("wavelet: db4")  # 打印所用小波基

    # 重建模型
    model = GRUWaveletModel(  # 按检查点中的参数重建模型结构
        input_dim=input_dim,  # 传入输入特征维度
        time_step=time_step,  # 传入时间窗口长度
        gru_hidden=128,  # 设置 GRU 隐藏层单元数为 128
        gru_layers=1,  # 设置 GRU 层数为 1
        dropout=0.2  # 设置 dropout 比例为 0.2
    ).to(device)  # 把模型移动到指定计算设备

    model.load_state_dict(checkpoint["model_state_dict"])  # 加载训练好的模型权重
    model.eval()  # 把模型切换为评估模式

    print("\n模型权重加载完成。")  # 打印模型权重加载完成提示

    # 加载测试数据
    raw_df = load_csv_with_multiple_encodings(TEST_FILE)  # 用多编码方式读取延时测试数据

    # 如果没有时间戳，不影响延时测试
    if "timestamp" not in raw_df.columns:  # 若数据中没有时间戳列
        raw_df["timestamp"] = pd.date_range(  # 按 5 分钟间隔补造时间戳列
            start="2025/6/21 0:04",  # 设置时间戳的起始时间
            periods=len(raw_df),  # 时间戳数量与数据行数保持一致
            freq="5min"  # 采样间隔为 5 分钟
        )

    # 检查必要列
    required_cols = list(set(features + [target_col]))  # 汇总所需的全部列并去除重复
    missing_cols = [col for col in required_cols if col not in raw_df.columns]  # 找出测试数据中缺失的必要列

    if missing_cols:  # 若存在缺失列
        raise ValueError(  # 抛出异常终止程序
            f"测试数据缺少必要列: {missing_cols}\n"  # 异常信息：缺少的必要列
            f"当前数据列为: {list(raw_df.columns)}"  # 异常信息：数据中实际存在的列
        )

    raw_df = fill_missing_values(  # 对测试数据执行缺失值处理
        raw_df,  # 传入原始数据框
        features=features,  # 传入特征列表
        target=target_col,  # 传入小波目标列
        dataset_name="延时测试数据"  # 传入数据集名称用于日志输出
    ).reset_index(drop=True)  # 处理完成后重置索引，保证下标连续

    # 开始延时测试
    latency_summary, latency_detail = measure_db4_online_latency(  # 执行延时测试，接收统计摘要与明细结果
        model=model,  # 传入模型
        raw_df=raw_df,  # 传入处理后的数据
        features=features,  # 传入特征列表
        target_col=target_col,  # 传入小波目标列
        feature_names=feature_names,  # 传入特征顺序
        feature_scaler=feature_scaler,  # 传入特征归一化器
        target_scaler=target_scaler,  # 传入目标归一化器
        time_step=time_step,  # 传入时间窗口长度
        pred_offset=pred_offset,  # 传入预测偏移步数
        level=wavelet_level,  # 传入小波分解层数
        wavelet_window=wavelet_window,  # 传入小波窗口大小
        n_warmup=N_WARMUP,  # 传入预热次数
        n_runs=N_RUNS,  # 传入正式测试次数
        save_path=SAVE_PATH,  # 传入结果保存路径
        device=device  # 传入计算设备
    )

    print("可写入论文的数值:")  # 打印便于写入论文的指标标题
    print(f"平均计算延时 = {latency_summary['mean_ms']:.3f} ms")  # 打印平均计算延时
    print(f"95%分位计算延时 = {latency_summary['p95_ms']:.3f} ms")  # 打印 95% 分位计算延时
    print(f"最大计算延时 = {latency_summary['max_ms']:.3f} ms")  # 打印最大计算延时


if __name__ == "__main__":  # 仅当脚本被直接运行时才执行主函数
    main()  # 调用主函数，启动整个延时测试流程
