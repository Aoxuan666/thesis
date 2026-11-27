# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8
"""
鸡舍环境实时滚动预测与双指标预警系统 v2
功能：加载数据、特征工程、模型推理、预警判断、结果可视化

系统按滚动时间窗口对温度偏差 TD 与温湿指数 THI 分别做批量推理，
再根据日龄对应的适宜温度区间判定 TD 偏热/偏冷，并按 THI 数值划分舒适、警告、危险、紧急等级，
最终输出四子图 SCI 风格结果图（PNG 与 TIFF）以及逐步预警日志 CSV。

"""

import os  # 导入操作系统接口模块，用于路径与文件判断
import warnings  # 导入 warnings 模块用于忽略告警

warnings.filterwarnings('ignore')  # 忽略全部运行告警，保持输出整洁

import numpy as np  # 导入 numpy 数值计算库
import pandas as pd  # 导入 pandas 数据分析库
import matplotlib  # 导入 matplotlib 绘图库

matplotlib.use('Agg')  # 指定 Agg 后端以便无界面保存图片

import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口
import matplotlib.patches as mpatches  # 导入 mpatches 用于自定义图例色块
from matplotlib.gridspec import GridSpec  # 导入 GridSpec 用于灵活排布子图
from matplotlib.ticker import FuncFormatter  # 导入 FuncFormatter 用于自定义坐标轴刻度显示

import torch  # 导入 PyTorch 主模块
import torch.nn as nn  # 导入 PyTorch 神经网络子模块
import pywt  # 导入 pywt 用于小波分解
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 归一化工具
from decimal import Decimal, ROUND_HALF_UP  # 导入 Decimal 与 ROUND_HALF_UP 用于严格四舍五入

# ============================================================
# Matplotlib 全局样式设置
# ============================================================
plt.rcParams['font.family'] = 'Arial'  # 设置全局字体为 Arial 以符合期刊风格

# 关键修改：使用真正的数学负号 U+2212，而不是普通 hyphen-minus "-"
plt.rcParams['axes.unicode_minus'] = True  # 开启 unicode 负号设置，使用真正的数学负号

# Animals / MDPI figure style
plt.rcParams['font.size'] = 9  # 设置全局基础字号为 9
plt.rcParams['axes.titlesize'] = 10  # 设置坐标轴标题字号为 10
plt.rcParams['axes.labelsize'] = 9  # 设置坐标轴标签字号为 9
plt.rcParams['xtick.labelsize'] = 8  # 设置横轴刻度标签字号为 8
plt.rcParams['ytick.labelsize'] = 8  # 设置纵轴刻度标签字号为 8
plt.rcParams['legend.fontsize'] = 8  # 设置图例字号为 8
plt.rcParams['figure.dpi'] = 300  # 设置画布显示 dpi 为 300
plt.rcParams['savefig.dpi'] = 600  # 设置图片保存 dpi 为 600
plt.rcParams['pdf.fonttype'] = 42  # 设置 PDF 字体类型为 42，保证字体嵌入
plt.rcParams['ps.fonttype'] = 42  # 设置 PS 字体类型为 42，保证字体嵌入

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据 GPU 是否可用选择计算设备


# ============================================================
# 工具函数
# ============================================================
def round_half_up(x, decimals=1):  # 定义严格四舍五入函数
    """严格四舍五入"""  # 补充：返回四舍五入到 decimals 位的结果；输入为缺失值时返回 NaN
    if pd.isna(x):  # 若输入是缺失值
        return np.nan  # 直接返回 NaN
    factor = 10 ** decimals  # 计算 10 的 decimals 次方作为放大因子
    return float(Decimal(str(float(x) * factor)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) / factor  # 用 Decimal 按 ROUND_HALF_UP 规则舍入后再缩回原量级


def unicode_minus_formatter(x, pos):  # 定义坐标轴刻度格式化函数，强制使用数学负号
    """
    坐标轴刻度格式化函数。
    作用：强制将普通 hyphen-minus '-' 替换为真正的数学负号 '−'，即 U+2212。
    作用：把刻度格式化后的字符串中的普通 ASCII 减号替换为数学负号 U+2212，
    参数 x 为刻度数值、pos 为刻度位置（此处未使用）；返回替换后的字符串。
    """
    return f'{x:g}'.replace('-', '−')  # 把刻度值格式化后把普通减号替换为数学负号 U+2212


# ============================================================
# 模型定义
# ============================================================
class ECALayer(nn.Module):  # 定义 ECA 通道注意力层
    """
    ECA 通道注意力层：用一维卷积按通道学习权重并重标定输入特征。
    """
    def __init__(self, ch, gamma=2, b=1):  # 定义构造函数，接收通道数与核大小参数
        """
        初始化 ECA 层。
        参数 ch 为输入通道数，gamma 与 b 用于自适应计算卷积核大小。
        无返回值。
        """
        super().__init__()  # 调用父类初始化方法
        t = int(abs((np.log2(ch) + b) / gamma))  # 按通道数计算卷积核大小基数 t
        k = t if t % 2 else t + 1  # 保证卷积核大小为奇数
        self.pool = nn.AdaptiveAvgPool1d(1)  # 定义自适应平均池化层
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)  # 定义一维卷积用于学习通道权重
        self.sig = nn.Sigmoid()  # 定义 Sigmoid 激活函数

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播。
        参数 x 为形状 (batch, seq_len, channels) 的输入张量；
        返回同形状的通道重标定结果。
        """
        p = x.permute(0, 2, 1)  # 把输入维度置换为通道在前的形式
        y = self.pool(p)  # 对通道做全局平均池化
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)  # 用一维卷积学习通道权重并转置回原维度
        return (p * self.sig(y).expand_as(p)).permute(0, 2, 1)  # 将权重乘回各通道并置换回原维度后返回


class GRUWaveletECA(nn.Module):  # 定义 GRU 加小波加 ECA 的预测模型
    """
    GRU 加小波加 ECA 模型：ECA 注意力后接多层 GRU，取末时间步输出预测。
    """
    def __init__(self, input_dim, gru_hidden=128, gru_layers=1, dropout=0.3):  # 定义构造函数，接收输入维度、隐藏尺寸、层数与丢弃率
        """
        初始化模型。
        参数 input_dim 为输入特征维度，gru_hidden 为隐藏尺寸或尺寸列表，
        gru_layers 为 GRU 层数，dropout 为丢弃率。无返回值。
        """
        super().__init__()  # 调用父类初始化方法
        self.eca = ECALayer(input_dim)  # 在 GRU 之前加入 ECA 通道注意力层
        hs = gru_hidden if isinstance(gru_hidden, list) else [gru_hidden] * gru_layers  # 把隐藏尺寸统一为列表形式
        self.gru_layers_list = nn.ModuleList()  # 创建存放多个 GRU 层的 ModuleList
        sz = input_dim  # 初始化当前层输入维度为输入特征维度
        for h in hs:  # 逐个隐藏尺寸构建 GRU 层
            self.gru_layers_list.append(nn.GRU(sz, h, num_layers=1, batch_first=True))  # 追加单层 GRU 到列表，批次维度在前
            sz = h  # 把下一层的输入维度更新为当前隐藏尺寸
        self.drop = nn.Dropout(dropout)  # 定义 Dropout 层
        self.fc = nn.Linear(hs[-1], 1)  # 定义全连接层输出单个预测值

    def forward(self, x):  # 定义前向传播方法
        """
        前向传播。
        参数 x 为小波特征序列张量；返回形状 (batch, 1) 的预测值。
        """
        x = self.eca(x)  # 先对输入应用 ECA 注意力
        for g in self.gru_layers_list:  # 依次通过每个 GRU 子层
            x, _ = g(x)  # 运行当前 GRU 层
            x = self.drop(x)  # 每层之后施加 Dropout
        return self.fc(x[:, -1, :])  # 取最后时间步输出接全连接层得到预测


# ============================================================
# 小波特征提取
# ============================================================
def make_wavelet_features(df, target_col, wavelet='db4', level=3, window=64):  # 定义小波特征提取函数
    """对数据进行小波分解，提取特征"""  # 补充：对目标列做滑动窗口小波分解生成近似与细节特征，其余列原样保留；返回特征 DataFrame
    out = pd.DataFrame(index=df.index)  # 创建与原数据同索引的空 DataFrame 存放结果

    for col in df.columns:  # 逐列遍历输入数据
        sig = df[col].values.astype(np.float32)  # 取出当前列的一维信号并转为 float32

        if col == target_col:  # 仅对目标列进行小波分解
            n = len(sig)  # 记录信号长度
            pad = np.pad(sig, (window, 0), mode='edge')  # 在信号前侧补齐 window 个边缘值
            app_all = []  # 初始化近似系数列表
            dets_all = [[] for _ in range(level)]  # 为每个分解层初始化细节系数列表

            for i in range(n):  # 逐时间步构造长度为 window 的窗口
                s = pad[i:i + window]  # 从补齐信号中截取当前窗口
                try:  # 捕获小波分解可能出现的异常
                    cs = pywt.wavedec(s, wavelet, level=level)  # 对窗口做多层离散小波分解
                    app_all.append(float(cs[0][-1]))  # 取近似系数最后一个值作为当前步特征
                    for j, d in enumerate(cs[1:]):  # 遍历各层细节系数
                        dets_all[j].append(float(d[-1]) if len(d) else 0.0)  # 取每层细节系数最后一个值，空则补 0
                except Exception:  # 分解失败时执行兜底逻辑
                    app_all.append(float(s[-1]))  # 用窗口最后一个原始值作为近似特征
                    for j in range(level):  # 遍历各分解层
                        dets_all[j].append(0.0)  # 细节特征统一填 0

            out[col + '_approx'] = app_all  # 把近似系数写入目标列对应特征列
            for j in range(level):  # 逐层写入细节系数列
                out[f'{col}_detail{j + 1}'] = dets_all[j]  # 写入第 j+1 层细节特征
        else:  # 非目标列分支
            out[col] = sig  # 把原始列整体复制到结果表

    return out  # 返回构造好的小波特征表


# ============================================================
# 温度规则
# ============================================================
TEMP_RANGES = [  # 定义鸡舍各日龄段对应的适宜温度范围表
    (1, 3, 34, 35),  # 1 至 3 日龄适宜温度 34 至 35 摄氏度
    (4, 7, 32, 33),  # 4 至 7 日龄适宜温度 32 至 33 摄氏度
    (8, 15, 30, 31),  # 8 至 15 日龄适宜温度 30 至 31 摄氏度
    (16, 23, 28, 29),  # 16 至 23 日龄适宜温度 28 至 29 摄氏度
    (24, 31, 26, 27),  # 24 至 31 日龄适宜温度 26 至 27 摄氏度
    (32, 39, 24, 25),  # 32 至 39 日龄适宜温度 24 至 25 摄氏度
    (40, 999, 22, 23),  # 40 日龄以上适宜温度 22 至 23 摄氏度
]  # 温度范围表定义结束

WARN_MARGIN = 2.5  # 预警阈值偏移  # 补充：预警阈值在适宜温度上下界之外再偏移 2.5 摄氏度


def get_temp_range(age):  # 定义按日龄查询适宜温度范围的函数
    """根据日龄获取适宜温度范围"""  # 补充：返回适宜温度下界、上界及其中间值；参数 age 为日龄，超出已知区间时返回默认范围
    age = int(age)  # 把日龄转为整数
    for lo, hi, tlo, thi in TEMP_RANGES:  # 遍历所有日龄区间
        if lo <= age <= hi:  # 若当前日龄落在该区间内
            return tlo, thi, (tlo + thi) / 2.0  # 返回该区间的温度下界、上界与中间值
    return 22, 23, 22.5  # 未匹配任何区间时返回默认温度范围


def classify_td(td_pred, target_temp, age):  # 定义温度偏差 TD 分类函数
    """
    TD 分类：
    返回 (状态, 等级, 偏热阈值, 偏冷阈值)
    状态: '偏热'/'偏冷'/'正常'
    等级: 1/-1/0
    参数 td_pred 为预测温度偏差，target_temp 为目标温度，age 为日龄；
    返回四元组（状态, 等级, 偏热 TD 阈值, 偏冷 TD 阈值），等级 1 偏热、-1 偏冷、0 正常。
    """  # 补充：返回四元组（状态, 等级, 偏热阈值, 偏冷阈值）；参数 td_pred 为预测偏差，target_temp 为目标温度，age 为日龄
    tlo, thi, _ = get_temp_range(age)  # 查询当前日龄对应的适宜温度范围
    cur_temp = td_pred + target_temp  # 计算预测的实际温度等于偏差加目标温度
    hot_thresh = thi + WARN_MARGIN  # 计算偏热阈值等于适宜上限加预警余量
    cold_thresh = tlo - WARN_MARGIN  # 计算偏冷阈值等于适宜下限减预警余量

    hot_td = hot_thresh - target_temp  # 把偏热温度阈值换算成相对目标温度的 TD 阈值
    cold_td = cold_thresh - target_temp  # 把偏冷温度阈值换算成相对目标温度的 TD 阈值

    if cur_temp > hot_thresh:  # 若预测实际温度高于偏热阈值
        return '偏热', 1, hot_td, cold_td  # 返回偏热状态及等级 1 与两个 TD 阈值
    elif cur_temp < cold_thresh:  # 若预测实际温度低于偏冷阈值
        return '偏冷', -1, hot_td, cold_td  # 返回偏冷状态及等级 -1 与两个 TD 阈值
    else:  # 其他情况为正常
        return '正常', 0, hot_td, cold_td  # 返回正常状态及等级 0 与两个 TD 阈值


def classify_thi(thi_val):  # 定义 THI 风险等级分类函数
    """
    THI 分类：
    返回 (状态, 等级)
    参数 thi_val 为温湿指数数值；
    返回二元组（状态, 等级），等级 0 舒适、1 警告、2 危险、3 紧急。
    """  # 补充：返回（状态, 等级）二元组，参数 thi_val 为温湿指数值，等级 0 至 3 对应舒适到紧急
    if thi_val < 70:  # 若 THI 小于 70
        return '舒适', 0  # 返回舒适状态及等级 0
    elif thi_val < 76:  # 若 THI 小于 76
        return '警告', 1  # 返回警告状态及等级 1
    elif thi_val < 81:  # 若 THI 小于 81
        return '危险', 2  # 返回危险状态及等级 2
    else:  # 其他情况为紧急
        return '紧急', 3  # 返回紧急状态及等级 3


# ============================================================
# 数据加载
# ============================================================
def load_data(path=None):  # 定义稳健读取 CSV 数据的函数
    """稳健读取 CSV：支持 10/11 列、自动分隔符检测，并明确打印失败原因"""  # 补充：自动探测编码与分隔符，兼容 10 列和 11 列格式；参数 path 可选，返回清洗后的 DataFrame
    import glob  # 导入 glob 模块用于匹配 CSV 文件

    col_names_10 = [  # 定义 10 列格式对应的列名列表
        '时间戳', '日龄', 'CO2', '当前负压',  # 10 列格式的列名内容
        '目标温度', '当前温度', '外界温度', '相对湿度', '温度偏差', 'THI'  # 10 列格式的列名内容续
    ]  # 10 列格式的列名内容续

    col_names_11 = [  # 定义 11 列格式对应的列名列表
        '时间戳', '日龄', 'CO2', '当前风速', '目标温度',  # 11 列格式的列名内容
        '当前温度', '外界温度', '相对湿度', '温度偏差', '额外列', 'THI'  # 11 列格式的列名内容续
    ]  # 11 列格式的列名内容续

    if path is None:  # 若未传入数据文件路径
        csv_files = [  # 匹配当前目录下所有 CSV 文件
            f for f in glob.glob('*.csv')  # 排除日志类 CSV 文件
            if f not in ['warning_simulation_log.csv', 'warning_metrics_result.csv']  # 排除条件结束
        ]  # 若没有找到任何 CSV 文件
        if not csv_files:
            raise FileNotFoundError('找不到 CSV 数据文件')  # 抛出找不到数据文件的错误
        path = max(csv_files, key=lambda f: os.path.getsize(f))  # 选择体积最大的 CSV 文件作为数据源
        print(f'  使用数据文件: {path}')  # 打印实际使用的数据文件名
    elif not os.path.exists(path):  # 若传入的路径不存在
        raise FileNotFoundError(f'指定的数据文件不存在: {path}')  # 抛出指定文件不存在的错误

    need = ['时间戳', '日龄', '目标温度', '当前温度', '相对湿度', '温度偏差', 'THI']  # 定义必须存在的关键列列表
    tried_errors = []  # 初始化记录尝试失败原因的列表

    for enc in ['gb18030', 'gbk', 'gb2312', 'utf-8-sig', 'utf-8']:  # 依次尝试多种字符编码
        for sep in [',', None]:  # 对每种编码尝试逗号分隔和自动分隔
            try:  # 捕获本次读取可能出现的异常
                if sep is None:  # 若使用自动分隔符探测
                    df = pd.read_csv(path, encoding=enc, header=0, sep=None, engine='python')  # 用 python 引擎自动探测分隔符读取数据
                else:  # 否则使用指定逗号作为分隔符读取
                    df = pd.read_csv(path, encoding=enc, header=0, sep=sep)  # 用逗号分隔符读取数据

                ncols = len(df.columns)  # 统计数据列数

                if ncols == 10:  # 若数据为 10 列
                    df.columns = col_names_10  # 套用 10 列格式的列名
                elif ncols == 11:  # 若数据为 11 列
                    df.columns = col_names_11  # 套用 11 列格式的列名
                else:  # 列数不符合预期
                    raise ValueError(f'列数异常: {ncols}，不是10列也不是11列')  # 抛出列数异常错误

                if '额外列' in df.columns:  # 若存在额外的占位列
                    df = df.drop(columns=['额外列'])  # 删除该额外列

                missing = [c for c in need if c not in df.columns]  # 检查是否缺少必要列
                if missing:  # 若存在缺失的必要列
                    raise ValueError(f'缺少必要列: {missing}')  # 抛出缺少必要列的错误

                df['时间戳'] = pd.to_datetime(df['时间戳'], errors='coerce')  # 把时间戳列转换为日期时间类型，错误值置空

                for c in df.columns:  # 遍历所有数据列
                    if c != '时间戳':  # 除时间戳外的列都转为数值类型
                        df[c] = pd.to_numeric(df[c], errors='coerce')  # 用 to_numeric 转换并容错

                before_n = len(df)  # 记录清洗前的行数
                df = df.dropna(subset=need).reset_index(drop=True)  # 删除必要列含缺失的行并重置索引
                after_n = len(df)  # 记录清洗后的行数

                if after_n == 0:  # 若清洗后没有有效数据
                    raise ValueError('清洗后有效数据为 0，请检查时间列、数值列或分隔符是否正确')  # 抛出无有效数据的错误并提示检查

                sep_name = 'auto' if sep is None else sep  # 根据分隔符是否自动探测生成显示名称
                print(f'  [OK] 读取成功 [{enc}] sep={sep_name} shape={df.shape} 删除={before_n - after_n}')  # 打印成功读取的编码、分隔符、形状与删除行数
                return df  # 返回清洗后的数据框

            except Exception as e:  # 捕获本次尝试的异常并绑定变量
                sep_name = 'auto' if sep is None else sep  # 生成显示用的分隔符名称
                tried_errors.append(f'[encoding={enc}, sep={sep_name}] {repr(e)}')  # 把本次失败原因追加到列表
                continue  # 继续尝试下一种编码或分隔符

    raise ValueError('无法读取文件: {}\n尝试详情：\n{}'.format(path, '\n'.join(tried_errors)))  # 所有尝试均失败时抛出汇总错误


# ============================================================
# 模型加载
# ============================================================
def load_model(path, tag):  # 定义加载 PyTorch 模型的函数
    """加载 PyTorch 模型"""  # 补充：参数 path 为权重文件路径，tag 为打印用标签；返回加载好的模型与检查点字典
    ck = torch.load(path, map_location=device, weights_only=False)  # 加载检查点文件并映射到指定设备
    m = GRUWaveletECA(ck['input_dim'], 128, 1, 0.3).to(device)  # 按检查点配置实例化模型并移到设备
    m.load_state_dict(ck['model_state_dict'])  # 把检查点权重加载进模型
    m.eval()  # 切换模型为评估模式
    print(f'  [OK] {tag}: input_dim={ck["input_dim"]} time_step={ck["time_step"]}')  # 打印模型加载成功信息及输入输出配置
    return m, ck  # 返回模型与检查点字典


# ============================================================
# 批量推理
# ============================================================
def batch_predict(model, data_norm, time_step, batch_size=128):  # 定义批量推理函数
    """批量推理"""  # 补充：参数 model 为模型，data_norm 为归一化特征，time_step 为窗口长度，batch_size 为批大小；返回展平后的一维预测数组
    n = len(data_norm) - time_step + 1  # 计算可构造的样本窗口数量

    X = np.stack([  # 用列表推导堆叠所有时间窗口并转为 float32 数组
        data_norm.iloc[i:i + time_step].values  # 取每个窗口对应的归一化特征片段
        for i in range(n)  # 遍历所有窗口起点
    ]).astype(np.float32)  # 堆叠完成并转为数组

    preds = []  # 初始化保存各批次预测结果的列表
    with torch.no_grad():  # 关闭梯度计算以加速推理
        for b in range(0, n, batch_size):  # 按批大小遍历所有样本
            xb = torch.FloatTensor(X[b:b + batch_size]).to(device)  # 把当前批数据转为张量并移到设备
            preds.append(model(xb).cpu().numpy())  # 调用模型预测并把结果取回 CPU 转为 numpy

    return np.concatenate(preds, axis=0).flatten()  # 拼接所有批次结果并展平成一维数组返回


# ============================================================
# 可视化
# ============================================================
def plot_results(res, output_path):  # 定义结果可视化函数，按 SCI 期刊风格绘图
    """Plot results in SCI publication format (English)"""  # 补充：参数 res 为结果 DataFrame，output_path 为图片保存路径；函数无返回值，同时保存 PNG 与 TIFF

    # ── SCI-grade color palette ──────────────────────────────────────────
    C_ACTUAL = '#2166AC'  # 定义实际值曲线配色
    C_PRED = '#B2182B'  # 定义预测值曲线配色
    C_THRESH_H = '#D6604D'  # 定义偏热阈值线配色
    C_THRESH_L = '#4393C3'  # 定义偏冷阈值线配色
    C_BAND = '#92C5DE'  # 定义阈值区间填充配色

    C_THI_L1 = '#4DAC26'  # 定义 THI 舒适等级配色
    C_THI_L2 = '#F1A340'  # 定义 THI 警告等级配色
    C_THI_L3 = '#B2182B'  # 定义 THI 危险等级配色

    C_ZONE0 = '#F7F7F7'  # 定义 TD 正常区间配色
    C_ZONE1 = '#FEE0B6'  # 定义 TD 偏热配色
    C_ZONE2 = '#FDDBC7'  # 定义 TD 偏冷配色

    C_TD_NORMAL = '#92C5DE'  # 定义 THI 等级 0 配色
    C_TD_HOT = '#D6604D'  # 定义 THI 等级 1 配色
    C_TD_COLD = '#4393C3'  # 定义 THI 等级 2 配色
  # 定义 THI 等级 3 配色
    C_THI_0 = '#92C5DE'
    C_THI_1 = '#FDB863'
    C_THI_2 = '#E08214'
    C_THI_3 = '#8B0000'
    # ────────────────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(8.2, 10.6))  # 创建 8.2x10.6 英寸画布
    gs = GridSpec(4, 1, figure=fig, hspace=0.45)  # 用 GridSpec 创建 4 行 1 列的子图布局

    steps = res['step'].values  # 取出步数列作为横轴数据
    datetimes = res['当前时间']  # 取出当前时间列用于刻度标签
    tick_interval = max(1, len(steps) // 8)  # 计算刻度线的间隔，最多显示约 8 个
    ticks = steps[::tick_interval]  # 按间隔抽取刻度位置
    tick_labels = datetimes.dt.strftime('%m-%d\n%H:%M').values[::tick_interval]  # 按间隔抽取日期时间刻度标签

    _leg_kw = dict(  # 定义统一的图例关键字参数字典
        loc='upper left',  # 图例放置在左上角
        bbox_to_anchor=(1.01, 1.0),  # 图例锚点稍微移出坐标轴右侧
        fontsize=8,  # 图例字号
        frameon=True,  # 显示图例边框
        framealpha=0.9,  # 图例边框透明度
        edgecolor='#CCCCCC',  # 图例边框颜色
        shadow=False  # 不显示阴影
    )  # 字典定义结束

    # ── Subplot 1: Temperature Deviation (TD) ───────────────────────────
    ax1 = fig.add_subplot(gs[0])  # 取得第一个子图，绘制温度偏差
    ax1.plot(steps, res['TD实际'], color=C_ACTUAL, lw=1.5, label='Actual TD', alpha=0.9)  # 绘制实际温度偏差曲线
    ax1.plot(steps, res['TD预测'], color=C_PRED, lw=1.5, label='Predicted TD', alpha=0.9, ls='--')  # 绘制预测温度偏差曲线，使用虚线
    ax1.plot(steps, res['TD偏热阈值'], color=C_THRESH_H, lw=1.2, label='Upper threshold', alpha=0.75, ls=':')  # 绘制偏热阈值线，使用点线
    ax1.plot(steps, res['TD偏冷阈值'], color=C_THRESH_L, lw=1.2, label='Lower threshold', alpha=0.75, ls=':')  # 绘制偏冷阈值线，使用点线
    ax1.fill_between(  # 在偏冷与偏热阈值之间填充浅色带
        steps,  # 填充区域横轴为步数
        res['TD偏冷阈值'],  # 填充下界为偏冷阈值
        res['TD偏热阈值'],  # 填充上界为偏热阈值
        alpha=0.08,  # 填充透明度
        color=C_BAND,  # 填充颜色
        label='_nolegend_'  # 该填充不进入图例
    )  # fill_between 调用结束
    ax1.set_ylabel('Temperature Deviation (°C)', fontsize=9)  # 设置温度偏差子图纵轴标签
    ax1.set_title('(a) Temperature deviation prediction and threshold-based warning',  # 设置温度偏差子图标题
                  fontsize=10, fontweight='bold')  # 标题字号与加粗设置
    ax1.legend(**_leg_kw)  # 应用统一图例参数
    ax1.set_xticks(ticks)  # 设置横轴刻度位置
    ax1.set_xticklabels(tick_labels, rotation=35, ha='right')  # 设置横轴刻度标签并旋转
    ax1.grid(True, alpha=0.25, linestyle='--', color='#AAAAAA')  # 添加半透明虚线网格
    ax1.set_xlim(steps[0], steps[-1])  # 限定横轴为实际步数范围
    ax1.spines[['top', 'right']].set_visible(False)  # 隐藏上边框和右边框

    # 关键修改：强制 y 轴负号为 U+2212
    ax1.yaxis.set_major_formatter(FuncFormatter(unicode_minus_formatter))  # 强制纵轴使用数学负号 U+2212

    # ── Subplot 2: Temperature-Humidity Index (THI) ──────────────────────
    ax2 = fig.add_subplot(gs[1])  # 取得第二个子图，绘制 THI
    ymax_fill = max(res['THI实际'].max(), res['THI预测'].max(), 85)  # 计算 THI 填充的最大高度

    ax2.fill_between(steps, 0, 70, alpha=0.06, color=C_ZONE0)  # 填充舒适区间背景
    ax2.fill_between(steps, 70, 76, alpha=0.12, color=C_ZONE1)  # 填充警告区间背景
    ax2.fill_between(steps, 76, ymax_fill, alpha=0.12, color=C_ZONE2)  # 填充危险及以上区间背景

    ax2.axhline(70, color=C_THI_L1, ls=':', lw=1.2, label='Comfort limit (70)', alpha=0.85)  # 在 70 处画舒适上限虚线
    ax2.axhline(76, color=C_THI_L2, ls=':', lw=1.2, label='Warning limit (76)', alpha=0.85)  # 在 76 处画警告上限虚线
    ax2.axhline(81, color=C_THI_L3, ls=':', lw=1.2, label='Danger limit (81)', alpha=0.85)  # 在 81 处画危险上限虚线

    ax2.plot(steps, res['THI实际'], color=C_ACTUAL, lw=1.5, label='Actual THI', alpha=0.9)  # 绘制实际 THI 曲线
    ax2.plot(steps, res['THI预测'], color=C_PRED, lw=1.5, label='Predicted THI', alpha=0.9, ls='--')  # 绘制预测 THI 曲线，使用虚线

    ax2.set_ylabel('THI', fontsize=9)  # 设置 THI 子图纵轴标签
    ax2.set_title('(b) THI prediction and risk assessment',  # 设置 THI 子图标题
                  fontsize=10, fontweight='bold')  # 标题字号与加粗设置
    ax2.legend(**_leg_kw)  # 应用统一图例参数
    ax2.set_xticks(ticks)  # 设置横轴刻度位置
    ax2.set_xticklabels(tick_labels, rotation=35, ha='right')  # 设置横轴刻度标签并旋转
    ax2.grid(True, alpha=0.25, linestyle='--', color='#AAAAAA')  # 添加网格
    ax2.set_xlim(steps[0], steps[-1])  # 限定横轴范围
    ax2.spines[['top', 'right']].set_visible(False)  # 隐藏上边框和右边框

    # 保险处理：若 THI 坐标轴出现负数，也使用 U+2212
    ax2.yaxis.set_major_formatter(FuncFormatter(unicode_minus_formatter))  # 若 THI 坐标轴出现负数也使用数学负号

    # ── Subplot 3: TD Status strip ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])  # 取得第三个子图，绘制 TD 状态条带
    colors_td = {  # 定义 TD 等级到颜色的映射字典
        0: C_TD_NORMAL,  # 正常对应浅蓝
        1: C_TD_HOT,  # 偏热对应红色
        -1: C_TD_COLD  # 偏冷对应蓝色
    }  # 字典定义结束

    td_levels = res['TD等级'].values  # 取出 TD 等级数组
    for lvl, col in colors_td.items():  # 遍历等级与颜色的映射
        mask = td_levels == lvl  # 筛选出当前等级对应的样本掩码
        if mask.any():  # 仅对存在的等级绘制
            ax3.bar(  # 绘制状态条带柱状图
                steps[mask],  # 横轴为当前等级对应的步数
                1,  # 柱高固定为 1
                width=1.0,  # 柱宽为 1
                color=col,  # 使用对应状态颜色
                alpha=0.9,  # 透明度
                edgecolor='none',  # 不绘制柱边框
                linewidth=0  # 线宽为 0
            )  # bar 调用结束

    ax3.legend(  # 为状态条带添加自定义图例
        handles=[  # 图例句柄列表开始
            mpatches.Patch(facecolor=C_TD_NORMAL, label='Normal', edgecolor='none'),  # 正常状态色块
            mpatches.Patch(facecolor=C_TD_HOT, label='Overheated', edgecolor='none'),  # 偏热状态色块
            mpatches.Patch(facecolor=C_TD_COLD, label='Overcooled', edgecolor='none'),  # 偏冷状态色块
        ],  # 图例句柄列表结束
        **_leg_kw  # 应用统一图例参数
    )  # legend 调用结束

    ax3.set_yticks([])  # 隐藏纵轴刻度
    ax3.set_ylabel('TD Warning Status', fontsize=9)  # 设置纵轴标签
    ax3.set_title('(c) TD status timeline', fontsize=10, fontweight='bold')  # 设置子图标题
    ax3.set_xticks(ticks)  # 设置横轴刻度位置
    ax3.set_xticklabels(tick_labels, rotation=35, ha='right')  # 设置横轴刻度标签并旋转
    ax3.set_xlim(steps[0] - 1, steps[-1] + 1)  # 横轴范围左右各留一步
    ax3.grid(True, alpha=0.2, axis='x', linestyle='--', color='#AAAAAA')  # 仅添加横向网格
    ax3.spines[['top', 'right', 'left']].set_visible(False)  # 隐藏上、右、左边框

    # ── Subplot 4: THI Risk Level strip ─────────────────────────────────
    ax4 = fig.add_subplot(gs[3])  # 取得第四个子图，绘制 THI 风险等级条带
    colors_thi = {  # 定义 THI 等级到颜色的映射字典
        0: C_THI_0,  # 等级 0 对应浅蓝
        1: C_THI_1,  # 等级 1 对应浅橙
        2: C_THI_2,  # 等级 2 对应橙色
        3: C_THI_3  # 等级 3 对应深红
    }  # 字典定义结束

    thi_levels = res['THI等级'].values  # 取出 THI 等级数组
    for lvl, col in colors_thi.items():  # 遍历等级与颜色的映射
        mask = thi_levels == lvl  # 筛选当前等级对应的样本掩码
        if mask.any():  # 仅对存在的等级绘制
            ax4.bar(  # 绘制风险等级条带
                steps[mask],  # 横轴为当前等级对应的步数
                1,  # 柱高固定为 1
                width=1.0,  # 柱宽为 1
                color=col,  # 使用对应等级颜色
                alpha=0.9,  # 透明度
                edgecolor='none',  # 不绘制柱边框
                linewidth=0  # 线宽为 0
            )  # bar 调用结束

    ax4.legend(  # 为风险条带添加自定义图例
        handles=[  # 图例句柄列表开始
            mpatches.Patch(facecolor=C_THI_0, label='Comfort (<70)', edgecolor='none'),  # 舒适等级色块
            mpatches.Patch(facecolor=C_THI_1, label='Warning (70–76)', edgecolor='none'),  # 警告等级色块
            mpatches.Patch(facecolor=C_THI_2, label='Danger (76–81)', edgecolor='none'),  # 危险等级色块
            mpatches.Patch(facecolor=C_THI_3, label='Emergency (≥81)', edgecolor='none'),  # 紧急等级色块
        ],  # 图例句柄列表结束
        **_leg_kw  # 应用统一图例参数
    )  # legend 调用结束

    ax4.set_yticks([])  # 隐藏纵轴刻度
    ax4.set_ylabel('THI Risk Level', fontsize=9)  # 设置纵轴标签
    ax4.set_title('(d) THI risk level timeline', fontsize=10, fontweight='bold')  # 设置子图标题
    ax4.set_xticks(ticks)  # 设置横轴刻度位置
    ax4.set_xticklabels(tick_labels, rotation=35, ha='right')  # 设置横轴刻度标签并旋转
    ax4.set_xlabel('Date and Time (MM-DD HH:MM)', fontsize=9)  # 设置横轴标签
    ax4.set_xlim(steps[0] - 1, steps[-1] + 1)  # 横轴范围左右各留一步
    ax4.grid(True, alpha=0.2, axis='x', linestyle='--', color='#AAAAAA')  # 仅添加横向网格
    ax4.spines[['top', 'right', 'left']].set_visible(False)  # 隐藏上、右、左边框

    plt.tight_layout(rect=[0, 0, 0.80, 0.99])  # 调整整体布局，为右侧图例留出空间

    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')  # 保存 PNG 格式结果图

    tiff_path = output_path.replace('.png', '.tiff')  # 把输出路径后缀替换为 tiff 得到 TIFF 路径
    plt.savefig(tiff_path, dpi=600, bbox_inches='tight', format='tiff')  # 保存 TIFF 格式结果图

    plt.close()  # 关闭画布释放内存

    print(f'  [OK] Figure saved: {output_path}')  # 打印 PNG 保存成功提示
    print(f'  [OK] Figure saved: {tiff_path}')  # 打印 TIFF 保存成功提示


# ============================================================
# 主程序
# ============================================================
def main():  # 定义主程序函数
    """
    主程序：完成数据加载、模型加载、特征工程、批量推理、预警判断与结果输出。
    无需外部参数；返回包含逐步预测与预警明细的 DataFrame。
    """
    print('=' * 70)  # 打印 70 个等号作为分隔线
    print('鸡舍环境实时滚动预测与双指标预警系统 v2')  # 打印系统名称标题
    print('=' * 70)  # 再次打印分隔线

    data_file = r'd:\resources\pythonTest\Total - 副本 - 副本\最后三天测试.csv'  # 定义测试数据文件路径
    td_model_file = os.path.join('温度偏差预测', 'gru+小波+eca', 'Model_GRUWavelet_no_leakage_pytorch.pth')  # 拼接 TD 模型权重文件路径
    thi_model_file = os.path.join('THI', 'Model_GRUWavelet_no_leakage_pytorch.pth')  # 拼接 THI 模型权重文件路径

    print('\n[1] 加载数据...')  # 打印第一步加载数据提示
    df = load_data(data_file)  # 调用 load_data 读取测试数据
    print(f'  时间范围: {df["时间戳"].iloc[0]} ~ {df["时间戳"].iloc[-1]}')  # 打印数据的时间范围
    print(f'  日龄: {sorted(df["日龄"].unique())}')  # 打印数据中包含的日龄集合

    print('\n[2] 加载模型...')  # 打印第二步加载模型提示
    td_model, td_ck = load_model(td_model_file, 'TD模型')  # 加载 TD 模型及其检查点
    thi_model, thi_ck = load_model(thi_model_file, 'THI模型')  # 加载 THI 模型及其检查点

    time_step = td_ck['time_step']  # 从 TD 检查点取出输入时间窗口长度
    pred_offset = 6  # 设定预测偏移为 6 步

    print('\n[3] 特征工程...')  # 打印第三步特征工程提示

    td_feats = [  # 构造 TD 特征列表的推导式开始
        f for f in td_ck.get(  # 从检查点取原始特征名，缺省使用默认列表
            'original_features',  # 默认 TD 特征列表
            ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 默认 TD 特征列表续
        )  # 仅保留数据中存在的特征
        if f in df.columns  # TD 特征列表推导式结束
    ]

    thi_feats = [  # 构造 THI 特征列表的推导式开始
        f for f in thi_ck.get(  # 从检查点取原始特征名，缺省使用默认列表
            'original_features',  # 默认 THI 特征列表
            ['日龄', 'THI', '当前温度', '目标温度']  # 默认 THI 特征列表续
        )  # 仅保留数据中存在的特征
        if f in df.columns  # THI 特征列表推导式结束
    ]

    print(f'  TD特征: {td_feats}')  # 打印最终使用的 TD 特征
    print(f'  THI特征: {thi_feats}')  # 打印最终使用的 THI 特征

    print('  小波分解中...')  # 打印正在进行小波分解的提示

    td_wdf = make_wavelet_features(  # 对 TD 特征做小波分解，参数取检查点中的配置
        df[td_feats],  # 传入 TD 特征数据
        td_ck.get('wavelet_target_col', '温度偏差'),  # 指定 TD 小波分解的目标列
        td_ck.get('wavelet', 'db4'),  # 指定 TD 小波基函数
        td_ck.get('level', 3),  # 指定 TD 分解层数
        td_ck.get('wavelet_window', 64)  # 指定 TD 小波窗口长度
    )  # TD 小波分解调用结束

    thi_wdf = make_wavelet_features(  # 对 THI 特征做小波分解，参数取检查点中的配置
        df[thi_feats],  # 传入 THI 特征数据
        thi_ck.get('wavelet_target_col', 'THI'),  # 指定 THI 小波分解的目标列
        thi_ck.get('wavelet', 'db4'),  # 指定 THI 小波基函数
        thi_ck.get('level', 3),  # 指定 THI 分解层数
        thi_ck.get('wavelet_window', 64)  # 指定 THI 小波窗口长度
    )  # THI 小波分解调用结束

    td_fsc = MinMaxScaler().fit(td_wdf)  # 拟合 TD 特征归一化器
    thi_fsc = MinMaxScaler().fit(thi_wdf)  # 拟合 THI 特征归一化器
    td_tsc = MinMaxScaler().fit(df[['温度偏差']])  # 拟合 TD 目标归一化器
    thi_tsc = MinMaxScaler().fit(df[['THI']])  # 拟合 THI 目标归一化器

    td_norm = pd.DataFrame(td_fsc.transform(td_wdf), columns=td_wdf.columns)  # 用 TD 归一化器转换小波特征并重建 DataFrame
    thi_norm = pd.DataFrame(thi_fsc.transform(thi_wdf), columns=thi_wdf.columns)  # 用 THI 归一化器转换小波特征并重建 DataFrame

    print(f'  TD特征维度: {td_norm.shape[1]}  THI特征维度: {thi_norm.shape[1]}')  # 打印两种特征的维度

    print('\n[4] 批量推理...')  # 打印第四步批量推理提示

    N = len(df)  # 记录数据总行数
    max_steps = N - time_step - pred_offset + 1  # 计算在给定窗口和偏移下可预测的步数

    print(f'  数据行数: {N}  可预测步数: {max_steps}')  # 打印数据行数与可预测步数
    print(f'  输入窗口: {time_step}步  预测范围: {pred_offset}步')  # 打印输入窗口与预测范围

    td_preds = batch_predict(td_model, td_norm, time_step)  # 用 TD 模型做批量推理
    thi_preds = batch_predict(thi_model, thi_norm, time_step)  # 用 THI 模型做批量推理

    td_preds = td_tsc.inverse_transform(td_preds.reshape(-1, 1)).flatten()  # 把 TD 预测反归一化回原始量纲
    thi_preds = thi_tsc.inverse_transform(thi_preds.reshape(-1, 1)).flatten()  # 把 THI 预测反归一化回原始量纲

    td_reals = df['温度偏差'].iloc[  # 按预测偏移对齐并取出 TD 真实值序列
        time_step + pred_offset - 1:  # 起始下标为窗口长度加偏移减一
        time_step + pred_offset - 1 + max_steps  # 截取长度为可预测步数
    ].values  # 转为 numpy 数组

    thi_reals = df['THI'].iloc[  # 按预测偏移对齐并取出 THI 真实值序列
        time_step + pred_offset - 1:  # 起始下标为窗口长度加偏移减一
        time_step + pred_offset - 1 + max_steps  # 截取长度为可预测步数
    ].values  # 转为 numpy 数组

    cur_times = df['时间戳'].iloc[  # 按输入窗口对齐取出对应的当前时间
        time_step - 1:  # 起始下标为窗口长度减一
        time_step - 1 + max_steps  # 截取长度为可预测步数
    ].values  # 转为 numpy 数组

    pred_times = df['时间戳'].iloc[  # 按预测偏移对齐取出预测时刻对应的时间
        time_step + pred_offset - 1:  # 起始下标为窗口长度加偏移减一
        time_step + pred_offset - 1 + max_steps  # 截取长度为可预测步数
    ].values  # 转为 numpy 数组

    ages = df['日龄'].iloc[  # 取出每个样本对应的日龄
        time_step - 1:  # 起始下标为窗口长度减一
        time_step - 1 + max_steps  # 截取长度为可预测步数
    ].values  # 转为 numpy 数组

    target_temps = df['目标温度'].iloc[  # 取出每个样本对应的目标温度
        time_step - 1:  # 起始下标为窗口长度减一
        time_step - 1 + max_steps  # 截取长度为可预测步数
    ].values  # 转为 numpy 数组

    print('\n[5] 预警判断...')  # 打印第五步预警判断提示

    records = []  # 初始化保存逐条预警记录的列表

    for i in range(max_steps):  # 遍历每一个可预测步
        td_p = round(float(td_preds[i]), 1)  # 把 TD 预测值四舍五入保留一位小数
        td_r = round(float(td_reals[i]), 1)  # 把 TD 真实值四舍五入保留一位小数
        thi_p = round(float(thi_preds[i]), 3)  # 把 THI 预测值四舍五入保留三位小数
        thi_r = round(float(thi_reals[i]), 3)  # 把 THI 真实值四舍五入保留三位小数

        age = int(ages[i])  # 把日龄转为整数
        tgt_temp = float(target_temps[i])  # 把目标温度转为浮点数

        td_state, td_level, hot_thr, cold_thr = classify_td(td_p, tgt_temp, age)  # 调用 classify_td 判断 TD 状态、等级与阈值
        thi_state, thi_level = classify_thi(thi_p)  # 调用 classify_thi 判断 THI 状态与等级

        triggered = (td_level != 0) or (thi_level >= 1)  # 任一指标触发即认为触发预警

        warn_types = []  # 初始化保存预警类型的列表

        if td_level > 0:  # 若 TD 等级为正表示偏热
            warn_types.append('偏热预警')  # 追加偏热预警类型
        elif td_level < 0:  # 若 TD 等级为负表示偏冷
            warn_types.append('偏冷预警')  # 追加偏冷预警类型

        if thi_level == 1:  # 若 THI 等级为 1
            warn_types.append('THI警告')  # 追加 THI 警告类型
        elif thi_level == 2:  # 若 THI 等级为 2
            warn_types.append('THI危险')  # 追加 THI 危险类型
        elif thi_level == 3:  # 若 THI 等级为 3
            warn_types.append('THI紧急')  # 追加 THI 紧急类型

        records.append({  # 把当前步的详细信息组装成字典并追加到记录列表
            'step': i,  # 当前步序号
            '当前时间': pd.Timestamp(cur_times[i]),  # 当前时刻
            '预测时间': pd.Timestamp(pred_times[i]),  # 预测对应的时刻
            '日龄': age,  # 日龄
            '目标温度': round(tgt_temp, 3),  # 目标温度
            'TD预测': td_p,  # TD 预测值
            'TD实际': td_r,  # TD 实际值
            'THI预测': thi_p,  # THI 预测值
            'THI实际': thi_r,  # THI 实际值
            'TD偏热阈值': round(hot_thr, 3),  # TD 偏热阈值
            'TD偏冷阈值': round(cold_thr, 3),  # TD 偏冷阈值
            'TD状态': td_state,  # TD 状态文字
            'THI状态': thi_state,  # THI 状态文字
            'TD等级': td_level,  # TD 等级数值
            'THI等级': thi_level,  # THI 等级数值
            '触发预警': triggered,  # 是否触发预警
            '预警类型': ' | '.join(warn_types) if warn_types else '正常',  # 预警类型文字，无预警时记为正常
        })  # 字典与记录追加结束

    res = pd.DataFrame(records)  # 把记录列表转换为 DataFrame

    print(f'  [OK] 预测完成，共 {len(res)} 步')  # 打印预测完成提示与总步数

    print('\n[6] 输出结果...')  # 打印第六步输出结果提示

    total = len(res)  # 统计总步数
    warn_count = res['触发预警'].sum()  # 统计触发预警的步数

    print(f'  总步数: {total}  触发预警: {warn_count} ({warn_count / total * 100:.1f}%)')  # 打印总步数、预警步数与占比
    print(f'  TD预警 - 偏热: {(res["TD状态"] == "偏热").sum()} 偏冷: {(res["TD状态"] == "偏冷").sum()}')  # 打印 TD 偏热与偏冷预警数量
    print(  # 打印 THI 各风险等级数量统计开始
        f'  THI分级 - 舒适: {(res["THI状态"] == "舒适").sum()} '  # 舒适与警告等级的数量
        f'警告: {(res["THI状态"] == "警告").sum()} '  # 危险与紧急等级的数量
        f'危险: {(res["THI状态"] == "危险").sum()} '  # 危险等级数量
        f'紧急: {(res["THI状态"] == "紧急").sum()}'  # 紧急等级数量
    )  # 打印语句结束

    out_img = 'warning_simulation_result.png'  # 定义结果图输出文件名
    plot_results(res, out_img)  # 调用 plot_results 绘制并保存结果图

    out_csv = 'warning_simulation_log.csv'  # 定义日志 CSV 输出文件名
    res.to_csv(out_csv, index=False, encoding='utf-8-sig')  # 把结果保存为带 BOM 的 UTF-8 CSV

    print(f'  [OK] 日志已保存: {out_csv}')  # 打印日志保存成功提示

    print('\n' + '=' * 70)  # 打印运行结束分隔线
    print('运行完成！')  # 打印运行完成提示
    print('=' * 70)  # 再次打印分隔线

    return res  # 返回结果 DataFrame


if __name__ == '__main__':  # 仅当脚本作为主程序运行时执行
    main()  # 调用主程序