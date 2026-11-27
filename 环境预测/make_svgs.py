# -*- coding: utf-8 -*-
"""Wavelet-ECA-GRU 论文配图批量生成脚本。

本脚本用纯 Python 字符串拼接 SVG 图元，生成两张矢量图并保存到脚本所在目录：
1. framework_diagram.svg（1580x940）：六栏式的模型框架图，展示从数据采集、预处理、
   小波分解、ECA 注意力、GRU 编码器到预测预警的完整流程，并附模型对比表；
2. abstract_diagram.svg（1200x700）：研究总览图，包含背景、数据流、模型流水线、
   实验结果与结论等内容。

脚本提供 R/T/L/card/wrap_svg 等图元辅助函数，所有 SVG 模板文本（DEFS 与三引号字符串）
保持原样，注释只添加在字符串之外。
"""

import os  # 导入 os 模块，用于获取脚本目录并拼接输出文件路径

OUT = os.path.dirname(os.path.abspath(__file__))  # 将输出目录设为脚本自身所在目录（绝对路径）

CB  = '#3b82f6'  # 蓝色主题色（数据采集）
CB2 = '#2563eb'  # 深蓝色主题色（预处理）
CG  = '#059669'  # 绿色主题色（小波分解）
CO  = '#d97706'  # 橙色主题色（ECA 注意力）
CP  = '#7c3aed'  # 紫色主题色（GRU 编码器）
CR  = '#dc2626'  # 红色主题色（预测预警）

# 以下三引号字符串是 SVG 公共定义（字体样式、五种箭头标记与阴影滤镜），不在其内部插入注释
DEFS = """<defs>
<style>.fn{font-family:'Segoe UI',Arial,sans-serif;}</style>
<marker id='ab' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#3b82f6'/></marker>
<marker id='ag' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#059669'/></marker>
<marker id='ao' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#d97706'/></marker>
<marker id='ap' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#7c3aed'/></marker>
<marker id='ar' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#dc2626'/></marker>
<filter id='sh'><feDropShadow dx='1' dy='2' stdDeviation='3' flood-color='#00000018'/></filter>
</defs>"""


def R(x, y, w, h, fill, stroke='none', sw=1.5, rx=7, ex=''):
    """生成一个 SVG 矩形元素字符串。

    参数:
        x, y: 左上角坐标；w, h: 宽高；fill: 填充色。
        stroke: 边框色；sw: 线宽；rx: 圆角半径；ex: 追加的额外属性。
    返回:
        <rect .../> 字符串。
    """
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}' {ex}/>"


def T(x, y, s, sz=10, fill='#334155', anchor='middle', bold=False, italic=False):
    """生成 SVG 文本元素字符串，并对文本内容做 XML 转义。

    参数:
        x, y: 文本坐标；s: 文本内容；sz: 字号；fill: 颜色。
        anchor: 对齐方式；bold/italic: 加粗/斜体。
    返回:
        <text ...>...</text> 字符串。
    """
    s = str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')  # 转义 & < > 防止破坏 XML 结构
    fw = 'bold' if bold else 'normal'  # 根据 bold 参数决定 font-weight
    fi = 'italic' if italic else 'normal'  # 根据 italic 参数决定 font-style
    return (f"<text x='{x}' y='{y}' class='fn' font-size='{sz}' fill='{fill}' "  # 拼接文本元素的属性部分
            f"text-anchor='{anchor}' font-weight='{fw}' font-style='{fi}'>{s}</text>")  # 拼接对齐、字重、样式与文本内容


def L(x1, y1, x2, y2, c='#64748b', sw=1.8, mk=''):
    """生成 SVG 连线元素字符串，可带箭头标记。

    参数:
        x1, y1, x2, y2: 起止坐标；c: 线条颜色；sw: 线宽；mk: 箭头标记 id。
    返回:
        <line .../> 字符串。
    """
    m = f" marker-end='url(#{mk})'" if mk else ''  # 若指定箭头则追加 marker-end 属性
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{c}' stroke-width='{sw}'{m}/>"


def card(x, y, w, h, color, title, rows):
    """rows: list of (text, sz, bold, italic, fill)

    生成带主题色标题栏与多行文字的信息卡片。
    参数:
        x, y, w, h: 位置与尺寸；color: 主题色；title: 标题。
        rows: 每行 (文本, 字号, 加粗, 斜体, 颜色) 的元组列表。
    返回:
        由多个 SVG 元素换行拼接而成的卡片字符串。
    """
    o = []  # 初始化图元列表
    o.append(R(x, y, w, h, '#fff', color, 1.5, 7, "filter='url(#sh)'"))  # 绘制带阴影的白色卡片底
    o.append(R(x, y, w, 24, color, color, 0, 7))  # 绘制顶部标题栏
    o.append(R(x, y+12, w, 12, color, color, 0, 0))  # 补齐标题栏圆角
    o.append(T(x+w//2, y+15, title, 10, '#fff', bold=True))  # 居中绘制白色标题
    for i, (s, sz, bo, it, fc) in enumerate(rows):  # 遍历每行文字配置
        o.append(T(x+w//2, y+33+i*15, s, sz, fc, bold=bo, italic=it))  # 按行号递增 y 坐标绘制文字
    return '\n'.join(o)  # 以换行拼接所有图元并返回


def wrap_svg(body, w, h):
    """把图元内容包裹为完整的 SVG 文档字符串。

    参数:
        body: 已拼好的 SVG 内部图元字符串。
        w, h: 画布宽高，用于 viewBox 与 width/height 属性。
    返回:
        包含 XML 声明、公共 DEFS、body 与闭合标签的完整 SVG 字符串。
    """
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'  # XML 声明
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'  # SVG 根元素与画布尺寸
            f'{DEFS}\n{body}\n</svg>')  # 依次写入公共定义、图元内容与闭合标签


# ==============================================================
# FRAMEWORK DIAGRAM
# ==============================================================
def build_framework():
    """构建六栏式框架图 SVG 的图元内容。

    返回:
        由所有图元字符串以换行拼接而成的框架图内容（不含 XML 声明与 <svg> 外壳）。
    """
    W, H = 1580, 940  # 定义框架图画布宽高
    E = []  # 初始化图元列表
    E.append(R(0, 0, W, H, '#f8fafc', 'none', 0, 0))  # 绘制整图浅灰背景

    # Title
    E.append(R(28, 12, W-56, 54, '#1e293b', 'none', 0, 8, "filter='url(#sh)'"))  # 绘制深色标题栏背景
    E.append(T(W//2, 36, 'Wavelet-ECA-GRU Framework: Broiler House Temperature Deviation Short-term Prediction', 17, '#f1f5f9', bold=True))  # 绘制主标题
    E.append(T(W//2, 56, 'Input L=36 steps (180 min) -> Predict H=6 ahead (30 min)  |  TD=T_indoor-T_target  |  5-min step  |  9,123 samples', 10, '#94a3b8'))  # 绘制副标题说明

    # Column configs: (x, w, color, bg, border, label)
    cols = [  # 六个栏目的配置列表
        (28,  215, CB,  '#eff6ff', '#bfdbfe', '(1) Data Collection'),  # 第 1 栏：数据采集
        (255, 215, CB2, '#eff6ff', '#bfdbfe', '(2) Preprocessing'),  # 第 2 栏：预处理
        (482, 225, CG,  '#ecfdf5', '#a7f3d0', '(3) Wavelet DWT'),  # 第 3 栏：小波分解
        (719, 215, CO,  '#fffbeb', '#fde68a', '(4) ECA Attention'),  # 第 4 栏：ECA 注意力
        (946, 215, CP,  '#f5f3ff', '#ddd6fe', '(5) GRU Encoder'),  # 第 5 栏：GRU 编码器
        (1173,379, CR,  '#fff1f2', '#fecdd3', '(6) Prediction & Warning'),  # 第 6 栏：预测与预警
    ]
    for cx, cw, cc, bg, bc, lbl in cols:  # 遍历栏目配置逐栏绘制
        E.append(R(cx, 78, cw, 848, bg, bc, 1.2, 10))  # 绘制栏目背景与边框
        E.append(R(cx, 78, cw, 30, cc, 'none', 0, 10))  # 绘制栏目标题条
        E.append(R(cx, 90, cw, 18, cc, 'none', 0, 0))  # 补齐标题条圆角
        E.append(T(cx+cw//2, 98, lbl, 11, '#fff', bold=True))  # 居中绘制栏目名称

    # --- COL 1: Data Collection ---
    E.append(card(42, 122, 187, 130, CB, 'Inspection Robot', [  # 绘制"巡检机器人"卡片
        ('Temperature / Humidity',       9.5, False, False, '#1e3a8a'),  # 采集温湿度
        ('Wind Speed (m/s)',              9.5, False, False, '#1e3a8a'),  # 采集风速
        ('CO2 Concentration (ppm)',       9.5, False, False, '#1e3a8a'),  # 采集 CO2 浓度
        ('Sample rate: 1 min',           9.5, False, False, '#1e3a8a'),  # 采样频率 1 分钟
        ('4G wireless upload to server', 9.5, False, False, '#1e3a8a'),  # 4G 上传服务器
    ]))
    E.append(card(42, 268, 187, 125, CB, 'Control Terminal', [  # 绘制"控制终端"卡片
        ('Age of flock (Day)',            9.5, False, False, '#1e3a8a'),  # 记录日龄
        ('Target Temperature (T_target)', 9.5, False, False, '#1e3a8a'),  # 记录目标温度
        ('Outdoor Temperature (T_out)',   9.5, False, False, '#1e3a8a'),  # 记录室外温度
        ('Negative Pressure / State',    9.5, False, False, '#1e3a8a'),  # 记录负压/状态
        ('1-min sync to server',         9.5, False, False, '#1e3a8a'),  # 每分钟同步
    ]))
    E.append(card(42, 408, 187, 118, CB, 'Sensor Specs', [  # 绘制"传感器规格"卡片
        ('Temp: +/-0.2C  Range:-20~50C', 9, False, False, '#334155'),  # 温度精度与量程
        ('Humidity: +/-2% RH',           9, False, False, '#334155'),  # 湿度精度
        ('Wind: +/-(0.2+0.02v) m/s',    9, False, False, '#334155'),  # 风速精度
        ('CO2: +/-50 ppm (300~5000)',    9, False, False, '#334155'),  # CO2 精度与量程
        ('Period: 2025/5/26 - 4/30',    9, False, False, '#334155'),  # 采集时段
    ]))
    E.append(card(42, 540, 187, 108, CB, 'Coop Info', [  # 绘制"鸡舍信息"卡片
        ('101m x 16m | 30,000+ birds',  9, False, False, '#334155'),  # 尺寸与养殖量
        ('7 rows x 4-layer cages',      9, False, False, '#334155'),  # 笼架排布
        ('24 large + 2 small fans',     9, False, False, '#334155'),  # 风机配置
        ('Longitudinal ventilation',    9, False, False, '#334155'),  # 通风方式
        ('Evap. cooling + wet curtains',9, False, False, '#334155'),  # 水帘降温
    ]))
    for ya in [185, 330]:  # 在两个高度绘制第 1 栏到第 2 栏的箭头
        E.append(L(241, ya, 252, ya, CB, 1.8, 'ab'))  # 绘制蓝色箭头

    # --- COL 2: Preprocessing ---
    E.append(card(269, 122, 187, 88, CB2, 'TD Definition', [  # 绘制"TD 定义"卡片
        ('TD = T_indoor - T_target',      10.5, False, True,  '#1e3a8a'),  # TD 为室内温度减目标温度
        ('Temperature Deviation (deg C)', 9,    False, False, '#334155'),  # 含义为温度偏差
        ('Core prediction target',        9,    False, False, '#334155'),  # 核心预测目标
        ('Non-stationary time series',    9,    False, False, '#334155'),  # 非平稳时间序列
    ]))
    E.append(card(269, 224, 187, 108, CB2, 'Align & Resample', [  # 绘制"对齐与重采样"卡片
        ('Timestamp alignment (server)', 9, False, False, '#334155'),  # 时间戳对齐
        ('1 min -> 5 min resampling',    9, False, False, '#334155'),  # 5 分钟重采样
        ('Total: 9,123 samples',         9, False, False, '#334155'),  # 样本总数
        ('Train:Test = 80:20 split',     9, False, False, '#334155'),  # 数据集划分
        ('No leakage across split',      9, False, False, '#334155'),  # 无数据泄漏
    ]))
    E.append(card(269, 346, 187, 90, CB2, 'Missing Value Fill', [  # 绘制"缺失值填充"卡片
        ('Continuous: linear interp.',   9, False, False, '#334155'),  # 连续变量线性插值
        ('+ forward / backward fill',   9, False, False, '#334155'),  # 前后填充
        ('Discrete: fwd/bwd fill',      9, False, False, '#334155'),  # 离散变量前后填充
        ('Drop remaining NaN rows',     9, False, False, '#334155'),  # 丢弃剩余缺失行
    ]))
    E.append(card(269, 450, 187, 108, CB2, 'Pearson Feature Select', [  # 绘制"Pearson 特征选择"卡片
        ('TD+: Age(+), T_out(+)',        9, False, False, '#334155'),  # 正相关特征
        ('TD-: CO2(-), T_tgt(-), RH(-)', 9, False, False, '#334155'),  # 负相关特征
        ('Selected 6 features:',         9, True,  False, '#1d4ed8'),  # 加粗提示所选特征
        ('TD, Age, T_out, CO2, T_tgt, RH', 9, False, False, '#334155'),  # 特征清单
        ('C=9 channels (TD->4 wavelet)', 9, False, False, '#334155'),  # 小波扩展后 9 通道
    ]))
    E.append(card(269, 572, 187, 80, CB2, 'Min-Max Normalize', [  # 绘制"最小-最大归一化"卡片
        ("x'=(x-xmin)/(xmax-xmin)",     9.5, False, True,  '#1e3a8a'),  # 归一化公式
        ('Fit on train set -> [0,1]',    9,   False, False, '#334155'),  # 训练集拟合
        ('Same scaler on test set',      9,   False, False, '#334155'),  # 测试集沿用缩放器
        ('TD target also scaled',        9,   False, False, '#334155'),  # 目标值也缩放
    ]))
    E.append(card(269, 666, 187, 88, CB2, 'Sliding Window', [  # 绘制"滑动窗口"卡片
        ('Input X: L=36 steps (180min)', 9, False, False, '#334155'),  # 输入 36 步窗口
        ('Target y: t+H, H=6 (30min)',   9, False, False, '#334155'),  # 目标未来第 6 步
        ('Stride=1, online aligned',     9, False, False, '#334155'),  # 步长为 1
        ('Supervised learning setup',    9, False, False, '#334155'),  # 监督学习样本
    ]))
    for ya in [168, 608]:  # 在两个高度绘制第 2 栏到第 3 栏的箭头
        E.append(L(469, ya, 479, ya, CG, 1.8, 'ag'))  # 绘制绿色箭头

    # --- COL 3: Wavelet DWT ---
    E.append(card(494, 122, 197, 108, CG, 'DWT Settings', [  # 绘制"DWT 设置"卡片
        ('Wavelet: db4 (Daubechies 4)',   9, False, False, '#334155'),  # 使用 db4 小波基
        ('Decomposition levels: L=3',    9, False, False, '#334155'),  # 分解层数 3
        ('Causal window: W=36 steps',    9, False, False, '#334155'),  # 因果窗口 36 步
        ('No future data leakage',       9, False, False, '#334155'),  # 无未来数据泄漏
        ('Split data before applying DWT', 9, False, False, '#334155'),  # 先划分数据集
    ]))
    # Decomp tree
    dx, dy, dw = 494, 244, 197  # 定义分解树的 x、y 与宽度
    E.append(R(dx, dy, dw, 200, '#fff', CG, 1.5, 7, "filter='url(#sh)'"))  # 绘制分解树白色背景框
    E.append(R(dx, dy, dw, 24, CG, CG, 0, 7))  # 绘制标题栏
    E.append(R(dx, dy+12, dw, 12, CG, CG, 0, 0))  # 补齐标题栏圆角
    E.append(T(dx+dw//2, dy+15, 'Decomposition Tree', 10, '#fff', bold=True))  # 绘制标题文字
    E.append(R(dx+24, dy+32, dw-48, 22, '#d1fae5', CG, 1, 4))  # 绘制 TD 信号输入框
    E.append(T(dx+dw//2, dy+46, 'TD Signal [t-35...t]', 9, '#065f46', bold=True))  # 标注信号窗口
    E.append(L(dx+dw//2, dy+54, dx+dw//2, dy+64, CG, 1.5, 'ag'))  # 绘制到分解层的绿色箭头
    E.append(R(dx+8, dy+66, 86, 22, '#6ee7b7', CG, 1, 4))  # 绘制 A3 近似分量框
    E.append(T(dx+51, dy+80, 'A3 (trend)', 9, '#064e3b', bold=True))  # 标注 A3 为趋势分量
    E.append(R(dx+100, dy+66, 89, 22, '#a7f3d0', CG, 1, 4))  # 绘制 D1/D2/D3 细节分量框
    E.append(T(dx+144, dy+80, 'D1, D2, D3', 9, '#064e3b', bold=True))  # 标注细节分量
    E.append(L(dx+51,  dy+88, dx+51,  dy+100, CG, 1))  # 绘制 A3 到拼接框的连线
    E.append(L(dx+144, dy+88, dx+144, dy+100, CG, 1))  # 绘制细节分量到拼接框的连线
    E.append(R(dx+24, dy+100, dw-48, 22, '#ecfdf5', CG, 1, 4))  # 绘制小波系数拼接框
    E.append(T(dx+dw//2, dy+114, 'w_t = [A3, D1, D2, D3]', 9, '#065f46', bold=True, italic=True))  # 标注拼接后的小波向量
    E.append(L(dx+dw//2, dy+122, dx+dw//2, dy+134, CG, 1.5, 'ag'))  # 绘制到下一步的绿色箭头
    E.append(R(dx+24, dy+134, dw-48, 22, '#d1fae5', CG, 1, 4))  # 绘制最终特征拼接框
    E.append(T(dx+dw//2, dy+148, 'concat w/ 5 other vars -> C=9', 9, '#065f46', bold=True))  # 说明拼接后共 9 个通道
    E.append(R(dx+24, dy+162, dw-48, 22, '#f0fdf4', CG, 1, 4))  # 绘制因果性说明框
    E.append(T(dx+dw//2, dy+176, 'Causal: only [t-35...t] used', 8, '#166534', italic=True))  # 强调仅使用当前及历史窗口
    E.append(T(dx+dw//2, dy+190, 'No leakage guaranteed', 8, '#166534', italic=True))  # 强调无数据泄漏

    E.append(card(494, 468, 197, 90, CG, 'Fusion Strategy', [  # 绘制"融合策略"卡片
        ('Only TD decomposed (wavelet)',  9, False, False, '#334155'),  # 仅对 TD 做小波分解
        ('Other 5 vars kept as-is',      9, False, False, '#334155'),  # 其余 5 个变量保持原样
        ('w_t || [Age,T_out,CO2,T_tgt,RH]', 9, False, False, '#334155'),  # 将小波系数与其他变量拼接
        ('-> 9 input channels total',    9, True,  False, CG),  # 加粗说明共 9 个输入通道
    ]))
    for ya in [176, 508]:  # 在两个高度绘制第 3 栏到第 4 栏的箭头
        E.append(L(706, ya, 716, ya, CO, 1.8, 'ao'))  # 绘制橙色箭头

    # --- COL 4: ECA Attention ---
    E.append(card(731, 122, 187, 92, CO, 'ECA Overview', [  # 绘制"ECA 概览"卡片
        ('Efficient Channel Attention',   9, False, False, '#334155'),  # 高效通道注意力
        ('Placed before GRU encoder',    9, False, False, '#334155'),  # 置于 GRU 编码器之前
        ('Adaptive channel re-weighting', 9, False, False, '#334155'),  # 自适应通道重加权
        ('No FC layer, no dim-reduction', 9, False, False, '#334155'),  # 无全连接层、不降维
    ]))
    # ECA flow diagram
    ex, ey, ew = 731, 228, 187  # 定义 ECA 流程图的 x、y 与宽度
    E.append(R(ex, ey, ew, 220, '#fff', CO, 1.5, 7, "filter='url(#sh)'"))  # 绘制流程图白色背景框
    E.append(R(ex, ey, ew, 24, CO, CO, 0, 7))  # 绘制标题栏
    E.append(R(ex, ey+12, ew, 12, CO, CO, 0, 0))  # 补齐标题栏圆角
    E.append(T(ex+ew//2, ey+15, 'ECA Pipeline', 10, '#fff', bold=True))  # 绘制标题文字
    steps_eca = [  # 定义 ECA 处理流程的各步骤（文本、背景色、文字色）
        ('Input X  (B, T, C)',        '#fef3c7', '#92400e'),  # 输入张量形状
        ('Permute -> (B, C, T)',      '#fef3c7', '#92400e'),  # 维度置换
        ('GAP over T -> (B, C, 1)',   '#fde68a', '#78350f'),  # 沿时间维全局平均池化
        ('1D Conv (kernel k)',        '#fbbf24', '#78350f'),  # 一维卷积学习通道权重
        ('Sigmoid -> weights (B,C,1)','#fde68a', '#78350f'),  # Sigmoid 归一化得到权重
        ('Broadcast x Multiply',     '#fef3c7', '#92400e'),  # 广播相乘完成重加权
        ('Permute -> (B, T, C)',      '#fef3c7', '#92400e'),  # 恢复原始维度顺序
        ('-> Weighted X_out',        '#fbbf24', '#78350f'),  # 输出加权后的张量
    ]
    for i, (s, bg, fc) in enumerate(steps_eca):  # 遍历每一步绘制流程框
        E.append(R(ex+12, ey+32+i*22, ew-24, 18, bg, CO, 0.8, 3))  # 绘制步骤背景框
        E.append(T(ex+ew//2, ey+44+i*22, s, 8.5, fc))  # 绘制步骤文字
        if i < len(steps_eca)-1:  # 若不是最后一步
            E.append(L(ex+ew//2, ey+50+i*22, ex+ew//2, ey+54+i*22, CO, 1, 'ao'))  # 绘制指向下一步的箭头

    E.append(card(731, 468, 187, 70, CO, 'Key Formula', [  # 绘制"核心公式"卡片
        ('g_c = (1/T) * sum_t(x_ct)',  9,   False, True,  '#92400e'),  # 全局平均池化公式
        ('a_c = Sigmoid(Conv1D(g))',   9,   False, True,  '#92400e'),  # 通道权重计算
        ('x_out = a_c * x',           9,   False, True,  '#92400e'),  # 加权输出
    ]))
    E.append(card(731, 552, 187, 76, CO, 'Kernel Size k', [  # 绘制"卷积核大小 k"卡片
        ('k = odd(|log2(C)/2 + 0.5|)', 9, False, True,  '#92400e'),  # k 的自适应计算公式
        ('Adaptive to channel count C', 9, False, False, '#334155'),  # 随通道数自适应
        ('Lightweight: no FC layers',   9, False, False, '#334155'),  # 轻量化设计
        ('Local cross-channel interact', 9, False, False, '#334155'),  # 局部跨通道交互
    ]))
    for ya in [168, 590]:  # 在两个高度绘制第 4 栏到第 5 栏的箭头
        E.append(L(933, ya, 943, ya, CP, 1.8, 'ap'))  # 绘制紫色箭头

    # --- COL 5: GRU Encoder ---
    E.append(card(958, 122, 187, 100, CP, 'GRU Settings', [  # 绘制"GRU 设置"卡片
        ('Single-layer GRU',           9, False, False, '#334155'),  # 单层 GRU
        ('Hidden units: 128',          9, False, False, '#334155'),  # 隐藏单元 128
        ('Dropout: 0.1',               9, False, False, '#334155'),  # Dropout 0.1
        ('Input: (B, 36, 9)',          9, False, False, '#334155'),  # 输入形状
        ('Output: last hidden h_T',    9, False, False, '#334155'),  # 输出最后隐状态
    ]))
    # GRU gate diagram
    gx, gy, gw = 958, 236, 187  # 定义 GRU 门控流程图的 x、y 与宽度
    E.append(R(gx, gy, gw, 230, '#fff', CP, 1.5, 7, "filter='url(#sh)'"))  # 绘制流程图白色背景框
    E.append(R(gx, gy, gw, 24, CP, CP, 0, 7))  # 绘制标题栏
    E.append(R(gx, gy+12, gw, 12, CP, CP, 0, 0))  # 补齐标题栏圆角
    E.append(T(gx+gw//2, gy+15, 'GRU Gate Flow', 10, '#fff', bold=True))  # 绘制标题文字
    steps_gru = [  # 定义 GRU 门控流程的各步骤
        ('x_t  (input at step t)',     '#ede9fe', '#4c1d95'),  # 当前时刻输入
        ('z_t = Sigmoid(W_z[h,x])',   '#ddd6fe', '#5b21b6'),  # 更新门
        ('r_t = Sigmoid(W_r[h,x])',   '#ddd6fe', '#5b21b6'),  # 重置门
        ('h~ = Tanh(W[r*h, x])',      '#c4b5fd', '#5b21b6'),  # 候选隐状态
        ('h_t = (1-z)*h_{t-1} + z*h~','#a78bfa', '#4c1d95'),  # 新隐状态更新公式
        ('Repeat for T=36 steps',     '#ede9fe', '#4c1d95'),  # 重复 36 个时间步
        ('Take h_T (last state)',      '#c4b5fd', '#5b21b6'),  # 取最后隐状态
        ('FC layer: h_T -> y_pred',   '#a78bfa', '#4c1d95'),  # 全连接层输出预测
    ]
    for i, (s, bg, fc) in enumerate(steps_gru):  # 遍历每一步绘制流程框
        E.append(R(gx+12, gy+32+i*24, gw-24, 19, bg, CP, 0.8, 3))  # 绘制步骤背景框
        E.append(T(gx+gw//2, gy+45+i*24, s, 8.5, fc))  # 绘制步骤文字
        if i < len(steps_gru)-1:  # 若不是最后一步
            E.append(L(gx+gw//2, gy+51+i*24, gx+gw//2, gy+56+i*24, CP, 1, 'ap'))  # 绘制指向下一步的箭头

    E.append(card(958, 484, 187, 76, CP, 'Training Setup', [  # 绘制"训练设置"卡片
        ('Loss: MSE  |  Optimizer: Adam', 9, False, False, '#334155'),  # 损失与优化器
        ('LR=0.001  |  Batch=32',        9, False, False, '#334155'),  # 学习率与批大小
        ('Early stop patience=10',       9, False, False, '#334155'),  # 早停耐心值
        ('5-fold cross-validation',      9, False, False, '#334155'),  # 五折交叉验证
    ]))
    for ya in [172, 522]:  # 在两个高度绘制第 5 栏到第 6 栏的箭头
        E.append(L(1160, ya, 1170, ya, CR, 1.8, 'ar'))  # 绘制红色箭头

    # --- COL 6: Prediction & Warning ---
    E.append(card(1185, 122, 347, 88, CR, 'Online Rolling Prediction', [  # 绘制"在线滚动预测"卡片
        ('Server receives data every 5 min',  9, False, False, '#334155'),  # 服务器每 5 分钟接收数据
        ('Build window X_t = [x_{t-35}...x_t]', 9, False, False, '#334155'),  # 构建输入窗口
        ('Forward pass -> TD_pred (t+30min)', 9, False, False, '#334155'),  # 前向推理得到 30 分钟后预测
        ('Write result to DB, update buffer',  9, False, False, '#334155'),  # 写库并更新缓冲区
    ]))
    E.append(card(1185, 224, 347, 96, CR, 'Age-based Threshold Mapping', [  # 绘制"基于日龄的阈值映射"卡片
        ('Age 0-10d:  cold<-1C, hot>+2C',  9, False, False, '#334155'),  # 0-10 日龄阈值
        ('Age 10-24d: thresholds adjusted', 9, False, False, '#334155'),  # 10-24 日龄阈值调整
        ('Age 24-35d: thresholds adjusted', 9, False, False, '#334155'),  # 24-35 日龄阈值调整
        ('Age 35-42d: thresholds adjusted', 9, False, False, '#334155'),  # 35-42 日龄阈值调整
        ('Lookup by current flock age',     9, False, False, '#334155'),  # 按当前日龄查表
    ]))
    E.append(card(1185, 334, 347, 88, CR, 'K-point Persistence Rule', [  # 绘制"K 点持续规则"卡片
        ('Trigger alert only if K consec.',  9, False, False, '#334155'),  # 仅连续 K 步才触发
        ('steps exceed threshold',           9, False, False, '#334155'),  # 超过阈值
        ('Suppresses single-point noise',    9, False, False, '#334155'),  # 抑制单点噪声
        ('Reduces false alarm rate',         9, False, False, '#334155'),  # 降低误报率
    ]))
    E.append(card(1185, 436, 347, 88, CR, 'Severity Grading', [  # 绘制"严重程度分级"卡片
        ('s = TD_pred - threshold',          9, False, False, '#334155'),  # 计算超出阈值的程度
        ('Level 1 (General):  s in [0, s1]', 9, False, False, '#7f1d1d'),  # 一级预警
        ('Level 2 (Important): s in (s1,s2]',9, False, False, '#991b1b'),  # 二级预警
        ('Level 3 (Critical): s > s2',       9, True,  False, '#b91c1c'),  # 三级预警（加粗）
    ]))
    E.append(card(1185, 538, 347, 76, CR, 'Performance Results', [  # 绘制"性能结果"卡片
        ('Wavelet-ECA-GRU (best model):',    9, True,  False, '#7f1d1d'),  # 最佳模型加粗显示
        ('MSE=0.3793  MAE=0.2546  R2=0.9041', 9, False, False, '#334155'),  # 最佳模型指标
        ('vs GRU: MSE-16.9% MAE-24.5%',     9, False, False, '#334155'),  # 相对 GRU 的提升
        ('Enables 30-min advance warning',   9, False, False, '#334155'),  # 可提前 30 分钟预警
    ]))
    # Comparison table
    tx, ty, tw = 1185, 628, 347  # 定义对比表的 x、y 与宽度
    E.append(R(tx, ty, tw, 138, '#fff', CR, 1.5, 7, "filter='url(#sh)'"))  # 绘制对比表白色背景框
    E.append(R(tx, ty, tw, 24, CR, CR, 0, 7))  # 绘制表格标题栏
    E.append(R(tx, ty+12, tw, 12, CR, CR, 0, 0))  # 补齐标题栏圆角
    E.append(T(tx+tw//2, ty+15, 'Model Comparison', 10, '#fff', bold=True))  # 绘制表格标题
    hdrs = ['Model', 'MSE', 'MAE', 'R2']  # 表头文字列表
    hx = [tx+10, tx+130, tx+210, tx+290]  # 各列的 x 坐标
    for i, (h, x) in enumerate(zip(hdrs, hx)):  # 遍历表头及其 x 坐标
        E.append(T(x, ty+36, h, 8.5, '#7f1d1d', 'start', bold=True))  # 绘制表头文字（左对齐加粗）
    rows_t = [  # 定义模型对比表的数据行
        ('LSTM',            '0.5075', '0.3867', '0.828'),  # LSTM 结果
        ('TCN',             '0.5260', '0.4232', '0.816'),  # TCN 结果
        ('GRU',             '0.4565', '0.3373', '0.861'),  # GRU 结果
        ('ECA-GRU',         '0.4104', '0.2967', '0.888'),  # ECA-GRU 结果
        ('Wavelet-GRU',     '0.4217', '0.2836', '0.881'),  # Wavelet-GRU 结果
        ('Wavelet-ECA-GRU', '0.3793', '0.2546', '0.904'),  # 最佳模型结果
    ]
    for j, (m, mse, mae, r2) in enumerate(rows_t):  # 遍历每条模型数据
        bg = '#fee2e2' if j == 5 else ('#fafafa' if j % 2 == 0 else '#fff')  # 最佳模型行浅红底，其余隔行变色
        E.append(R(tx+6, ty+42+j*15, tw-12, 14, bg, 'none', 0, 2))  # 绘制该行背景
        fc = '#7f1d1d' if j == 5 else '#334155'  # 最佳模型行用深红文字
        bld = j == 5  # 最佳模型行加粗
        for val, x in zip([m, mse, mae, r2], hx):  # 遍历该行各列的值与 x 坐标
            E.append(T(x, ty+53+j*15, val, 8.5, fc, 'start', bold=bld))  # 绘制单元格文字

    return '\n'.join(E)  # 返回所有图元以换行拼接的框架图内容


# ==============================================================
# ABSTRACT DIAGRAM
# ==============================================================
def build_abstract():
    """构建研究总览图（Abstract Diagram）SVG 的图元内容。

    返回:
        由所有图元以换行拼接而成的总览图内容（不含 XML 声明与 <svg> 外壳）。
    """
    W, H = 1200, 700  # 定义总览图画布宽高
    E = []  # 初始化图元列表
    E.append(R(0, 0, W, H, '#f8fafc', 'none', 0, 0))  # 绘制整图浅灰背景

    # Title
    E.append(R(28, 14, W-56, 52, '#1e293b', 'none', 0, 8, "filter='url(#sh)'"))  # 绘制深色标题栏背景
    E.append(T(W//2, 36, 'Research Overview: Wavelet-ECA-GRU for Broiler House TD Prediction', 16, '#f1f5f9', bold=True))  # 绘制主标题
    E.append(T(W//2, 56, 'Shandong Dezhou Pingyuan Smart Farming Base | 2025/5/26-4/30 | One full broiler batch | 30,000+ birds', 10, '#94a3b8'))  # 绘制副标题（试验地点与规模）

    # ---- Row 1: three context boxes ----
    ctx = [  # 定义第一行的三个背景介绍卡片
        (28,  80, 360, 130, CB,  'Background & Motivation',  # 卡片位置、尺寸、颜色与标题
         ['High-density broiler farming requires stable thermal env.',  # 高密度养殖需要稳定热环境
          'Ventilation, weather & flock growth all disturb temperature.',  # 通风、天气与生长都会扰动温度
          'Fixed-point sensors miss spatial heterogeneity.',  # 定点传感器难以反映空间差异
          'Need: online TD prediction + advance early warning.']),  # 因此需要在线预测与提前预警
        (408, 80, 360, 130, CG,  'Data Collection System',  # 数据采集系统卡片
         ['Inspection robot: Temp/RH/Wind/CO2, 1-min, 4G upload.',  # 巡检机器人采集多参数并上传
          'Control terminal: Age, T_target, T_out, neg. pressure.',  # 控制终端记录日龄等参数
          'Server: timestamp align -> 5-min resample (9,123 pts).',  # 服务器对齐并重采样
          'Train:Test = 80:20 | 5-fold CV on train set.']),  # 数据集划分与交叉验证
        (788, 80, 384, 130, CO,  'Problem Formulation',  # 问题定义卡片
         ['TD = T_indoor - T_target  (deg C)',  # TD 的定义
          'Input: L=36 steps = 180 min history (9 channels)',  # 输入为 36 步历史
          'Output: TD at t+H, H=6 steps = 30 min ahead',  # 输出为未来第 6 步
          'TD decomposed via causal DWT; others kept raw.']),  # TD 做小波分解，其余保持原样
    ]
    for x, y, w, h, c, title, lines in ctx:  # 遍历三个卡片配置
        E.append(card(x, y, w, h, c, title,  # 绘制卡片
                      [(l, 9, False, False, '#334155') for l in lines]))  # 将每行文字转换为卡片行格式

    # ---- Row 2: model pipeline (horizontal flow) ----
    E.append(T(W//2, 232, 'Proposed Model Pipeline', 13, '#1e293b', bold=True))  # 绘制第二行标题
    pipe = [  # 定义模型流水线的六个环节
        (28,  244, 168, 100, CG, 'Wavelet DWT',  # 小波分解环节
         ['db4, L=3 levels', 'Causal window W=36', 'A3, D1, D2, D3', 'No leakage']),  # 该环节要点
        (220, 244, 168, 100, CO, 'ECA Attention',  # ECA 注意力环节
         ['Channel GAP', '1D Conv (odd k)', 'Sigmoid weights', 'Lightweight']),  # 该环节要点
        (412, 244, 168, 100, CP, 'GRU Encoder',  # GRU 编码器环节
         ['1-layer, 128 units', 'Update & Reset gates', 'Take h_T', 'Dropout=0.1']),  # 该环节要点
        (604, 244, 168, 100, CR, 'FC Output',  # 全连接输出环节
         ['Linear: 128->1', 'Predict TD_t+6', '30-min ahead', 'MSE loss']),  # 该环节要点
        (796, 244, 168, 100, '#0891b2', 'Rolling Deploy',  # 滚动部署环节
         ['5-min step update', 'Buffer len=36', 'Online inference', 'Write to DB']),  # 该环节要点
        (988, 244, 184, 100, '#7c3aed', 'Warning System',  # 预警系统环节
         ['Age-based thresholds', 'K-point persist rule', 'Severity grading', '30-min lead time']),  # 该环节要点
    ]
    arrows_pipe = [(28+168, 294), (220+168, 294), (412+168, 294), (604+168, 294), (796+168, 294)]  # 计算各环节之间箭头的起点坐标
    for x, y, w, h, c, title, lines in pipe:  # 遍历流水线各环节
        E.append(card(x, y, w, h, c, title,  # 绘制环节卡片
                      [(l, 8.5, False, False, '#334155') for l in lines]))  # 将要点转换为卡片行
    for ax, ay in arrows_pipe:  # 遍历箭头起点
        E.append(L(ax, ay, ax+16, ay, '#64748b', 2, 'ak'))  # 绘制灰色箭头连接相邻环节

    # ---- Row 3: Results ----
    E.append(T(W//2, 368, 'Experimental Results', 13, '#1e293b', bold=True))  # 绘制第三行标题

    # Performance table
    tx, ty, tw, th = 28, 380, 440, 170  # 定义性能表格的位置与尺寸
    E.append(R(tx, ty, tw, th, '#fff', CR, 1.5, 7, "filter='url(#sh)'"))  # 绘制表格白色背景框
    E.append(R(tx, ty, tw, 24, CR, CR, 0, 7))  # 绘制表格标题栏
    E.append(R(tx, ty+12, tw, 12, CR, CR, 0, 0))  # 补齐标题栏圆角
    E.append(T(tx+tw//2, ty+15, 'Model Performance Comparison (Test Set)', 10, '#fff', bold=True))  # 绘制表格标题
    cols_t = ['Model', 'MSE', 'MAE', 'R2']  # 定义表头文字
    col_x  = [tx+12, tx+180, tx+270, tx+360]  # 定义各列 x 坐标
    for c, cx in zip(cols_t, col_x):  # 遍历表头与列坐标
        E.append(T(cx, ty+38, c, 9, '#7f1d1d', 'start', bold=True))  # 绘制表头文字
    E.append(L(tx+8, ty+42, tx+tw-8, ty+42, '#fecdd3', 1))  # 绘制表头下方的分隔线
    trows = [  # 定义性能表的数据行
        ('LSTM',             '0.5075', '0.3867', '0.828'),  # LSTM 结果
        ('TCN',              '0.5260', '0.4232', '0.816'),  # TCN 结果
        ('GRU (baseline)',   '0.4565', '0.3373', '0.861'),  # GRU 基线结果
        ('ECA-GRU',          '0.4104', '0.2967', '0.888'),  # ECA-GRU 结果
        ('Wavelet-GRU',      '0.4217', '0.2836', '0.881'),  # Wavelet-GRU 结果
        ('Wavelet-ECA-GRU',  '0.3793', '0.2546', '0.904'),  # 最佳模型结果
    ]
    for j, (m, mse, mae, r2) in enumerate(trows):  # 遍历每条模型数据
        bg = '#fee2e2' if j == 5 else ('#f8fafc' if j % 2 == 0 else '#fff')  # 最佳模型行浅红底，其余隔行变色
        E.append(R(tx+6, ty+46+j*18, tw-12, 16, bg, 'none', 0, 2))  # 绘制该行背景
        fc = '#991b1b' if j == 5 else '#334155'  # 最佳模型行用深红文字
        bld = (j == 5)  # 最佳模型行加粗
        for val, cx in zip([m, mse, mae, r2], col_x):  # 遍历该行各列的值与 x 坐标
            E.append(T(cx, ty+58+j*18, val, 9, fc, 'start', bold=bld))  # 绘制单元格文字

    # Improvement badges
    badges = [  # 定义三个改进点徽章
        (490, 380, 200, 78, CB2,  'vs Baseline GRU',  # 相对基线的提升
         ['MSE: -16.91%', 'MAE: -24.52%', 'R2:  +4.31%']),  # 提升数据
        (710, 380, 200, 78, CG,   'Wavelet Contribution',  # 小波模块的贡献
         ['MAE vs GRU: -15.92%', 'Multi-scale features', 'Trend + detail decomp']),  # 贡献说明
        (930, 380, 242, 78, CO,   'ECA Contribution',  # ECA 模块的贡献
         ['MSE vs GRU: -10.10%', 'Lightweight attention', 'Dynamic ch. weighting']),  # 贡献说明
    ]
    for x, y, w, h, c, title, lines in badges:  # 遍历三个徽章
        E.append(card(x, y, w, h, c, title,  # 绘制徽章卡片
                      [(l, 9, False, False, '#334155') for l in lines]))  # 将要点转换为卡片行

    # Warning strategy
    E.append(card(490, 474, 682, 76, CR, 'Warning Strategy (deployed online)',  # 绘制"预警策略"卡片
                  [('Age-segmented thresholds: cold/hot limits differ by flock age stage (0-10d, 10-24d, 24-35d, 35-42d)', 9, False, False, '#334155'),  # 按日龄分段阈值
                   ('K-point persistence rule: alert only after K consecutive steps exceed threshold -> suppress noise', 9, False, False, '#334155'),  # K 点持续规则
                   ('Severity grading: Level 1 General / Level 2 Important / Level 3 Critical -> priority dispatch', 9, False, False, '#334155')]))  # 严重程度分级

    # Conclusion box
    E.append(card(28, 566, 1144, 100, '#1e293b', 'Conclusions & Contributions',  # 绘制"结论与贡献"卡片
                  [('1. First use of mobile inspection robot for continuous broiler house env. monitoring + server-side alignment.', 9, False, False, '#e2e8f0'),  # 贡献 1
                   ('2. Wavelet-ECA-GRU: causal DWT multi-scale features + ECA channel re-weighting + GRU temporal encoding.', 9, False, False, '#e2e8f0'),  # 贡献 2
                   ('3. Best performance: MSE=0.3793, MAE=0.2546, R2=0.9041 (30-min ahead prediction).', 9, True,  False, '#93c5fd'),  # 贡献 3（加粗）
                   ('4. Online rolling warning: age-based thresholds + K-point rule + severity grading for real deployment.', 9, False, False, '#e2e8f0')]))  # 贡献 4

    return '\n'.join(E)  # 返回所有图元以换行拼接的总览图内容


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':  # 仅当脚本被直接运行时执行
    # marker for abstract
    DEFS2 = DEFS.replace("fill='#dc2626'", "fill='#dc2626'"  )  # reuse same defs
    DEFS2 = DEFS2.replace("id='ab'", "id='ab'")  # 保持原样替换 marker id，仅作占位说明
    # add gray marker for abstract
    # 以下三引号字符串向 DEFS 中追加灰色箭头标记（ak），作为 SVG 模板文本保持原样
    DEFS_ABS = DEFS.replace('</defs>', """
<marker id='ak' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#64748b'/></marker>
</defs>""")

    fw_body = build_framework()  # 构建框架图的图元内容
    fw_svg  = wrap_svg(fw_body, 1580, 940)  # 包裹为完整 SVG 文档（1580x940）
    fw_path = os.path.join(OUT, 'framework_diagram.svg')  # 拼接框架图输出路径
    with open(fw_path, 'w', encoding='utf-8') as f:  # 以 UTF-8 写模式打开框架图文件
        f.write(fw_svg)  # 写入框架图 SVG 内容
    print(f'Written: {fw_path}')  # 打印框架图文件路径

    ab_body = build_abstract()  # 构建总览图的图元内容
    ab_svg_inner = (f'<?xml version="1.0" encoding="UTF-8"?>\n'  # 拼接 XML 声明（跨行表达式）
                    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700" width="1200" height="700">\n'  # 拼接 SVG 根元素与画布尺寸
                    f'{DEFS_ABS}\n{ab_body}\n</svg>')  # 写入含 ak 标记的 DEFS、图元内容与闭合标签
    ab_path = os.path.join(OUT, 'abstract_diagram.svg')  # 拼接总览图输出路径
    with open(ab_path, 'w', encoding='utf-8') as f:  # 以 UTF-8 写模式打开总览图文件
        f.write(ab_svg_inner)  # 写入总览图 SVG 内容
    print(f'Written: {ab_path}')  # 打印总览图文件路径
  
