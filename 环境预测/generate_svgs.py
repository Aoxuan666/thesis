# -*- coding: utf-8 -*-
"""Wavelet-ECA-GRU 框架图 SVG 生成脚本（原始文件即被截断）。

本脚本定义了一套 SVG 图元辅助函数（矩形 R、文本 T、连线 L、水平线 HL、卡片 card），
并利用它们拼装 1580x940 的框架流程图，涵盖数据采集、预处理、小波分解、ECA 注意力、
GRU 编码器与预测预警六个栏目；字体与箭头等公共定义放在 DEFS 字符串中。

注意：原始文件在末尾一行（未闭合的字符串）处被截断，本文件仅对截断行之前的完整代码行
补充注释，未修复、补全或改写该未闭合字符串，也未在其内部插入任何内容；
所有 SVG 模板文本保持原样，注释只出现在字符串之外。
"""

import os  # 导入 os 模块（本段代码中未直接使用，保留原始导入）

OUT = r'd:\resources\pythonTest\Total - 副本 - 副本'  # 输出目录的路径字符串

# ============================================================
# SVG primitives
# ============================================================
FN = "font-family='Segoe UI,Arial,sans-serif'"  # 定义文本元素共用的字体样式属性

def R(x,y,w,h,fill,stroke='none',sw=1.5,rx=7,extra=''):
    """生成一个 SVG 矩形元素字符串。

    参数:
        x, y: 左上角坐标；w, h: 宽高；fill: 填充色。
        stroke: 边框色；sw: 线宽；rx: 圆角半径；extra: 追加的额外属性。
    返回:
        <rect .../> 字符串。
    """
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}' {extra}/>"

def T(x,y,s,sz=10,fill='#334155',anchor='middle',bold=False,italic=False,extra=''):
    """生成一个 SVG 文本元素字符串，并转义文本中的特殊字符。

    参数:
        x, y: 文本坐标；s: 文本内容；sz: 字号；fill: 颜色。
        anchor: 对齐方式；bold/italic: 加粗/斜体；extra: 额外属性。
    返回:
        <text ...>...</text> 字符串。
    """
    fw='bold' if bold else 'normal'  # 根据 bold 参数决定 font-weight
    fi='italic' if italic else 'normal'  # 根据 italic 参数决定 font-style
    s=str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')  # 转义 & < > 避免破坏 XML 结构
    return f"<text x='{x}' y='{y}' {FN} font-size='{sz}' fill='{fill}' text-anchor='{anchor}' font-weight='{fw}' font-style='{fi}' {extra}>{s}</text>"

def L(x1,y1,x2,y2,stroke='#64748b',sw=1.8,mk=''):
    """生成一条 SVG 连线，可带箭头标记。

    参数:
        x1, y1, x2, y2: 起止坐标；stroke: 颜色；sw: 线宽；mk: 箭头标记 id。
    返回:
        <line .../> 字符串。
    """
    m=f" marker-end='url(#{mk})'" if mk else ''  # 若指定箭头标记则追加 marker-end 属性
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' stroke-width='{sw}'{m}/>"

def HL(x1,x2,y,stroke='#e2e8f0',sw=1):
    """生成一条水平线：复用 L 函数，将起点与终点的 y 设为相同值。

    参数:
        x1, x2: 水平线两端 x 坐标；y: 所在高度；stroke: 颜色；sw: 线宽。
    返回:
        <line .../> 字符串。
    """
    return L(x1,y,x2,y,stroke,sw)  # 以相同的 y 调用 L 绘制水平线

def card(x,y,w,h,color,title,rows):
    """rows: list of (text, fontsize, bold, italic, fill)

    生成一个带主题色标题栏与多行文字的信息卡片。
    参数:
        x, y, w, h: 卡片位置与尺寸；color: 主题色；title: 标题。
        rows: 每行 (文本, 字号, 加粗, 斜体, 颜色) 的元组列表。
    返回:
        由多个 SVG 元素换行拼接而成的卡片字符串。
    """
    o=[]  # 初始化图元列表
    o.append(R(x,y,w,h,'#fff',color,1.5,7,'filter="url(#sh)"'))  # 绘制带阴影的白色卡片底
    o.append(R(x,y,w,24,color,color,0,7))  # 绘制顶部标题栏
    o.append(R(x,y+12,w,12,color,color,0,0))  # 补齐标题栏下方圆角
    o.append(T(x+w//2,y+15,title,10,'#fff',bold=True))  # 居中绘制白色标题
    for i,(s,sz,bo,it,fc) in enumerate(rows):  # 遍历每一行文字配置
        o.append(T(x+w//2,y+33+i*15,s,sz,fc,bold=bo,italic=it))  # 按行号下移绘制文字
    return '\n'.join(o)  # 以换行拼接所有图元并返回

# 以下三引号字符串是 SVG 公共定义（字体样式、五种箭头标记与阴影滤镜），不在其内部插入注释
DEFS="""<defs>
<style>.fn{font-family:'Segoe UI',Arial,sans-serif;}</style>
<marker id='ab' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#3b82f6'/></marker>
<marker id='ag' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#059669'/></marker>
<marker id='ao' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#d97706'/></marker>
<marker id='ap' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#7c3aed'/></marker>
<marker id='ar' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#dc2626'/></marker>
<marker id='ak' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#64748b'/></marker>
<filter id='sh'><feDropShadow dx='1' dy='2' stdDeviation='3' flood-color='#00000018'/></filter>
</defs>"""

# ============================================================
# FRAMEWORK SVG
# ============================================================
W,H=1580,940  # 定义整幅框架图的画布宽高

CB='#3b82f6'; CB2='#2563eb'; CG='#059669'; CO='#d97706'; CP='#7c3aed'; CR='#dc2626'  # 定义六个栏目的主题色
cols_cfg=[  # 六个栏目的 (x, 宽, 主题色, 背景色, 边框色, 标题) 配置
    (28, 215, CB,  '#eff6ff','#bfdbfe', '\u2460 Data Collection'),  # 第 1 栏：数据采集
    (255,215, CB2, '#eff6ff','#bfdbfe', '\u2461 Preprocessing'),  # 第 2 栏：数据预处理
    (482,225, CG,  '#ecfdf5','#a7f3d0', '\u2462 Wavelet Decomp. (DWT)'),  # 第 3 栏：小波分解
    (719,215, CO,  '#fffbeb','#fde68a', '\u2463 ECA Attention'),  # 第 4 栏：ECA 注意力
    (946,215, CP,  '#f5f3ff','#ddd6fe', '\u2464 GRU Encoder'),  # 第 5 栏：GRU 编码器
    (1173,379,CR,  '#fff1f2','#fecdd3', '\u2465 Prediction & Warning'),  # 第 6 栏：预测与预警
]

E=[]  # 初始化图元列表
E.append(R(0,0,W,H,'#f8fafc','none',0,0))  # 绘制整图浅灰背景

# Title bar
E.append(R(28,12,W-56,54,'#1e293b','none',0,8,'filter="url(#sh)"'))  # 绘制深色标题栏背景
E.append(T(W//2,36,'Wavelet–ECA–GRU Framework: Broiler House Temperature Deviation Short-term Prediction',  # 绘制主标题文本（跨行调用）
           17,'#f1f5f9',bold=True))  # 主标题字号 17、颜色浅灰、加粗
E.append(T(W//2,56,'Input L=36 steps (180 min) → Predict H=6 steps ahead (30 min)  |  TD = T_indoor − T_target  |  5-min step  |  9,123 samples',  # 绘制副标题文本（跨行调用）
           10,'#94a3b8'))  # 副标题字号 10、颜色次级灰

# Column backgrounds + headers
for (cx,cw,cc,bg,bc,lbl) in cols_cfg:  # 遍历栏目配置
    E.append(R(cx,78,cw,848,bg,bc,1.2,10))  # 绘制栏目的背景与边框
    E.append(R(cx,78,cw,30,cc,'none',0,10))  # 绘制栏目标题条
    E.append(R(cx,90,cw,18,cc,'none',0,0))  # 补齐标题条圆角
    E.append(T(cx+cw//2,98,lbl,11,'#fff',bold=True))  # 居中绘制白色栏目名称

# ---- COL 1: DATA COLLECTION (cx=28, cw=215) ----
c1x,c1m=28,28+107  # 定义第 1 栏左边界 x 与中线 x
E.append(card(c1x+14,122,187,130,CB,'Inspection Robot',[  # 绘制"巡检机器人"卡片
    ('● Temperature / Humidity',9.5,False,False,'#1e3a8a'),  # 采集温湿度
    ('● Wind Speed (m/s)',9.5,False,False,'#1e3a8a'),  # 采集风速
    ('● CO₂ Concentration (ppm)',9.5,False,False,'#1e3a8a'),  # 采集 CO2 浓度
    ('● Sample rate: 1 min',9.5,False,False,'#1e3a8a'),  # 采样频率 1 分钟
    ('● 4G wireless upload to server',9.5,False,False,'#1e3a8a'),  # 4G 无线上传服务器
]))
E.append(card(c1x+14,268,187,125,CB,'Control Terminal',[  # 绘制"控制终端"卡片
    ('● Age of flock (Day)',9.5,False,False,'#1e3a8a'),  # 记录鸡群日龄
    ('● Target Temperature (T_target)',9.5,False,False,'#1e3a8a'),  # 记录目标温度
    ('● Outdoor Temperature (T_out)',9.5,False,False,'#1e3a8a'),  # 记录室外温度
    ('● Negative Pressure / State',9.5,False,False,'#1e3a8a'),  # 记录负压/状态
    ('● 1-min sync to server',9.5,False,False,'#1e3a8a'),  # 每分钟同步服务器
]))
E.append(card(c1x+14,408,187,118,CB,'Sensor Specifications',[  # 绘制"传感器规格"卡片
    ('Temp: ±0.2°C  Range: −20~50°C',9,False,False,'#334155'),  # 温度精度与量程
    ('Humidity: ±2% RH',9,False,False,'#334155'),  # 湿度精度
    ('Wind: ±(0.2+0.02v) m/s',9,False,False,'#334155'),  # 风速精度
    ('CO₂: ±50 ppm  (300~5000 ppm)',9,False,False,'#334155'),  # CO2 精度与量程
    ('Period: 2025/5/26 – 4/30',9,False,False,'#334155'),  # 数据采集时段
]))
E.append(card(c1x+14,540,187,108,CB,'Coop Information',[  # 绘制"鸡舍信息"卡片
    ('101m×16m, 30,000+ birds',9,False,False,'#334155'),  # 鸡舍尺寸与养殖量
    ('7 rows × 4 layers of cages',9,False,False,'#334155'),  # 笼架排布
    ('24 large + 2 small exhaust fans',9,False,False,'#334155'),  # 风机配置
    ('Longitudinal ventilation mode',9,False,False,'#334155'),  # 通风模式
    ('Evap. cooling + wet curtains',9,False,False,'#334155'),  # 蒸发冷却与水帘
]))

# col1->col2
for ya in [185,330]:  # 在两个高度绘制第 1 栏到第 2 栏的箭头
    E.append(L(241,ya,252,ya,CB,1.8,'ab'))  # 绘制蓝色箭头

# ---- COL 2: PREPROCESSING (cx=255, cw=215) ----
c2x=255  # 定义第 2 栏左边界 x
E.append(card(c2x+14,122,187,92,CB2,'TD Definition',[  # 绘制"TD 定义"卡片
    ('TD = T_indoor − T_target',10.5,False,True,'#1e3a8a'),  # TD 为室内温度减目标温度
    ('Temperature Deviation (°C)',9,False,False,'#334155'),  # 含义为温度偏差
    ('Core prediction target variable',9,False,False,'#334155'),  # 是核心预测目标
    ('Reflects thermal control error',9,False,False,'#334155'),  # 反映热控制误差
]))
E.append(card(c2x+14,228,187,105,CB2,'Align & Resample',[  # 绘制"对齐与重采样"卡片
    ('Timestamp alignment (server)',9,False,False,'#334155'),  # 服务器端时间戳对齐
    ('1 min → 5 min resampling',9,False,False,'#334155'),  # 1 分钟重采样为 5 分钟
    ('Total samples: 9,123',9,False,False,'#334155'),  # 共 9123 个样本
    ('Train : Test = 80 : 20',9,False,False,'#334155'),  # 按 8:2 划分数据集
    ('No leakage across split',9,False,False,'#334155'),  # 无数据泄漏
]))
E.append(card(c2x+14,347,187,92,CB2,'Missing Value Fill',[  # 绘制"缺失值填充"卡片
    ('Continuous: linear interpolation',9,False,False,'#334155'),  # 连续变量线性插值
    ('+ forward / backward fill',9,False,False,'#334155'),  # 辅以前后填充
    ('Discrete: forward/backward fill',9,False,False,'#334155'),  # 离散变量前后填充
    ('Drop remaining NaN rows',9,False,False,'#334155'),  # 丢弃剩余缺失行
]))
E.append(card(c2x+14,453,187,108,CB2,'Pearson Feature Select',[  # 绘制"Pearson 特征选择"卡片
    ('TD+: Age(+), T_out(+)',9,False,False,'#334155'),  # 与 TD 正相关特征
    ('TD−: CO₂(−), T_target(−), RH(−)',9,False,False,'#334155'),  # 与 TD 负相关特征
    ('Selected 6 features:',9,True,False,'#1d4ed8'),  # 加粗提示所选 6 个特征
    ('TD, Age, T_out, CO₂, T_tgt, RH',9,False,False,'#334155'),  # 列出所选特征
    ('C=9 channels (TD → 4 wavelet)',9,False,False,'#334155'),  # 小波扩展后共 9 通道
]))
E.append(card(c2x+14,575,187,84,CB2,'Min-Max Normalization',[  # 绘制"最小-最大归一化"卡片
    ("x' = (x−x_min)/(x_max−x_min)",9.5,False,True,'#1e3a8a'),  # 归一化公式
    ('Fit on train set only → [0,1]',9,False,False,'#334155'),  # 仅在训练集拟合
    ('Apply same scaler to test set',9,False,False,'#334155'),  # 测试集沿用同一缩放器
    ('TD target also scaled',9,False,False,'#334155'),  # TD 目标也缩放
]))
E.append(card(c2x+14,673,187,90,CB2,'Sliding Window Samples',[  # 绘制"滑动窗口样本"卡片
    ('Input X: L=36 steps (180 min)',9,False,False,'#334155'),  # 输入 36 步窗口
    ('Target y: step t+H, H=6',9,False,False,'#334155'),  # 目标为未来第 6 步
    ('Stride = 1 (online aligned)',9,False,False,'#334155'),  # 步长为 1
    ('Supervised learning samples',9,False,False,'#334155'),  # 构成监督学习样本
]))

# col2->col3
for ya in [168,610]:  # 在两个高度绘制第 2 栏到第 3 栏的箭头
    E.append(L(469,ya,479,ya,CG,1.8,'ag'))  # 绘制绿色箭头

# ---- COL 3: WAVELET (cx=482, cw=225) ----
c3x=482  # 定义第 3 栏左边界 x
E.append(card(c3x+12,122,197,108,CG,'DWT Settings',[  # 绘制"DWT 设置"卡片
    ('Wavelet basis: db4 (Daubechies 4)',9,False,False,'#334155'),  # 使用 db4 小波基
    ('Decomposition levels: L = 3',9,False,False,'#334155'),  # 分解层数 3
    ('Causal sliding window: W = 36',9,False,False,'#334155'),  # 因果窗口 36 步
    ('No future data leakage',9,False,False,'#334155'),  # 无未来数据泄漏
    ('Train/test split before DWT',9,False,False,'#334155'),  # 先划分数据集再做 DWT
]))

# Decomp tree
dtx,dty,dtw,dth=c3x+12,244,197,196  # 定义分解树的 x、y、宽、高
E.append(R(dtx,dty,dtw,dth,'#fff',CG,1.5,7,'filter="url(#sh)"'))  # 绘制分解树白色背景框
E.append(R(dtx,dty,dtw,24,CG,CG,0,7))  # 绘制标题栏
E.append(R(dtx,dty+12,dtw,12,CG,CG,0,0))  # 补齐标题栏圆角
E.append(T(dtx+dtw//2,dty+15,'Decomposition Tree',10,'#fff',bold=True))  # 绘制标题文字
# signal box
E.append(R(dtx+24,dty+30,dtw-48,22,'#d1fae5',CG,1,4))  # 绘制 TD 信号输入框
E.append(T(dtx+dtw//2,dty+44,'TD Signal Window [t-35 ... t]',9,'#065f46',bold=True))  # 标注信号窗口范围
# arrow
E.append(L(dtx+dtw//2,dty+52,dtx+dtw//2,dty+62,CG,1.5,'ag'))  # 绘制信号框到分解层的绿色箭头
# A3 and Detail boxes
E.append(R(dtx+8,dty+64,88,22,'#6ee7b7',CG,1,4))  # 绘制 A3 近似分量框
E.append(T(dtx+52,dty+78,'A3 