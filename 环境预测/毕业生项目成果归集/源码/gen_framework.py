# -*- coding: utf-8 -*-
"""Wavelet-ECA-GRU 框架图（Framework）SVG 生成脚本（原始文件即被截断）。

本脚本通过一系列轻量 SVG 图元函数（矩形、文本、连线、卡片）拼装一幅 1580x940 的
"Wavelet-ECA-GRU 框架"流程图，依次展示数据采集、预处理、小波分解、ECA 注意力、
GRU 编码器以及预测预警六个纵向栏目。

注意：原始文件在末尾一行（未闭合的字符串）处被截断，本文件仅对截断行之前的完整代码行
补充注释，未修复、补全或改写该未闭合字符串，也未在其内部插入任何内容。
"""

import os  # 导入 os 模块（本段代码中未直接使用，保留原始导入）

OUT = r'd:\resources\pythonTest\Total - \u526f\u672c - \u526f\u672c'  # 输出目录的原始字符串路径，其中包含 \u 转义序列

def svg(content, w, h):
    """将给定的 SVG 图元内容包裹为完整的 SVG 文档字符串。

    参数:
        content: 已拼好的 SVG 内部图元字符串。
        w: SVG 画布宽度（同时作为 viewBox 与 width）。
        h: SVG 画布高度（同时作为 viewBox 与 height）。
    返回:
        包含 XML 声明、defs 样式/箭头/阴影定义及 content 的完整 SVG 字符串。
    """
    # 以下三引号内为 SVG 模板文本，包含样式、五种箭头标记与阴影滤镜，不在其内部插入注释
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<defs>
<style>.fn{{font-family:'Segoe UI',Arial,sans-serif;}}</style>
<marker id="ab" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#3b82f6"/></marker>
<marker id="ag" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#059669"/></marker>
<marker id="ao" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#d97706"/></marker>
<marker id="ap" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#7c3aed"/></marker>
<marker id="ar" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#dc2626"/></marker>
<filter id="sh"><feDropShadow dx="1" dy="2" stdDeviation="3" flood-color="#00000018"/></filter>
</defs>
{content}
</svg>'''

def R(x,y,w,h,fill,stroke='none',sw=1.5,rx=7,ex=''):
    """生成一个 SVG 矩形元素字符串。

    参数:
        x, y: 矩形左上角坐标；w, h: 宽高；fill: 填充色。
        stroke: 边框色；sw: 边框宽度；rx: 圆角半径；ex: 追加到标签中的额外属性。
    返回:
        <rect .../> 字符串。
    """
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {ex}/>'

def T(x,y,s,sz=10,fill='#334155',anchor='middle',bold=False,italic=False):
    """生成一个 SVG 文本元素字符串，并对文本内容做 HTML 转义。

    参数:
        x, y: 文本锚点坐标；s: 文本内容；sz: 字号；fill: 文字颜色。
        anchor: 文本对齐方式；bold/italic: 是否加粗/斜体。
    返回:
        <text ...>...</text> 字符串。
    """
    s=str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')  # 对 & < > 做转义，避免破坏 SVG/XML 结构
    fw='bold' if bold else 'normal'  # 根据 bold 参数决定 font-weight
    fi='italic' if italic else 'normal'  # 根据 italic 参数决定 font-style
    return f'<text x="{x}" y="{y}" class="fn" font-size="{sz}" fill="{fill}" text-anchor="{anchor}" font-weight="{fw}" font-style="{fi}">{s}</text>'

def L(x1,y1,x2,y2,c='#64748b',sw=1.8,mk=''):
    """生成一个 SVG 连线元素字符串，可带箭头标记。

    参数:
        x1, y1, x2, y2: 线段起点与终点坐标；c: 线条颜色；sw: 线宽。
        mk: 箭头标记 id，为空则不加箭头。
    返回:
        <line .../> 字符串。
    """
    m=f' marker-end="url(#{mk})"' if mk else ''  # 若指定了箭头标记，则拼接 marker-end 属性
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"{m}/>'

def card(x,y,w,h,color,title,rows):
    """生成一张带标题栏与多行文字的信息卡片（由矩形和文本组合而成）。

    参数:
        x, y, w, h: 卡片位置与尺寸；color: 主题色。
        title: 卡片标题；rows: (文本, 字号, 加粗, 斜体, 颜色) 的元组列表。
    返回:
        由多个 SVG 元素以换行拼接而成的卡片字符串。
    """
    # rows: list of (text, sz, bold, italic, fill)
    o=[]  # 初始化卡片图元列表
    o.append(R(x,y,w,h,'#fff',color,1.5,7,'filter="url(#sh)"'))  # 绘制带阴影的白色卡片背景
    o.append(R(x,y,w,24,color,color,0,7))  # 绘制卡片顶部标题栏
    o.append(R(x,y+12,w,12,color,color,0,0))  # 用矩形补齐标题栏下方的圆角
    o.append(T(x+w//2,y+15,title,10,'#fff',bold=True))  # 在标题栏居中绘制白色标题文字
    for i,(s,sz,bo,it,fc) in enumerate(rows):  # 遍历每一行文字配置
        o.append(T(x+w//2,y+33+i*15,s,sz,fc,bold=bo,italic=it))  # 按行号下移绘制每行文字
    return '\n'.join(o)  # 将所有图元以换行符拼接返回


# ===================== FRAMEWORK =====================
W,H=1580,940  # 定义整幅框架图的画布宽高
CB,CB2,CG,CO,CP,CR='#3b82f6','#2563eb','#059669','#d97706','#7c3aed','#dc2626'  # 定义六个栏目的主题色

cfg=[  # 定义六个栏目的 (x坐标, 宽度, 主题色, 背景色, 边框色, 标题) 配置
    (28,215,CB, '#eff6ff','#bfdbfe','(1) Data Collection'),  # 第 1 栏：数据采集
    (255,215,CB2,'#eff6ff','#bfdbfe','(2) Preprocessing'),  # 第 2 栏：数据预处理
    (482,225,CG, '#ecfdf5','#a7f3d0','(3) Wavelet DWT'),  # 第 3 栏：小波离散分解
    (719,215,CO, '#fffbeb','#fde68a','(4) ECA Attention'),  # 第 4 栏：ECA 注意力机制
    (946,215,CP, '#f5f3ff','#ddd6fe','(5) GRU Encoder'),  # 第 5 栏：GRU 编码器
    (1173,379,CR,'#fff1f2','#fecdd3','(6) Prediction & Warning'),  # 第 6 栏：预测与预警
]

E=[]  # 初始化整幅图的图元列表
E.append(R(0,0,W,H,'#f8fafc','none',0,0))  # 绘制整图浅灰背景
E.append(R(28,12,W-56,54,'#1e293b','none',0,8,'filter="url(#sh)"'))  # 绘制深色标题栏背景
E.append(T(W//2,36,'Wavelet-ECA-GRU Framework: Broiler House Temperature Deviation Short-term Prediction',17,'#f1f5f9',bold=True))  # 绘制主标题
E.append(T(W//2,56,'Input L=36 steps (180 min) -> Predict H=6 ahead (30 min)  |  TD=T_indoor-T_target  |  5-min step  |  9,123 samples',10,'#94a3b8'))  # 绘制副标题说明输入输出与采样信息

for cx,cw,cc,bg,bc,lbl in cfg:  # 遍历栏目配置，逐栏绘制背景与标题
    E.append(R(cx,78,cw,848,bg,bc,1.2,10))  # 绘制该栏目的背景矩形与边框
    E.append(R(cx,78,cw,30,cc,'none',0,10))  # 绘制栏目顶部标题条
    E.append(R(cx,90,cw,18,cc,'none',0,0))  # 补齐标题条下方的圆角
    E.append(T(cx+cw//2,98,lbl,11,'#fff',bold=True))  # 在标题条上居中绘制栏目名称

# --- COL1 ---
E.append(card(42,122,187,130,CB,'Inspection Robot',[  # 绘制第 1 栏"巡检机器人"信息卡片
    ('Temperature / Humidity',9.5,False,False,'#1e3a8a'),  # 巡检机器人采集温湿度
    ('Wind Speed (m/s)',9.5,False,False,'#1e3a8a'),  # 采集风速
    ('CO2 Concentration (ppm)',9.5,False,False,'#1e3a8a'),  # 采集 CO2 浓度
    ('Sample rate: 1 min',9.5,False,False,'#1e3a8a'),  # 采样频率为 1 分钟
    ('4G wireless upload',9.5,False,False,'#1e3a8a'),  # 通过 4G 无线上传
]))
E.append(card(42,268,187,125,CB,'Control Terminal',[  # 绘制"控制终端"信息卡片
    ('Age of flock (Day)',9.5,False,False,'#1e3a8a'),  # 记录鸡群日龄
    ('Target Temperature (T_target)',9.5,False,False,'#1e3a8a'),  # 记录目标温度
    ('Outdoor Temperature (T_out)',9.5,False,False,'#1e3a8a'),  # 记录室外温度
    ('Negative Pressure / State',9.5,False,False,'#1e3a8a'),  # 记录负压/状态
    ('1-min sync to server',9.5,False,False,'#1e3a8a'),  # 每分钟同步至服务器
]))
E.append(card(42,408,187,118,CB,'Sensor Specs',[  # 绘制"传感器参数"信息卡片
    ('Temp: +/-0.2C  Range:-20~50C',9,False,False,'#334155'),  # 温度精度与量程
    ('Humidity: +/-2% RH',9,False,False,'#334155'),  # 湿度精度
    ('Wind: +/-(0.2+0.02v) m/s',9,False,False,'#334155'),  # 风速精度
    ('CO2: +/-50 ppm (300~5000)',9,False,False,'#334155'),  # CO2 精度与量程
    ('2025/5/26 - 4/30 (1 batch)',9,False,False,'#334155'),  # 数据采集时间范围
]))
E.append(card(42,540,187,108,CB,'Coop Info',[  # 绘制"鸡舍信息"信息卡片
    ('101m x 16m | 30,000+ birds',9,False,False,'#334155'),  # 鸡舍尺寸与养殖数量
    ('7 rows x 4-layer cages',9,False,False,'#334155'),  # 笼架排布
    ('24 large + 2 small fans',9,False,False,'#334155'),  # 风机配置
    ('Longitudinal ventilation',9,False,False,'#334155'),  # 纵向通风方式
    ('Evap. cooling + wet curtain',9,False,False,'#334155'),  # 蒸发冷却与水帘
]))
for ya in [185,330]:  # 在两个高度位置绘制从第 1 栏指向第 2 栏的箭头
    E.append(L(241,ya,252,ya,CB,1.8,'ab'))  # 绘制蓝色箭头

# --- COL2 ---
E.append(card(269,122,187,88,CB2,'TD Definition',[  # 绘制第 2 栏"TD 定义"卡片
    ('TD = T_indoor - T_target',10.5,False,True,'#1e3a8a'),  # TD 为室内温度减目标温度
    ('Temperature Deviation (degC)',9,False,False,'#334155'),  # 含义为温度偏差
    ('Core prediction target',9,False,False,'#334155'),  # 是核心预测目标
    ('Non-stationary time series',9,False,False,'#334155'),  # 属于非平稳时间序列
]))
E.append(card(269,224,187,108,CB2,'Align & Resample',[  # 绘制"对齐与重采样"卡片
    ('Timestamp alignment (server)',9,False,False,'#334155'),  # 在服务器端做时间戳对齐
    ('1 min -> 5 min resampling',9,False,False,'#334155'),  # 由 1 分钟重采样为 5 分钟
    ('Total: 9,123 samples',9,False,False,'#334155'),  # 共得到 9123 个样本
    ('Train:Test = 80:20 split',9,False,False,'#334155'),  # 按 8:2 划分训练与测试集
    ('No leakage across split',9,False,False,'#334155'),  # 划分间无数据泄漏
]))
E.append(card(269,346,187,90,CB2,'Missing Value Fill',[  # 绘制"缺失值填充"卡片
    ('Continuous: linear interpolation',9,False,False,'#334155'),  # 连续变量使用线性插值
    ('+ forward/backward fill',9,False,False,'#334155'),  # 辅以向前/向后填充
    ('Discrete: fwd/bwd fill',9,False,False,'#334155'),  # 离散变量用前后填充
    ('Drop remaining NaN rows',9,False,False,'#334155'),  # 丢弃仍为 NaN 的行
]))
E.append(card(269,450,187,108,CB2,'Pearson Feature Select',[  # 绘制"Pearson 特征选择"卡片
    ('TD+: Age(+), T_out(+)',9,False,False,'#334155'),  # 与 TD 正相关的特征
    ('TD-: CO2(-), T_tgt(-), RH(-)',9,False,False,'#334155'),  # 与 TD 负相关的特征
    ('Selected 6 features:',9,True,False,'#1d4ed8'),  # 加粗提示最终选出的 6 个特征
    ('TD,Age,T_out,CO2,T_tgt,RH',9,False,False,'#334155'),  # 列出所选特征
    ('C=9 channels (TD->4 wavelet)',9,False,False,'#334155'),  # TD 经小波扩展后共 9 个通道
]))
E.append(card(269,572,187,80,CB2,'Min-Max Normalize',[  # 绘制"最小-最大归一化"卡片
    ("x'=(x-xmin)/(xmax-xmin)",9.5,False,True,'#1e3a8a'),  # 归一化公式
    ('Fit on train set -> [0,1]',9,False,False,'#334155'),  # 仅在训练集上拟合，映射到 [0,1]
    ('Same scaler on test set',9,False,False,'#334155'),  # 测试集使用同一缩放器
    ('TD target also scaled',9,False,False,'#334155'),  # TD 目标值同样被缩放
]))
E.append(card(269,666,187,88,CB2,'Sliding Window',[  # 绘制"滑动窗口"卡片
    ('Input X: L=36 steps (180min)',9,False,False,'#334155'),  # 输入为 36 步即 180 分钟窗口
    ('Target y: t+H, H=6 (30min)',9,False,False,'#334155'),  # 目标为未来第 6 步即 30 分钟后
    ('Stride=1, online aligned',9,False,False,'#334155'),  # 步长为 1，与线上对齐
    ('Supervised learning setup',9,False,False,'#334155'),  # 构成监督学习样本
]))
for ya in [168,608]:  # 在两个高度位置绘制从第 2 栏指向第 3 栏的箭头
    E.append(L(469,ya,479,ya,CG,1.8,'ag'))  # 绘制绿色箭头

# --- COL3: WAVELET ---
E.append(card(494,122,197,108,CG,'DWT Settings',[  # 绘制第 3 栏"DWT 设置"卡片
    ('Wavelet: db4 (Daubechies 4)',9,False,False,'#334155'),  # 使用 db4 小波基
    ('Decomposition levels: L=3',9,False,False,'#334155'),  # 分解层数为 3
    ('Causal window: W=36 steps',9,False,False,'#334155'),  # 因果窗口为 36 步
    ('No future data leakage',9,False,False,'#334155'),  # 不使用未来数据，避免泄漏
    ('Fit before train/test split',9,False,False,'#334155'),  # 在划分前完成分解拟合
]))
# decomp tree
dx,dy,dw=494,244,197  # 定义分解树图表的起始 x、y 与宽度
E.append(R(dx,dy,dw,196,'#fff',CG,1.5,7,'filter="url(#sh)"'))  # 绘制分解树白色背景框
E.append(R(dx,dy,dw,24,CG,CG,0,7))  # 绘制分解树标题栏
E.append(R(dx,dy+12,dw,12,CG,CG,0,0))  # 补齐标题栏圆角
E.append(T(dx+dw//2,dy+15,'Decomposition Tree',10,'#fff',bold=True))  # 绘制分解树标题文字
E.append(R(dx+24,dy+30,dw-48,22,'#d1fae5',CG,1,4))  # 绘制 TD 信号输入框
E.append(T(dx+dw//2,dy+44,'TD Signal [t-35 ... t]',9,'#065f46',bold=True))  # 标注 TD 信号窗口范围
E.append(L(dx+dw//2,dy+52,dx+dw//2,dy+62,CG,1.5,'ag'))  # 绘制信号框到分解层的绿色箭头
E.append(R(dx+8,dy+64,86,22,'#6ee7b7',CG,1,4))  # 绘制 A3 近似分量框
E.append(T(dx+51,dy+78,'A3 (trend)',9,'#064e3b',bold=True))  # 标注 A3 为趋势分量
E.append(R(dx+100,dy+64,89,22,'#a7f3d0',CG,1,4))  # 绘制 D1/D2/D3 细节分量框
E.append(T(dx+144,dy+78,'D1,D2,D3',9,'#064e3b',bold=True))  # 标注三个细节分量
E.append(L(dx+51,dy+86,dx+51,dy+98,CG,1))  # 绘制 A3 到拼接框的连线
E.append(L(dx+144,dy+86,dx+144,dy+98,CG,1))  # 绘制细节分量到拼接框的连线
E.append(R(dx+24,dy+98,dw-48,22,'#ecfdf5',CG,1,4))  # 绘制小波系数拼接框
E.append(T(dx+dw//2,dy+112,'w_t = [A3, D1, D2, D3]',9,'#065f46',bold=True,italic=True))  # 标注拼接后的小波向量
E.append(L(dx+dw//2,dy+120,dx+dw//2,dy+130,CG,1.5,'ag'))  # 绘制拼接框到下一步的绿色箭头
E.append(R(dx+24,dy+132,dw-48,22,'#d1fae5',CG,1,4))  # 绘制最终特征拼接框（此行之后的文件被截断）
E.append(T(dx+dw//2,dy+146,'concat w/ 