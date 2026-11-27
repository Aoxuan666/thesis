# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8
"""
SVG 基础图元生成工具模块。

本模块集中定义生成 SVG 片段的轻量辅助函数：矩形 R、文本 T、直线 L，
以及由这些图元组合而成的卡片 card；同时提供全局样式与箭头标记定义 DEFS 字符串。
其他绘图脚本通过复用这些函数，避免重复拼接 SVG 标签。
"""
# Shared SVG primitive helpers

FN = "class='fn'"  # 定义所有文本元素共用的 CSS 类属性字符串

def R(x,y,w,h,fill,stroke='none',sw=1.5,rx=7,ex=''):  # 定义生成 SVG 矩形元素的辅助函数
    """
    生成 SVG 矩形元素字符串。
    参数 x/y 为左上角坐标，w/h 为宽高，fill 为填充色，stroke 为描边色，
    sw 为描边宽度，rx 为圆角半径，ex 为附加属性字符串；返回 rect 标签文本。
    """
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}' {ex}/>"  # 用 f-string 拼接 rect 标签，填入坐标、尺寸、圆角、填充与描边属性

def T(x,y,s,sz=10,fill='#334155',anchor='middle',bold=False,italic=False):  # 定义生成 SVG 文本元素的辅助函数
    """
    生成 SVG 文本元素字符串。
    参数 x/y 为文字锚点坐标，s 为文本内容（自动转义 XML 特殊字符），
    sz 为字号，fill 为颜色，anchor 为对齐方式，bold/italic 控制字重与斜体；
    返回拼接好的 text 标签文本。
    """
    s=str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')  # 对文本内容做 XML 转义，避免 & < > 破坏 SVG 结构
    fw='bold' if bold else 'normal'  # 根据 bold 标志计算 CSS 字重值
    fi='italic' if italic else 'normal'  # 根据 italic 标志计算 CSS 字形样式值
    return f"<text x='{x}' y='{y}' {FN} font-size='{sz}' fill='{fill}' text-anchor='{anchor}' font-weight='{fw}' font-style='{fi}'>{s}</text>"  # 拼接 text 标签，写入位置、字体属性与转义后的文本

def L(x1,y1,x2,y2,c='#64748b',sw=1.8,mk=''):  # 定义生成 SVG 直线元素的辅助函数
    """
    生成 SVG 直线元素字符串。
    参数 x1/y1 与 x2/y2 为两端坐标，c 为颜色，sw 为线宽，
    mk 为箭头标记 id（非空时附加 marker-end 属性）；返回 line 标签文本。
    """
    m=f" marker-end='url(#{mk})'" if mk else ''  # 若指定了箭头标记 id，则生成 marker-end 属性；否则为空字符串
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{c}' stroke-width='{sw}'{m}/>"  # 拼接 line 标签，写入两端坐标、颜色、线宽及可选箭头属性

def card(x,y,w,h,color,title,rows):  # 定义生成信息卡片的辅助函数
    """
    生成一张信息卡片的 SVG 片段。
    参数 x/y/w/h 为卡片位置与尺寸，color 为主题色，title 为标题文字，
    rows 为内容行列表，每项依次含文本、字号、是否加粗、是否斜体、颜色；
    返回由外框、标题条与各内容行拼接而成的 SVG 字符串。
    """
    # rows: list of (text, sz, bold, italic, fill)  # 补充：rows 中每项依次为（文本内容, 字号, 是否加粗, 是否斜体, 颜色）
    o=[]  # 初始化用于收集卡片各 SVG 片段的列表
    o.append(R(x,y,w,h,'#fff',color,1.5,7,"filter='url(#sh)'"))  # 追加卡片外框矩形，带阴影滤镜与主题色描边
    o.append(R(x,y,w,24,color,color,0,7))  # 追加卡片顶部彩色标题条矩形
    o.append(R(x,y+12,w,12,color,color,0,0))  # 追加一个覆盖标题条上半部分的矩形，形成平整的标题栏效果
    o.append(T(x+w//2,y+15,title,10,'#fff',bold=True))  # 追加居中显示的白色粗体标题文字
    for i,(s,sz,bo,it,fc) in enumerate(rows):  # 遍历内容行列表，enumerate 同时取得行号与该行的格式参数
        o.append(T(x+w//2,y+33+i*15,s,sz,fc,bold=bo,italic=it))  # 按行号计算纵向偏移，追加一行格式化后的文本元素
    return '\n'.join(o)  # 用换行符把所有 SVG 片段拼接成完整字符串并返回

# SVG 公共定义字符串：包含全局字体样式、五种箭头标记与阴影滤镜，供各图复用。
DEFS = """<defs>
<style>.fn{font-family:'Segoe UI',Arial,sans-serif;}</style>
<marker id='ab' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#3b82f6'/></marker>
<marker id='ag' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#059669'/></marker>
<marker id='ao' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#d97706'/></marker>
<marker id='ap' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#7c3aed'/></marker>
<marker id='ar' markerWidth='8' markerHeight='6' refX='7' refY='3' orient='auto'><polygon points='0 0,8 3,0 6' fill='#dc2626'/></marker>
<filter id='sh'><feDropShadow dx='1' dy='2' stdDeviation='3' flood-color='#00000018'/></filter>
</defs>"""
























































































































































