"""
模型框架图绘制脚本：Wavelet-ECA-GRU 鸡舍温度偏差短期预测框架。

使用 matplotlib 绘制六列式流程示意图（数据采集、数据预处理、小波分解、
ECA 注意力、GRU 编码、预测与预警），并通过 FancyBboxPatch 圆角框与箭头表达各模块关系。
脚本使用 Agg 后端，直接输出静态图像，不依赖图形界面。
"""

import matplotlib  # 导入 matplotlib 基础模块，为设置后端做准备
matplotlib.use('Agg')  # 设置 Agg 非交互后端，便于无图形界面下保存图像
import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制框架图
from matplotlib.patches import FancyBboxPatch  # 导入圆角矩形补丁类，用于绘制各模块框

plt.rcParams.update({  # 统一更新全局字体等绘图参数
    'font.family': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],  # 字体族按优先级设置，优先使用中文字体
    'axes.unicode_minus': False,  # 设置正常显示负号，避免坐标轴负号显示为方块
})

BG='#F7F9FC'; W='#FFFFFF'; BORDER='#D0D7E3'  # 定义画布背景色、白色填充与边框色等基础配色
C_DATA='#3B82F6'; C_WAV='#10B981'; C_ECA='#F59E0B'  # 定义数据采集、小波分解、ECA 模块的主题色
C_GRU='#8B5CF6'; C_OUT='#EF4444'; C_ARR='#64748B'  # 定义 GRU、输出与箭头的主题色
TW='#1E293B'; TL='#64748B'  # 定义标题文字与辅助说明文字的颜色

fig=plt.figure(figsize=(22,14),facecolor=BG)  # 创建 22×14 英寸的画布并设置背景色
ax=fig.add_axes([0,0,1,1])  # 添加一个铺满整个画布的坐标轴
ax.set_xlim(0,22); ax.set_ylim(0,14); ax.axis('off'); ax.set_facecolor(BG)  # 设置坐标范围并关闭坐标轴显示，同时设定坐标轴背景色

def box(ax,x,y,w,h,fc=W,ec=BORDER,lw=1.5,r=0.2,zorder=3):  # 定义绘制圆角矩形框的辅助函数
    """绘制一个圆角矩形框。
    参数: ax 目标坐标轴；x、y 为左下角坐标；w、h 为宽高；
          fc 填充色；ec 边框色；lw 线宽；r 圆角半径；zorder 绘制层级。
    """
    ax.add_patch(FancyBboxPatch((x,y),w,h,  # 向坐标轴添加一个圆角矩形补丁
        boxstyle=f'round,pad=0,rounding_size={r}',  # 设置圆角样式与圆角半径
        facecolor=fc,edgecolor=ec,linewidth=lw,zorder=zorder))  # 设置填充色、边框色、线宽与绘制层级

def t(ax,x,y,s,sz=9,c=TW,ha='center',va='center',bold=False,zorder=5):  # 定义添加文本标注的辅助函数
    """在指定位置添加文本标注。
    参数: ax 目标坐标轴；x、y 为文本位置；s 为文本内容；sz 字号；c 颜色；
          ha、va 为水平与垂直对齐方式；bold 是否加粗；zorder 绘制层级。
    """
    ax.text(x,y,s,fontsize=sz,color=c,ha=ha,va=va,  # 在指定坐标处添加文本内容
            fontweight='bold' if bold else 'normal',linespacing=1.35,zorder=zorder)  # 设置字重（是否加粗）、行距与绘制层级

def arr(ax,x0,y0,x1,y1,c=C_ARR,lw=1.8):  # 定义绘制箭头的辅助函数
    """绘制一条由起点指向终点的箭头，用于连接各流程模块。
    参数: ax 目标坐标轴；(x0,y0) 起点；(x1,y1) 终点；c 颜色；lw 线宽。
    """
    ax.annotate('',xy=(x1,y1),xytext=(x0,y0),  # 用 annotate 绘制从起点指向终点的箭头
        arrowprops=dict(arrowstyle='->,head_width=0.14,head_length=0.18',color=c,lw=lw),zorder=6)  # 设置箭头样式、颜色与线宽

def hdr(ax,x,y,w,h,color,label,sz=9):  # 定义绘制带顶部标题栏的模块框的辅助函数
    """绘制带顶部彩色标题栏的模块框，返回后可用于在其中继续添加内容。
    参数: ax 目标坐标轴；x、y、w、h 为框的位置与尺寸；color 标题栏颜色；
          label 标题文字；sz 字号。
    """
    box(ax,x,y+h-0.40,w,0.40,fc=color,ec=color,r=0.15,zorder=4)  # 在模块框顶部绘制一条彩色标题栏
    box(ax,x,y,w,h,fc=W,ec=color,lw=1.8,r=0.18,zorder=3)  # 绘制模块主体的白色圆角框
    t(ax,x+w/2,y+h-0.20,label,sz=sz,c='white',bold=True)  # 在标题栏上绘制白色加粗的标题文字

COLS=[0.4,4.0,7.6,11.2,14.6,18.1]  # 定义六列布局的左边界 x 坐标
CW=[3.3,3.3,3.3,3.1,3.2,3.5]  # 定义六列布局中每一列的宽度
CMID=[c+cw/2 for c,cw in zip(COLS,CW)]  # 计算各列的中心 x 坐标，便于文字居中

# 背景分栏
for x,cw,cb in zip(COLS,CW,[C_DATA+'18',C_DATA+'10',C_WAV+'18',C_ECA+'18',C_GRU+'18',C_OUT+'18']):  # 遍历各列，准备绘制背景分栏
    box(ax,x,1.2,cw,11.1,fc=cb,ec=BORDER+'88',lw=0.8,r=0.3,zorder=1)  # 为每一列绘制一个浅色背景矩形分栏

# 列标题
for i,(lb,col) in enumerate(zip(  # 遍历列标题文本与对应主题色
    ['1.数据采集','2.数据预处理','3.小波分解','4.ECA注意力','5.GRU编码','6.预测与预警'],  # 六列的标题文本列表
    [C_DATA,C_DATA,C_WAV,C_ECA,C_GRU,C_OUT])):  # 六列标题分别对应的主题色列表
    box(ax,COLS[i]+0.1,11.7,CW[i]-0.2,0.42,fc=col,ec=col,r=0.15,zorder=4)  # 为每一列绘制标题色块
    t(ax,CMID[i],11.91,lb,sz=8.5,c='white',bold=True)  # 在标题色块上绘制白色加粗的列标题

# 总标题
t(ax,11,13.45,'Wavelet-ECA-GRU  鸡舍温度偏差(TD)短期预测框架',sz=16,bold=True,c=TW)  # 绘制整幅框架图的总标题
t(ax,11,13.0,'输入窗口 L=36步(180min)  ->  预测偏移 H=6步(30min 超前预测)',sz=10,c=TL)  # 绘制输入窗口与预测偏移的说明副标题
