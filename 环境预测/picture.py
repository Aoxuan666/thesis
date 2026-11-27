"""
鸡舍温度偏差预测框架示意图绘制脚本。

本脚本使用 matplotlib 绘制 Wavelet-ECA-GRU 短期预测框架的流程示意图，
依次展示数据采集、数据预处理、小波分解、ECA 注意力、GRU 编码与预测预警六个阶段，
并通过圆角方框、彩色分栏和箭头呈现整体数据流。输出为静态图片。
"""
import matplotlib  # 导入 matplotlib 绘图库
matplotlib.use('Agg')  # 指定 Agg 非交互式后端，便于在无图形界面环境下保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 模块并简写为 plt
from matplotlib.patches import FancyBboxPatch  # 导入圆角矩形补丁类，用于绘制流程图的方框

plt.rcParams.update({  # 批量更新全局绘图参数
    'font.family': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],  # 设置中文字体候选列表，保证中文标题能够正常显示
    'axes.unicode_minus': False,  # 关闭 unicode 负号显示，避免坐标轴负号显示异常
})  # 参数字典定义结束

BG='#F7F9FC'; W='#FFFFFF'; BORDER='#D0D7E3'  # 定义画布背景色、白色与边框色
C_DATA='#3B82F6'; C_WAV='#10B981'; C_ECA='#F59E0B'  # 定义数据采集、小波分解与注意力模块的主题色
C_GRU='#8B5CF6'; C_OUT='#EF4444'; C_ARR='#64748B'  # 定义 GRU 编码、预测输出与箭头的主题色
TW='#1E293B'; TL='#64748B'  # 定义标题文字色与说明文字色

fig=plt.figure(figsize=(22,14),facecolor=BG)  # 创建 22x14 英寸画布并设置背景色
ax=fig.add_axes([0,0,1,1])  # 在画布上添加铺满整个画布的坐标轴
ax.set_xlim(0,22); ax.set_ylim(0,14); ax.axis('off'); ax.set_facecolor(BG)  # 设置坐标范围、隐藏坐标轴刻度并统一背景色

def box(ax,x,y,w,h,fc=W,ec=BORDER,lw=1.5,r=0.2,zorder=3):  # 定义绘制圆角矩形方框的辅助函数
    """
    绘制圆角矩形方框。
    参数 ax 为坐标轴，x/y 为左下角坐标，w/h 为宽高；
    fc/ec/lw 分别控制填充色、边框色与线宽，r 为圆角半径，zorder 控制绘制层级。
    无返回值，直接把补丁添加到坐标轴。
    """
    ax.add_patch(FancyBboxPatch((x,y),w,h,  # 向坐标轴添加一个圆角矩形补丁，参数为左下角坐标以及宽度和高度
        boxstyle=f'round,pad=0,rounding_size={r}',  # 通过 boxstyle 指定圆角样式与圆角半径
        facecolor=fc,edgecolor=ec,linewidth=lw,zorder=zorder))  # 设置填充色、边框色、线宽与绘制层级，语句在此闭合

def t(ax,x,y,s,sz=9,c=TW,ha='center',va='center',bold=False,zorder=5):  # 定义绘制文本内容的辅助函数
    """
    在指定坐标绘制文本内容。
    参数 x/y 为文字位置，s 为文本，sz 为字号，c 为文字颜色；
    ha/va 为水平与垂直对齐方式，bold 控制是否加粗，zorder 控制绘制层级。
    无返回值。
    """
    ax.text(x,y,s,fontsize=sz,color=c,ha=ha,va=va,  # 在指定坐标绘制文字，并设置字号、颜色以及对齐方式
            fontweight='bold' if bold else 'normal',linespacing=1.35,zorder=zorder)  # 根据 bold 决定字重，同时设置行距与绘制层级

def arr(ax,x0,y0,x1,y1,c=C_ARR,lw=1.8):  # 定义绘制指示箭头的辅助函数
    """
    绘制从起点指向终点的箭头。
    参数 x0/y0 为起点坐标，x1/y1 为终点坐标，c 为箭头颜色，lw 为线宽；
    箭头样式固定为带宽头的箭头，通过 zorder 置于较上层。无返回值。
    """
    ax.annotate('',xy=(x1,y1),xytext=(x0,y0),  # 使用 annotate 绘制一条由起点指向终点的带箭头连线
        arrowprops=dict(arrowstyle='->,head_width=0.14,head_length=0.18',color=c,lw=lw),zorder=6)  # 设置箭头样式、颜色、线宽与绘制层级

def hdr(ax,x,y,w,h,color,label,sz=9):  # 定义绘制带彩色标题栏卡片的辅助函数
    """
    绘制带彩色标题栏的卡片。
    参数 x/y/w/h 为卡片位置与尺寸，color 为标题栏及边框颜色，
    label 为标题文字，sz 为标题字号。无返回值，依次叠加标题条与主体框。
    """
    box(ax,x,y+h-0.40,w,0.40,fc=color,ec=color,r=0.15,zorder=4)  # 在卡片顶部绘制一条彩色标题栏
    box(ax,x,y,w,h,fc=W,ec=color,lw=1.8,r=0.18,zorder=3)  # 绘制白色主体的卡片边框框体
    t(ax,x+w/2,y+h-0.20,label,sz=sz,c='white',bold=True)  # 在标题栏中央写入白色粗体标题文字

COLS=[0.4,4.0,7.6,11.2,14.6,18.1]  # 定义六个流程列的横坐标起点
CW=[3.3,3.3,3.3,3.1,3.2,3.5]  # 定义六个流程列各自的宽度
CMID=[c+cw/2 for c,cw in zip(COLS,CW)]  # 计算每一列中心点横坐标，便于文字居中绘制

# 背景分栏
for x,cw,cb in zip(COLS,CW,[C_DATA+'18',C_DATA+'10',C_WAV+'18',C_ECA+'18',C_GRU+'18',C_OUT+'18']):  # 遍历各列的位置、宽度与带透明度的背景颜色
    box(ax,x,1.2,cw,11.1,fc=cb,ec=BORDER+'88',lw=0.8,r=0.3,zorder=1)  # 为每一列绘制淡色背景分栏框

# 列标题
for i,(lb,col) in enumerate(zip(  # 遍历六个列标题文字及其对应的主题色
    ['1.数据采集','2.数据预处理','3.小波分解','4.ECA注意力','5.GRU编码','6.预测与预警'],  # 六个流程阶段的中文标题文字列表
    [C_DATA,C_DATA,C_WAV,C_ECA,C_GRU,C_OUT])):  # 六个标题对应的颜色列表，与标题列表配对后逐项处理
    box(ax,COLS[i]+0.1,11.7,CW[i]-0.2,0.42,fc=col,ec=col,r=0.15,zorder=4)  # 在每一列顶部绘制彩色标题条
    t(ax,CMID[i],11.91,lb,sz=8.5,c='white',bold=True)  # 在标题条内部居中写入白色标题文字

# 总标题
t(ax,11,13.45,'Wavelet-ECA-GRU  鸡舍温度偏差(TD)短期预测框架',sz=16,bold=True,c=TW)  # 绘制整张示意图的总标题
t(ax,11,13.0,'输入窗口 L=36步(180min)  ->  预测偏移 H=6步(30min 超前预测)',sz=10,c=TL)  # 绘制输入窗口与预测偏移的说明副标题