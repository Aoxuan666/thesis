"""
一周环境参数变化绘图与统计脚本。

本脚本读取 GBK 编码的温湿度数据，筛选 2024 年 3 月 27 日起一周的记录，
用 2x2 子图分别绘制温度、湿度、风速与 CO2 浓度的变化趋势，
把图表保存为“环境参数一周变化.png”，并在终端打印这四项指标的均值、极值统计。
"""
import pandas as pd  # 导入 pandas 数据分析库并简写为 pd
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 绘图接口并简写为 plt
from matplotlib import rcParams  # 导入 matplotlib 的全局参数配置对象 rcParams
from datetime import datetime, timedelta  # 导入 datetime 与 timedelta，用于构造日期与计算一周时间跨度

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 指定 sans-serif 字体为黑体，保证中文标签正常显示
rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示，避免坐标轴负号显示异常

# 读取数据
df = pd.read_csv('温湿度数据111.csv', encoding='gbk')  # 以 GBK 编码读取环境温湿度 CSV 数据文件
df['时间'] = pd.to_datetime(df['时间'])  # 把“时间”列转换为 pandas 的 datetime 时间类型

# 筛选3月27日到4月2日（一周）的数据
start_date = datetime(2024, 3, 27)  # 设定数据统计的起始日期为 2024 年 3 月 27 日
end_date = start_date + timedelta(days=7)  # 用 timedelta 计算出一周后的结束日期（4 月 3 日，不含）
df_week = df[(df['时间'] >= start_date) & (df['时间'] < end_date)]  # 按时间区间筛选出一整周的数据子集

print(f"数据时间范围: {df_week['时间'].min()} 到 {df_week['时间'].max()}")  # 打印筛选后数据的实际起止时间
print(f"共 {len(df_week)} 条数据")  # 打印筛选后数据的总条数

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 创建 2 行 2 列、尺寸为 16x10 英寸的子图画布
fig.suptitle('3月27日-4月2日环境参数变化情况', fontsize=16, fontweight='bold')  # 设置整张图的中文主标题

# 1. 温度变化
ax1 = axes[0, 0]  # 取左上角子图用于绘制温度变化
ax1.plot(df_week['时间'], df_week['当前温度'], color='#e74c3c', linewidth=1.5, label='当前温度')  # 用红色实线绘制当前温度随时间的变化曲线
ax1.plot(df_week['时间'], df_week['目标温度'], color='#3498db', linewidth=1.5, linestyle='--', label='目标温度')  # 用蓝色虚线绘制目标温度随时间的变化曲线
ax1.fill_between(df_week['时间'], df_week['当前温度'], alpha=0.3, color='#e74c3c')  # 用半透明红色填充当前温度曲线下方的区域
ax1.set_xlabel('时间', fontsize=11)  # 设置温度子图的横轴标签
ax1.set_ylabel('温度 (°C)', fontsize=11)  # 设置温度子图的纵轴标签（摄氏度）
ax1.set_title('温度变化', fontsize=13, fontweight='bold')  # 设置温度子图的标题
ax1.legend(loc='best')  # 显示温度子图的图例，位置自动选择最优处
ax1.grid(True, alpha=0.3)  # 为温度子图添加半透明网格线
ax1.tick_params(axis='x', rotation=45)  # 将温度子图横轴刻度标签旋转 45 度避免重叠

# 2. 湿度变化
ax2 = axes[0, 1]  # 取右上角子图用于绘制湿度变化
ax2.plot(df_week['时间'], df_week['当前湿度'], color='#3498db', linewidth=1.5, label='当前湿度')  # 用蓝色实线绘制当前湿度随时间的变化曲线
ax2.plot(df_week['时间'], df_week['目标湿度'], color='#2ecc71', linewidth=1.5, linestyle='--', label='目标湿度')  # 用绿色虚线绘制目标湿度随时间的变化曲线
ax2.fill_between(df_week['时间'], df_week['当前湿度'], alpha=0.3, color='#3498db')  # 用半透明蓝色填充当前湿度曲线下方的区域
ax2.set_xlabel('时间', fontsize=11)  # 设置湿度子图的横轴标签
ax2.set_ylabel('湿度 (%)', fontsize=11)  # 设置湿度子图的纵轴标签（百分比）
ax2.set_title('湿度变化', fontsize=13, fontweight='bold')  # 设置湿度子图的标题
ax2.legend(loc='best')  # 显示湿度子图的图例
ax2.grid(True, alpha=0.3)  # 为湿度子图添加半透明网格线
ax2.tick_params(axis='x', rotation=45)  # 将湿度子图横轴刻度标签旋转 45 度

# 3. 风速变化
ax3 = axes[1, 0]  # 取左下角子图用于绘制风速变化
ax3.plot(df_week['时间'], df_week['风速'], color='#9b59b6', linewidth=1.5)  # 用紫色实线绘制风速随时间的变化曲线
ax3.fill_between(df_week['时间'], df_week['风速'], alpha=0.3, color='#9b59b6')  # 用半透明紫色填充风速曲线下方的区域
ax3.set_xlabel('时间', fontsize=11)  # 设置风速子图的横轴标签
ax3.set_ylabel('风速', fontsize=11)  # 设置风速子图的纵轴标签
ax3.set_title('风速变化', fontsize=13, fontweight='bold')  # 设置风速子图的标题
ax3.grid(True, alpha=0.3)  # 为风速子图添加半透明网格线
ax3.tick_params(axis='x', rotation=45)  # 将风速子图横轴刻度标签旋转 45 度

# 4. CO2变化
ax4 = axes[1, 1]  # 取右下角子图用于绘制 CO2 浓度变化
ax4.plot(df_week['时间'], df_week['CO2'], color='#f39c12', linewidth=1.5)  # 用橙色实线绘制 CO2 浓度随时间的变化曲线
ax4.fill_between(df_week['时间'], df_week['CO2'], alpha=0.3, color='#f39c12')  # 用半透明橙色填充 CO2 曲线下方的区域
ax4.set_xlabel('时间', fontsize=11)  # 设置 CO2 子图的横轴标签
ax4.set_ylabel('CO2浓度 (ppm)', fontsize=11)  # 设置 CO2 子图的纵轴标签（ppm）
ax4.set_title('CO2浓度变化', fontsize=13, fontweight='bold')  # 设置 CO2 子图的标题
ax4.grid(True, alpha=0.3)  # 为 CO2 子图添加半透明网格线
ax4.tick_params(axis='x', rotation=45)  # 将 CO2 子图横轴刻度标签旋转 45 度

plt.tight_layout()  # 自动调整各子图间距，避免标签被裁切
plt.savefig('环境参数一周变化.png', dpi=300, bbox_inches='tight')  # 以 300 dpi 把整张图保存为 PNG 文件并紧凑裁切边缘
print("\n图表已保存为: 环境参数一周变化.png")  # 打印图片保存路径提示信息
plt.show()  # 在窗口中显示绘制好的图表

# 打印统计信息
print("\n=== 一周数据统计 ===")  # 打印一周数据统计的开始分隔标题
print(f"\n温度统计:")  # 打印温度统计的小标题
print(f"  平均温度: {df_week['当前温度'].mean():.2f}°C")  # 打印当前温度的平均值（保留两位小数）
print(f"  最高温度: {df_week['当前温度'].max():.2f}°C")  # 打印当前温度的最高值
print(f"  最低温度: {df_week['当前温度'].min():.2f}°C")  # 打印当前温度的最低值

print(f"\n湿度统计:")  # 打印湿度统计的小标题
print(f"  平均湿度: {df_week['当前湿度'].mean():.2f}%")  # 打印当前湿度的平均值
print(f"  最高湿度: {df_week['当前湿度'].max():.2f}%")  # 打印当前湿度的最高值
print(f"  最低湿度: {df_week['当前湿度'].min():.2f}%")  # 打印当前湿度的最低值

print(f"\n风速统计:")  # 打印风速统计的小标题
print(f"  平均风速: {df_week['风速'].mean():.2f}")  # 打印风速的平均值
print(f"  最大风速: {df_week['风速'].max():.2f}")  # 打印风速的最大值
print(f"  最小风速: {df_week['风速'].min():.2f}")  # 打印风速的最小值

print(f"\nCO2统计:")  # 打印 CO2 统计的小标题
print(f"  平均CO2: {df_week['CO2'].mean():.2f} ppm")  # 打印 CO2 浓度的平均值（ppm）
print(f"  最高CO2: {df_week['CO2'].max():.2f} ppm")  # 打印 CO2 浓度的最高值
print(f"  最低CO2: {df_week['CO2'].min():.2f} ppm")  # 打印 CO2 浓度的最低值





































































































































































































