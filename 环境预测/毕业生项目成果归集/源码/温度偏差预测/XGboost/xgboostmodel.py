# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8
"""
XGBoost 温度偏差预测模型推理脚本。

脚本加载训练阶段保存的 Model_XGBoost.pkl 检查点，读取测试数据并完成编码探测、
缺失值处理与特征归一化，再按检查点中的时间窗口和预测偏移构造展平特征矩阵。
随后调用模型预测、反归一化后计算 R2/MSE/RMSE/MAE，
并输出预测对比图、预测值与真实值散点图、误差直方图/时序图/箱线图以及误差统计信息。
"""
import sys  # 导入解释器与路径相关模块
import os  # 导入操作系统接口模块
# 防止与xgboost库同名冲突：从sys.path中移除当前脚本目录  # 补充：确保 import xgboost 导入第三方库而非当前同名脚本
_this_dir = os.path.dirname(os.path.abspath(__file__))  # 取得当前脚本所在目录的绝对路径
if _this_dir in sys.path:  # 判断当前目录是否位于模块搜索路径中
    sys.path.remove(_this_dir)  # 从搜索路径移除当前目录，避免同名模块冲突
# 说明：紧随其后的三引号字符串位于导入语句之后，属于普通字符串表达式（原文件即如此），此处原样保留。
"""
XGBoost模型推理脚本
用于加载训练好的XGBoost模型并进行预测
"""
import pandas as pd  # 导入 pandas 数据分析库
import numpy as np  # 导入 numpy 数值计算库
import pickle  # 导入 pickle 反序列化模块
import matplotlib  # 导入 matplotlib 绘图库
matplotlib.use('Agg')  # 指定 Agg 后端，支持无界面保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 归一化工具
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、均方误差与平均绝对误差评估函数

# ===================== 环境配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示
print("XGBoost 模型推理脚本")  # 打印脚本启动提示

# ===================== 数据加载 =====================
print("\n加载测试数据...")  # 打印开始加载测试数据的提示
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 列出按优先级尝试的 CSV 文件编码
data = None  # 初始化数据变量为空，表示尚未成功读取
for enc in encodings:  # 依次尝试每种编码读取测试数据
    try:  # 捕获解码异常以尝试下一种编码
        data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取测试数据 CSV
        print(f"成功使用 {enc} 编码加载测试数据")  # 打印成功使用的编码
        break  # 读取成功后停止尝试
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码相关异常
        continue  # 当前编码失败则继续尝试
if data is None:  # 若所有编码均失败则数据仍为空
    raise ValueError("无法使用任何已知编码读取测试数据文件")  # 抛出错误提示无法读取测试数据

data['timestamp'] = pd.to_datetime(data.index * 5, unit='m',  # 按 5 分钟间隔从固定起点为每行生成时间戳
                                   origin=pd.Timestamp('2025/6/21 0:04'))  # 时间戳生成的起始时刻参数

features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 定义候选特征列名列表
target = '温度偏差'  # 定义目标变量为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列

print(f"使用特征: {features}")  # 打印最终采用的特征列表
print(f"数据形状: {data.shape}")  # 打印数据规模

if target not in data.columns:  # 检查目标列是否存在于测试数据中
    raise KeyError(f"错误: 测试数据中缺少目标列 '{target}'")  # 缺少目标列时抛出明确错误

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印缺失值处理标题的分隔线
print("缺失值处理")  # 打印缺失值处理标题
print(f"{'='*60}")  # 打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据规模
for col in features + [target]:  # 遍历特征与目标列统计缺失情况
    if col in data.columns:  # 仅处理实际存在的列
        nan_count = data[col].isna().sum()  # 统计当前列缺失值个数
        if nan_count > 0:  # 仅在存在缺失时输出并处理
            print(f"  {col}: {nan_count} 个缺失值 ({nan_count/len(data)*100:.2f}%)")  # 打印该列缺失数量与占比

continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 定义需要线性插值的连续型变量
discrete_vars = ['日龄']  # 定义只需前后填充的离散型变量
for col in continuous_vars:  # 遍历连续型变量处理缺失值
    if col in data.columns and data[col].isna().sum() > 0:  # 仅对存在缺失的连续列处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').ffill().bfill()  # 先线性插值再用前向/后向填充兜底
for col in discrete_vars:  # 遍历离散型变量处理缺失值
    if col in data.columns and data[col].isna().sum() > 0:  # 仅对存在缺失的离散列处理
        data[col] = data[col].ffill().bfill()  # 用前向与后向填充补齐离散列缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计剩余缺失值总数
if remaining_nan > 0:  # 若仍有缺失则删除对应行
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印剩余缺失值告警信息
    data = data.dropna(subset=features + [target])  # 删除含缺失值的样本行
print(f"\n处理后数据形状: {data.shape}")  # 打印清洗后的数据规模
print(f"{'='*60}\n")  # 打印分隔线并换行
data = data.reset_index(drop=True)  # 重置索引，保证清洗后的行号从 0 连续编号

# ===================== 加载模型 =====================
print("\n加载训练好的模型...")  # 打印加载训练好模型的提示
MODEL_PATH = 'Model_XGBoost.pkl'  # 定义模型检查点文件路径
try:  # 捕获模型文件读取异常
    with open(MODEL_PATH, 'rb') as f:  # 以二进制只读模式打开模型文件
        checkpoint = pickle.load(f)  # 用 pickle 从文件中反序列化出检查点字典
    print(f"成功加载模型: {MODEL_PATH}")  # 打印模型加载成功提示
except FileNotFoundError:  # 捕获文件不存在的异常
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")  # 模型文件缺失时抛出明确错误

model         = checkpoint['model']  # 从检查点中取出训练好的模型对象
time_step     = checkpoint['time_step']  # 取出训练时保存的时间窗口长度
pred_offset   = checkpoint['pred_offset']  # 取出训练时保存的预测偏移步数
features      = checkpoint['features']  # 取出训练时使用的特征列表
feature_scaler = checkpoint['feature_scaler']  # 取出特征归一化器
target_scaler  = checkpoint['target_scaler']  # 取出目标归一化器

print(f"\n模型配置:")  # 打印模型配置标题
print(f"  时间步长: {time_step}")  # 打印时间步长
print(f"  预测偏移: {pred_offset}")  # 打印预测偏移
print(f"  输入特征: {features}")  # 打印输入特征列表

# ===================== 数据预处理 =====================
missing_features = [f for f in features if f not in data.columns]  # 找出测试数据中缺失的特征列
if missing_features:  # 若存在缺失特征则中止
    raise ValueError(f"测试数据缺少以下特征: {missing_features}")  # 抛出错误并列出缺失的特征

test_features_scaled = pd.DataFrame(  # 用保存的归一化器转换测试特征并重建 DataFrame
    feature_scaler.transform(data[features]), columns=features)  # 指定转换后 DataFrame 的列名
target_scaled = target_scaler.transform(data[[target]]).flatten()  # 转换测试集真实目标并展平为一维数组

# ===================== 序列构造 =====================
def create_flat_sequences(feature_df, target_arr, time_step, pred_offset):  # 定义把时间序列展平为二维特征矩阵的函数
    """
    将时间序列展平为XGBoost可用的2D特征矩阵。
    每个样本: time_step 个时间步的所有特征展平拼接。
    输出维度: (n_samples, time_step * n_features)
    参数 feature_df 为已归一化的特征 DataFrame，target_arr 为对应的目标数组；
    time_step 为输入窗口步数，pred_offset 为预测偏移步数。
    返回二元组 (X, y)：X 为展平后的二维特征矩阵，y 为每个窗口对应的目标值数组。
    """
    X, y = [], []  # 初始化特征列表与标签列表
    for i in range(len(feature_df) - time_step - pred_offset):  # 滑动窗口遍历，留出窗口与预测偏移长度
        window = feature_df.iloc[i:i + time_step].values.flatten()  # 取窗口内所有特征并展平成一个样本向量
        label = target_arr[i + time_step + pred_offset]  # 取预测偏移位置之后的目标值作为标签
        X.append(window)  # 保存当前窗口特征向量
        y.append(label)  # 保存当前样本标签
    return np.array(X), np.array(y)  # 转换为 numpy 数组并返回特征矩阵与标签数组

print("\n创建序列数据...")  # 打印构造序列数据的提示
X_test, y_test = create_flat_sequences(test_features_scaled, target_scaled, time_step, pred_offset)  # 构造测试集展平特征与标签
print(f"\n测试数据形状:")  # 打印测试数据形状标题
print(f"  X_test: {X_test.shape}")  # 打印测试集特征矩阵形状
print(f"  y_test: {y_test.shape}")  # 打印测试集标签数组形状

# ===================== 模型预测 =====================
print("\n开始预测...")  # 打印开始预测的提示
y_pred = model.predict(X_test).reshape(-1, 1)  # 用模型预测并把结果 reshape 成二维列向量

# 反归一化
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将真实目标反归一化回原始量纲
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测结果反归一化回原始量纲

# ===================== 评估指标 =====================
r2   = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse  = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse = np.sqrt(mse)  # 由 MSE 开方得到 RMSE
mae  = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print("\n" + "=" * 60)  # 打印 60 个等号作为分隔线
print("XGBoost模型测试结果")  # 打印模型测试结果标题
print("=" * 60)  # 打印分隔线
print(f"R2 Score:  {r2:.4f}")  # 打印 R2 得分
print(f"MSE:       {mse:.4f}")  # 打印 MSE
print(f"RMSE:      {rmse:.4f}")  # 打印 RMSE
print(f"MAE:       {mae:.4f}")  # 打印 MAE
print("=" * 60)  # 打印分隔线

# ===================== 结果可视化 =====================
plot_length = min(120, len(y_test_rescaled))  # 限定绘图长度为样本数与 120 的较小值
y_test_plot = y_test_rescaled[:plot_length].flatten()  # 截取用于绘图的真实值并展平
y_pred_plot = y_pred_rescaled[:plot_length].flatten()  # 截取用于绘图的预测值并展平
steps = np.arange(plot_length)  # 生成绘图用的横轴步数序列

plt.figure(figsize=(15, 8))  # 创建 15x8 英寸画布
plt.tick_params(axis='both', labelsize=16)  # 设置坐标轴刻度标签字号为 16
plt.plot(steps, y_test_plot, label='Real',      linewidth=2, alpha=1, color='#1f77b4')  # 用蓝色实线绘制真实温度偏差曲线
plt.plot(steps, y_pred_plot, label='Predicted', linewidth=2, alpha=1, color='#ff7f0e')  # 用橙色实线绘制模型预测曲线
plt.xlabel('Timestep', fontsize=16)  # 设置横轴标签
plt.ylabel('TD (℃)', fontsize=16)  # 设置纵轴标签
plt.title(f'XGBoost Model Prediction Results (R2={r2:.4f}, RMSE={rmse:.4f})', fontsize=14)  # 设置含评估指标的图表标题
plt.legend(fontsize=16)  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.text(0.5, 0.95, '(a)', transform=plt.gca().transAxes,  # 在图内上方添加 (a) 子图编号文字
         fontsize=16, fontweight='bold', va='top', ha='center')  # 设置该编号文字的字号、字重与对齐方式
plt.tight_layout()  # 自动调整布局避免标签被裁切
output_filename = '../../test_XGBoost_prediction.png'  # 设置预测对比图输出文件名
plt.savefig(output_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存预测对比图
print(f"\n预测结果图已保存: {output_filename}")  # 打印预测对比图保存提示

# ===================== 误差分析 =====================
errors = y_pred_rescaled.flatten() - y_test_rescaled.flatten()  # 计算预测值与真实值之间的误差

plt.figure(figsize=(15, 10))  # 创建 15x10 英寸画布用于误差分析

# 子图1: 预测值 vs 真实值散点图
plt.subplot(2, 2, 1)  # 选中第一个子图：预测值对真实值散点图
plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.5, s=20)  # 以真实值为横轴、预测值为纵轴绘制散点
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()],  # 绘制理想预测对角线（起点到终点）
         [y_test_rescaled.min(), y_test_rescaled.max()],  # 对角线的终点坐标
         'r--', lw=2, label='理想预测线')  # 用红色虚线表示理想预测线
plt.xlabel('真实温度偏差 (℃)', fontsize=11)  # 设置散点图横轴标签
plt.ylabel('预测温度偏差 (℃)', fontsize=11)  # 设置散点图纵轴标签
plt.title(f'预测值 vs 真实值 (R2={r2:.4f})', fontsize=12)  # 设置含 R2 的散点图标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格

# 子图2: 误差分布直方图
plt.subplot(2, 2, 2)  # 选中第二个子图：误差分布直方图
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)  # 绘制 50 个区间的误差直方图
plt.xlabel('预测误差 (℃)', fontsize=11)  # 设置直方图横轴标签
plt.ylabel('频数', fontsize=11)  # 设置直方图纵轴标签
plt.title(f'误差分布 (MAE={mae:.4f})', fontsize=12)  # 设置含 MAE 的标题
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)  # 在零误差处画红色参考虚线
plt.grid(True, alpha=0.3, axis='y')  # 仅为直方图添加横向网格

# 子图3: 误差时序图
plt.subplot(2, 2, 3)  # 选中第三个子图：误差时序图
error_steps = np.arange(min(120, len(errors)))  # 生成误差时序图的横轴步数（最多 120 步）
plt.plot(error_steps, errors[:120], alpha=0.7, linewidth=1)  # 绘制误差随时间变化的折线
plt.xlabel('Time Steps', fontsize=11)  # 设置横轴标签
plt.ylabel('Prediction Error (℃)', fontsize=11)  # 设置纵轴标签
plt.title('Error Time Series', fontsize=12)  # 设置标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 在零误差处画参考线
plt.grid(True, alpha=0.3)  # 添加网格

# 子图4: 误差箱线图
plt.subplot(2, 2, 4)  # 选中第四个子图：误差箱线图
plt.boxplot([errors], tick_labels=['XGBoost'])  # 用箱线图展示误差分布并指定类别标签
plt.ylabel('预测误差 (℃)', fontsize=11)  # 设置纵轴标签
plt.title('误差统计分布', fontsize=12)  # 设置标题
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 在零误差处画参考线
plt.grid(True, alpha=0.3, axis='y')  # 添加横向网格

plt.tight_layout()  # 自动调整误差分析图的子图间距
error_filename = '../../test_XGBoost_error_analysis.png'  # 设置误差分析图输出文件名
plt.savefig(error_filename, dpi=600, bbox_inches='tight')  # 以 600 dpi 保存误差分析图
print(f"误差分析图已保存: {error_filename}")  # 打印误差分析图保存提示

# ===================== 统计信息 =====================
print("\n" + "=" * 60)  # 打印统计信息标题分隔线
print("误差统计信息")  # 打印误差统计标题
print("=" * 60)  # 打印分隔线
print(f"误差均值:     {np.mean(errors):.4f} ℃")  # 打印误差均值
print(f"误差标准差:   {np.std(errors):.4f} ℃")  # 打印误差标准差
print(f"误差最大值:   {np.max(errors):.4f} ℃")  # 打印误差最大值
print(f"误差最小值:   {np.min(errors):.4f} ℃")  # 打印误差最小值
print(f"误差中位数:   {np.median(errors):.4f} ℃")  # 打印误差中位数
print("=" * 60)  # 打印分隔线

print("\n测试完成！")  # 打印脚本执行完成提示

