# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8
"""
XGBoost 温度偏差预测模型训练脚本。

脚本读取温湿度数据，经过编码探测、缺失值插值、按时间顺序 8:2 划分训练/测试集与 MinMax 归一化，
再把固定长度时间窗口内的多特征展平为二维矩阵；随后完成 5 折交叉验证、训练最终模型、
在测试集与新数据集上评估 R2/RMSE/MAE、绘制损失曲线与预测对比图，
并将最终模型和归一化器保存为 pickle 检查点。为避免与第三方 xgboost 库同名，
脚本在导入前会把自身目录从 sys.path 中移除。
"""
import sys  # 导入解释器与路径相关模块
import os  # 导入操作系统接口模块
import random  # 导入 Python 内置随机数模块
import pickle  # 导入 pickle 模块，用于序列化保存模型
# 防止与xgboost库同名冲突：从sys.path中移除当前脚本目录  # 补充：确保 import xgboost 导入第三方库而非当前同名脚本
_this_dir = os.path.dirname(os.path.abspath(__file__))  # 取得当前脚本所在目录的绝对路径
if _this_dir in sys.path:  # 判断当前目录是否位于模块搜索路径中
    sys.path.remove(_this_dir)  # 从搜索路径移除当前目录，避免同名模块冲突
import pandas as pd  # 导入 pandas 数据分析库
import numpy as np  # 导入 numpy 数值计算库
import matplotlib  # 导入 matplotlib 绘图库
matplotlib.use('Agg')  # 指定 Agg 后端，支持无界面保存图片
import matplotlib.pyplot as plt  # 导入 pyplot 绘图接口
from sklearn.preprocessing import MinMaxScaler  # 导入 MinMaxScaler 归一化工具
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # 导入 R2、均方误差与平均绝对误差评估函数
from sklearn.model_selection import KFold  # 导入 KFold 交叉验证划分器
import xgboost as xgb  # 导入 xgboost 库并简写为 xgb

# ===================== 随机种子设置 =====================
def set_seed(seed=42):  # 定义固定随机种子的函数，保证结果可复现
    """
    设置全局随机种子以保证实验可复现。
    参数 seed 为整数种子，默认 42；无返回值，
    分别固定 Python random、numpy.random 与 PYTHONHASHSEED 环境变量。
    """
    random.seed(seed)  # 固定 Python 内置随机数生成器的种子
    np.random.seed(seed)  # 固定 numpy 随机数生成器的种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 通过环境变量固定 Python 哈希种子
set_seed(42)  # 调用种子设置函数，使用默认种子 42

# ===================== 环境配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 关闭 unicode 负号显示
print(f"XGBoost版本: {xgb.__version__}")  # 打印当前使用的 XGBoost 版本

# ===================== 数据加载 =====================
encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']  # 列出按优先级尝试的 CSV 文件编码
data = None  # 初始化数据变量为空，表示尚未成功读取
for enc in encodings:  # 依次尝试每种编码读取训练数据
    try:  # 捕获解码异常，便于尝试下一种编码
        data = pd.read_csv('温湿度数据.csv', encoding=enc)  # 用当前编码读取训练数据 CSV
        print(f"成功使用 {enc} 编码加载训练数据")  # 打印成功使用的编码名称
        break  # 读取成功后停止尝试其他编码
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码相关异常
        continue  # 当前编码失败则继续尝试下一编码
if data is None:  # 若所有编码均失败则数据仍为空
    raise ValueError("无法使用任何已知编码读取训练数据文件")  # 抛出明确错误，提示无法读取训练数据

data['timestamp'] = pd.date_range(start='2025/5/26 8:22', periods=len(data), freq='5min')  # 按固定起始时间和 5 分钟频率为每个样本生成时间戳
features = ['日龄', '温度偏差', '外界温度', 'CO2', '目标温度']  # 定义候选特征列名列表
target = '温度偏差'  # 定义目标变量为温度偏差
features = [f for f in features if f in data.columns]  # 仅保留数据中实际存在的特征列

# ===================== 缺失值处理 =====================
print(f"\n{'='*60}")  # 打印 60 个等号作为分隔线
print("缺失值处理")  # 打印缺失值处理环节标题
print(f"{'='*60}")  # 再次打印分隔线
print(f"原始数据形状: {data.shape}")  # 打印原始数据行列规模
for col in features + [target]:  # 遍历所有特征与目标列，统计各自缺失情况
    if col in data.columns:  # 仅处理数据中实际存在的列
        nan_count = data[col].isna().sum()  # 统计当前列的缺失值个数
        if nan_count > 0:  # 仅当存在缺失值时输出并处理
            print(f"  {col}: {nan_count} 个缺失值 ({nan_count/len(data)*100:.2f}%)")  # 打印该列缺失数量与占比

continuous_vars = ['温度偏差', '外界温度', 'CO2', '目标温度']  # 定义需要线性插值的连续型变量列表
discrete_vars = ['日龄']  # 定义只需前后填充的离散型变量列表
for col in continuous_vars:  # 遍历连续型变量进行缺失值插值
    if col in data.columns and data[col].isna().sum() > 0:  # 仅对存在缺失的连续列进行处理
        data[col] = data[col].interpolate(method='linear', limit_direction='both').ffill().bfill()  # 先线性插值，再以前向/后向填充兜底
for col in discrete_vars:  # 遍历离散型变量进行缺失值填充
    if col in data.columns and data[col].isna().sum() > 0:  # 仅对存在缺失的离散列进行处理
        data[col] = data[col].ffill().bfill()  # 用前向填充与后向填充补齐离散列缺失值
remaining_nan = data[features + [target]].isna().sum().sum()  # 统计特征与目标列中剩余的缺失值总数
if remaining_nan > 0:  # 若仍有缺失值则删除对应行
    print(f"\n警告: 仍有 {remaining_nan} 个缺失值，将删除这些行")  # 打印剩余缺失值告警信息
    data = data.dropna(subset=features + [target])  # 删除特征或目标列含缺失值的样本行
print(f"\n处理后数据形状: {data.shape}")  # 打印清洗后的数据规模
print(f"{'='*60}\n")  # 打印分隔线并换行

print("\n【XGBoost模型】")  # 打印当前使用的模型名称
print(f"使用特征: {features}")  # 打印最终采用的特征列表
print(f"特征数量: {len(features)}")  # 打印特征数量

# ===================== 数据集划分 =====================
time_step = 36  # 定义输入时间窗口为 36 步
pred_offset = 6  # 定义预测偏移为 6 步
print(f"\n时间窗口配置:")  # 打印时间窗口配置标题
print(f"  时间窗口: {time_step} 步 ({time_step * 5} 分钟)")  # 换算并打印时间窗口对应的分钟数
print(f"  预测偏移: {pred_offset} 步 ({pred_offset * 5} 分钟)")  # 换算并打印预测偏移对应的分钟数

split_index = int(len(data) * 0.8)  # 取前 80% 样本作为训练集，计算切分下标
train_data = data.iloc[:split_index].copy()  # 按切分下标取得训练集副本
test_data = data.iloc[split_index:].copy()  # 按切分下标取得测试集副本
print(f"\n数据集划分:")  # 打印数据集划分标题
print(f"训练集大小: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")  # 打印训练集样本数及占比
print(f"测试集大小: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")  # 打印测试集样本数及占比

# ===================== 归一化 =====================
print("\n归一化处理...")  # 打印归一化处理进度提示
feature_scaler = MinMaxScaler()  # 创建特征归一化器
target_scaler = MinMaxScaler()  # 创建目标值归一化器
feature_scaler.fit(train_data[features])  # 仅在训练集特征上拟合归一化器，避免数据泄漏
target_scaler.fit(train_data[[target]])  # 仅在训练集目标上拟合目标归一化器
train_features_scaled = pd.DataFrame(feature_scaler.transform(train_data[features]), columns=features)  # 用训练集拟合的归一化器转换训练特征并重建 DataFrame
test_features_scaled = pd.DataFrame(feature_scaler.transform(test_data[features]), columns=features)  # 用同一归一化器转换测试特征
train_target_scaled = target_scaler.transform(train_data[[target]]).flatten()  # 转换训练目标并展平为一维数组
test_target_scaled = target_scaler.transform(test_data[[target]]).flatten()  # 转换测试目标并展平为一维数组

# ===================== 滑动窗口特征构造 =====================
def create_flat_sequences(feature_df, target_arr, time_step, pred_offset):  # 定义把时间序列展平为二维特征矩阵的函数
    """
    将时间序列展平为XGBoost可用的2D特征矩阵。
    每个样本: time_step 个时间步的所有特征展平拼接。
    输出维度: (n_samples, time_step * n_features)
    参数 feature_df 为归一化后的特征 DataFrame，target_arr 为对应的目标数组；
    time_step 为输入窗口步数，pred_offset 为预测偏移步数。
    返回二元组 (X, y)：X 为展平后的二维特征矩阵，y 为每个窗口对应的目标值数组。
    """
    X, y = [], []  # 初始化特征列表与标签列表
    for i in range(len(feature_df) - time_step - pred_offset):  # 滑动窗口遍历，留出时间窗口与预测偏移的长度
        window = feature_df.iloc[i:i + time_step].values.flatten()  # 取窗口内所有特征并展平成一维向量作为一个样本
        label = target_arr[i + time_step + pred_offset]  # 取偏移位置之后的目标值作为该样本的标签
        X.append(window)  # 保存当前窗口特征向量
        y.append(label)  # 保存当前样本对应的标签
    return np.array(X), np.array(y)  # 把列表转换为 numpy 数组并返回特征矩阵与标签数组

print("\n创建训练集序列...")  # 打印构造训练集序列的提示
X_train, y_train = create_flat_sequences(train_features_scaled, train_target_scaled, time_step, pred_offset)  # 调用展平函数构造训练集特征与标签
print("\n创建测试集序列...")  # 打印构造测试集序列的提示
X_test, y_test = create_flat_sequences(test_features_scaled, test_target_scaled, time_step, pred_offset)  # 调用展平函数构造测试集特征与标签
print(f"\n序列数据形状:")  # 打印序列数据形状标题
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")  # 打印训练集特征与标签的形状
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")  # 打印测试集特征与标签的形状
print(f"每个样本特征数: {X_train.shape[1]} (= {time_step} 步 x {len(features)} 特征)")  # 打印单个样本的特征数及其计算式

# ===================== XGBoost 超参数 =====================
xgb_params = {  # 定义 XGBoost 超参数字典
    'n_estimators': 500,  # 树的数量（boosting 轮数）
    'max_depth': 6,  # 树的最大深度
    'learning_rate': 0.05,  # 学习率（步长）
    'subsample': 0.8,  # 每棵树使用的样本子采样比例
    'colsample_bytree': 0.8,  # 每棵树使用的特征子采样比例
    'min_child_weight': 3,  # 叶子节点最小样本权重和
    'reg_alpha': 0.1,  # L1 正则化系数
    'reg_lambda': 1.0,  # L2 正则化系数
    'objective': 'reg:squarederror',  # 目标函数为平方误差回归
    'eval_metric': 'rmse',  # 评估指标为 RMSE
    'random_state': 42,  # 固定随机状态保证结果可复现
    'n_jobs': -1,  # 使用全部 CPU 核心并行训练
    'tree_method': 'hist',  # 使用 hist 直方图算法加速训练
}
print("\nXGBoost 超参数:")  # 超参数字典定义结束
for k, v in xgb_params.items():  # 打印超参数标题
    print(f"  {k}: {v}")  # 遍历并逐项打印超参数
  # 打印单个超参数的键与值
# ===================== K折交叉验证 =====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_losses_all = []  # 创建 5 折交叉验证划分器（打乱数据并固定种子）
history_list = []  # 初始化保存各折最佳验证损失的列表
print("\n========== 开始5折交叉验证（仅训练集） ==========")  # 初始化保存各折训练/验证损失曲线的列表
print(f"交叉验证数据量: {len(X_train)} 样本")  # 打印开始交叉验证的提示
  # 打印参与交叉验证的样本数量
for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
    print(f"\n{'=' * 50}")  # 遍历每一折，enumerate 从 1 开始编号
    print(f"第 {fold}/5 折训练")  # 打印该折的分隔线
    print(f"{'=' * 50}")  # 打印当前是第几折训练
    X_tr, X_val = X_train[train_index], X_train[val_index]  # 打印该折的分隔线
    y_tr, y_val = y_train[train_index], y_train[val_index]  # 按索引划分本折的训练特征与验证特征
    print(f"训练集样本数: {len(X_tr)}")  # 按索引划分本折的训练标签与验证标签
    print(f"验证集样本数: {len(X_val)}")  # 打印本折训练集样本数
    fold_model = xgb.XGBRegressor(**xgb_params)  # 打印本折验证集样本数
    fold_model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=50)  # 用统一超参数创建本折的回归模型
    evals_result = fold_model.evals_result()  # 在训练集上拟合，并用训练集与验证集作为评估集每 50 轮打印一次
    train_mse = [v ** 2 for v in evals_result['validation_0']['rmse']]  # 取得训练过程中记录的评估结果
    val_mse   = [v ** 2 for v in evals_result['validation_1']['rmse']]  # 把训练 RMSE 平方换算成 MSE 曲线
    history_list.append({'train_losses': train_mse, 'val_losses': val_mse})  # 把验证 RMSE 平方换算成 MSE 曲线
    best_val_loss = min(val_mse)  # 把本折两条损失曲线加入历史列表
    val_losses_all.append(best_val_loss)  # 取本折验证损失的最小值作为最佳成绩
    print(f"\n第 {fold} 折最佳验证损失 (MSE): {best_val_loss:.6f}")  # 记录本折最佳验证损失
  # 打印本折最佳验证损失
print(f"\n{'=' * 50}")
print("5折交叉验证结果汇总")  # 打印交叉验证汇总标题行的分隔线
print(f"{'=' * 50}")  # 打印汇总标题
for i, loss in enumerate(val_losses_all, 1):  # 打印汇总标题行的分隔线
    print(f"第 {i} 折验证损失: {loss:.6f}")  # 逐折打印最佳验证损失
print(f"\n平均验证损失: {np.mean(val_losses_all):.6f}")  # 打印各折验证损失的平均值
print(f"验证损失标准差: {np.std(val_losses_all):.6f}")  # 打印各折验证损失的标准差

# ===================== 绘制K折曲线 =====================
plt.figure(figsize=(15, 10))  # 创建 15x10 英寸画布用于绘制交叉验证结果
plt.subplot(2, 2, 1)  # 选中第一个子图，绘制各折训练损失曲线
for i, history in enumerate(history_list, 1):  # 遍历历史列表，逐折绘制曲线
    plt.plot(history['train_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制单折训练损失并添加图例标签
plt.title('5折交叉验证 - 训练损失曲线', fontsize=14)  # 设置第一个子图标题
plt.xlabel('迭代轮数 (Estimators)')  # 设置第一个子图横轴标签
plt.ylabel('损失值 (MSE)')  # 设置第一个子图纵轴标签
plt.legend()  # 显示第一个子图图例
plt.grid(True, alpha=0.3)  # 为第一个子图添加网格
plt.subplot(2, 2, 2)  # 选中第二个子图，绘制各折验证损失曲线
for i, history in enumerate(history_list, 1):  # 再次遍历历史列表逐折绘制
    plt.plot(history['val_losses'], label=f'Fold {i}', alpha=0.7)  # 绘制单折验证损失曲线
plt.title('5折交叉验证 - 验证损失曲线', fontsize=14)  # 设置第二个子图标题
plt.xlabel('迭代轮数 (Estimators)')  # 设置第二个子图横轴标签
plt.ylabel('损失值 (MSE)')  # 设置第二个子图纵轴标签
plt.legend()  # 显示第二个子图图例
plt.grid(True, alpha=0.3)  # 为第二个子图添加网格
plt.subplot(2, 2, 3)  # 选中第三个子图，绘制各折最佳验证损失柱状图
folds_labels = [f'Fold {i}' for i in range(1, 6)]  # 生成各折的标签文本
plt.bar(folds_labels, val_losses_all, color='steelblue', alpha=0.7)  # 以柱状图展示各折最佳验证损失
plt.axhline(y=np.mean(val_losses_all), color='red', linestyle='--',  # 画一条红色虚线表示平均验证损失
            label=f'平均值: {np.mean(val_losses_all):.6f}')  # 在平均线上标注平均值图例内容
plt.title('各折最佳验证损失对比', fontsize=14)  # 设置第三个子图标题
plt.xlabel('折数')  # 设置第三个子图横轴标签
plt.ylabel('验证损失 (MSE)')  # 设置第三个子图纵轴标签
plt.legend()  # 显示第三个子图图例
plt.grid(True, alpha=0.3, axis='y')  # 仅为第三个子图添加横向网格
plt.subplot(2, 2, 4)  # 选中第四个子图，绘制平均训练/验证损失曲线
max_iters = max(len(h['train_losses']) for h in history_list)  # 求所有折中最长的迭代轮数
avg_train_losses, avg_val_losses = [], []  # 初始化保存平均训练损失与平均验证损失的列表
for it in range(max_iters):  # 逐轮迭代计算各折损失的平均值
    tv = [h['train_losses'][it] for h in history_list if it < len(h['train_losses'])]  # 收集当前轮次各折的训练损失（跳过长度不足的折）
    vv = [h['val_losses'][it]   for h in history_list if it < len(h['val_losses'])]  # 收集当前轮次各折的验证损失（跳过长度不足的折）
    if tv: avg_train_losses.append(np.mean(tv))  # 若本轮有训练损失则记录其平均值
    if vv: avg_val_losses.append(np.mean(vv))  # 若本轮有验证损失则记录其平均值
plt.plot(avg_train_losses, label='平均训练损失', linewidth=2)  # 绘制平均训练损失曲线
plt.plot(avg_val_losses,   label='平均验证损失', linewidth=2)  # 绘制平均验证损失曲线
plt.title('5折平均损失曲线', fontsize=14)  # 设置第四个子图标题
plt.xlabel('迭代轮数 (Estimators)')  # 设置第四个子图横轴标签
plt.ylabel('损失值 (MSE)')  # 设置第四个子图纵轴标签
plt.legend()  # 显示第四个子图图例
plt.grid(True, alpha=0.3)  # 为第四个子图添加网格
plt.tight_layout()  # 自动调整子图间距
plt.savefig('kfold_training_XGBoost.png', dpi=600, bbox_inches='tight')  # 保存交叉验证损失曲线图片
print("\nK折交叉验证损失曲线已保存: kfold_training_XGBoost.png")  # 打印图片保存提示

# ===================== 训练最终模型 =====================
print(f"\n{'=' * 50}")  # 打印最终模型训练标题的分隔线
print("训练最终模型（使用全部训练数据）")  # 打印正在训练最终模型的提示
print(f"{'=' * 50}")  # 打印分隔线
print(f"训练集样本数: {len(X_train)}")  # 打印训练集样本数
print(f"测试集样本数: {len(X_test)}")  # 打印测试集样本数

train_val_split = int(len(X_train) * 0.8)  # 把训练集再按 8:2 切分出用于监控的验证集
X_train_split = X_train[:train_val_split]  # 取最终训练集特征
y_train_split = y_train[:train_val_split]  # 取最终训练集标签
X_val_split   = X_train[train_val_split:]  # 取最终验证集特征
y_val_split   = y_train[train_val_split:]  # 取最终验证集标签
print(f"最终训练集样本数: {len(X_train_split)}")  # 打印最终训练集样本数
print(f"最终验证集样本数: {len(X_val_split)}")  # 打印最终验证集样本数

final_model = xgb.XGBRegressor(**xgb_params)  # 用统一超参数创建最终回归模型
final_model.fit(  # 在最终训练集上训练，并用训练集与验证集评估
    X_train_split, y_train_split,  # 传入训练特征与标签
    eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],  # 指定评估集为训练集与验证集
    verbose=50  # 每 50 轮打印一次训练信息
)  # 训练调用结束
final_evals = final_model.evals_result()  # 取得最终模型的训练过程评估结果
train_losses = [v ** 2 for v in final_evals['validation_0']['rmse']]  # 把训练 RMSE 平方换算为 MSE
val_losses   = [v ** 2 for v in final_evals['validation_1']['rmse']]  # 把验证 RMSE 平方换算为 MSE
print(f"\n训练完成！最终验证损失: {min(val_losses):.6f}")  # 打印训练完成信息与最佳验证损失

# ===================== 绘制训练过程曲线 =====================
plt.figure(figsize=(12, 6))  # 创建 12x6 英寸画布绘制训练损失曲线
plt.plot(train_losses, label='训练损失', linewidth=2)  # 绘制训练损失曲线
plt.plot(val_losses,   label='验证损失', linewidth=2)  # 绘制验证损失曲线
plt.title('模型训练过程 - 损失曲线')  # 设置图表标题
plt.xlabel('迭代轮数 (Estimators)')  # 设置横轴标签
plt.ylabel('损失值 (MSE)')  # 设置纵轴标签
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
plt.tight_layout()  # 自动调整布局
plt.savefig('training_loss_XGBoost.png', dpi=600, bbox_inches='tight')  # 保存训练损失曲线图片
print("训练损失曲线已保存: training_loss_XGBoost.png")  # 打印训练损失曲线保存提示

# ===================== 测试集预测 =====================
print("\n开始测试集预测...")  # 打印开始测试集预测的提示
y_pred = final_model.predict(X_test).reshape(-1, 1)  # 用最终模型预测测试集并把结果 reshape 成二维列向量
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # 将测试集真实目标反归一化回原始量纲
y_pred_rescaled = target_scaler.inverse_transform(y_pred)  # 将预测结果反归一化回原始量纲

r2        = r2_score(y_test_rescaled, y_pred_rescaled)  # 计算决定系数 R2
mse_test  = mean_squared_error(y_test_rescaled, y_pred_rescaled)  # 计算均方误差 MSE
rmse_test = np.sqrt(mse_test)  # 由 MSE 开方得到 RMSE
mae_test  = mean_absolute_error(y_test_rescaled, y_pred_rescaled)  # 计算平均绝对误差 MAE

print(f"\n{'=' * 60}")  # 打印评估结果标题分隔线
print("测试集评估结果")  # 打印评估结果标题
print(f"{'=' * 60}")  # 打印评估结果标题分隔线
print(f"测试集 R2:   {r2:.4f}")  # 打印测试集 R2
print(f"测试集 RMSE: {rmse_test:.4f}")  # 打印测试集 RMSE
print(f"测试集 MAE:  {mae_test:.4f}")  # 打印测试集 MAE
print(f"{'=' * 60}")  # 打印分隔线

y_test_flat = y_test_rescaled.flatten()  # 把真实值展平为一维数组便于绘图
y_pred_flat = y_pred_rescaled.flatten()  # 把预测值展平为一维数组便于绘图
test_timestamps = test_data['timestamp'].values[time_step + pred_offset:]  # 截取与预测样本数对齐的测试集时间戳
test_timestamps = test_timestamps[:len(y_test_flat)]  # 按预测结果长度再截取一次时间戳保证一一对应
plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布绘制测试集预测对比
plt.plot(test_timestamps, y_test_flat, label='真实温度偏差', linewidth=2)  # 用蓝色实线绘制真实温度偏差
plt.plot(test_timestamps, y_pred_flat, label='XGBoost预测', linewidth=2)  # 用橙色实线绘制 XGBoost 预测值
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签
plt.title(f'测试集温度偏差结果 (R2: {r2:.4f}, RMSE: {rmse_test:.4f})')  # 设置含评估指标的图表标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
plt.savefig('test_prediction_XGBoost.png', dpi=600, bbox_inches='tight')  # 保存测试集预测对比图
print("测试集预测结果已保存: test_prediction_XGBoost.png")  # 打印测试集预测图保存提示

# ===================== 模型保存 =====================
model_checkpoint = {  # 构造要保存的模型检查点字典
    'model': final_model,  # 保存训练好的最终模型对象
    'input_dim': len(features),  # 记录模型输入特征维度
    'time_step': time_step,  # 记录输入时间窗口长度
    'pred_offset': pred_offset,  # 记录预测偏移步数
    'features': features,  # 记录使用的特征名列表
    'feature_scaler': feature_scaler,  # 保存特征归一化器，供推理阶段复用
    'target_scaler': target_scaler,  # 保存目标归一化器，供推理阶段复用
}
with open('Model_XGBoost.pkl', 'wb') as f:  # 以二进制写模式打开模型文件
    pickle.dump(model_checkpoint, f)  # 用 pickle 把检查点字典序列化写入文件
print("模型已保存: Model_XGBoost.pkl")  # 打印模型保存成功的提示

# ===================== 新数据预测 =====================
print("\n开始处理新数据...")  # 打印开始处理新数据的提示
new_data = None  # 初始化新数据变量为空
for enc in encodings:  # 依次尝试各种编码读取新数据
    try:  # 捕获解码异常以尝试下一种编码
        new_data = pd.read_csv('部分温湿度数据.csv', encoding=enc)  # 用当前编码读取新数据 CSV
        print(f"成功使用 {enc} 编码加载新数据")  # 打印成功使用的编码
        break  # 读取成功后停止尝试
    except (UnicodeDecodeError, UnicodeError):  # 捕获 Unicode 解码异常
        continue  # 当前编码失败则继续尝试
if new_data is None:  # 若所有编码失败则新数据仍为空
    raise ValueError("无法使用任何已知编码读取新数据文件")  # 抛出错误提示无法读取新数据

new_data['timestamp'] = pd.to_datetime(new_data.index * 5, unit='m',  # 按 5 分钟间隔从固定起点为新数据生成时间戳
                                        origin=pd.Timestamp('2025/6/21 0:04'))  # 时间戳生成的起始时刻参数
new_features_scaled = pd.DataFrame(  # 用训练集归一化器转换新数据特征并重建 DataFrame
    feature_scaler.transform(new_data[features]), columns=features)  # 指定归一化后 DataFrame 的列名
y_new_original_scaled = target_scaler.transform(new_data[[target]]).flatten()  # 转换新数据真实目标值并展平
X_new, y_new_scaled = create_flat_sequences(  # 调用展平函数构造新数据的特征与标签
    new_features_scaled, y_new_original_scaled, time_step, pred_offset)  # 传入归一化特征、真实目标与时间窗口参数

y_new_pred = final_model.predict(X_new).reshape(-1, 1)  # 用最终模型对新数据特征进行预测
y_new_pred_rescaled = target_scaler.inverse_transform(y_new_pred)  # 把新数据预测结果反归一化
y_new_rescaled = target_scaler.inverse_transform(y_new_scaled.reshape(-1, 1))  # 把新数据真实目标反归一化

r2_new   = r2_score(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据上的 R2
mse_new  = mean_squared_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据上的 MSE
rmse_new = np.sqrt(mse_new)  # 由 MSE 开方得到新数据的 RMSE
mae_new  = mean_absolute_error(y_new_rescaled, y_new_pred_rescaled)  # 计算新数据上的 MAE

print(f"\n{'=' * 60}")  # 打印新数据评估结果标题分隔线
print("新数据评估结果")  # 打印新数据评估结果标题
print(f"{'=' * 60}")  # 打印分隔线
print(f"新数据 R2:   {r2_new:.4f}")  # 打印新数据 R2
print(f"新数据 RMSE: {rmse_new:.4f}")  # 打印新数据 RMSE
print(f"新数据 MAE:  {mae_new:.4f}")  # 打印新数据 MAE
print(f"{'=' * 60}")  # 打印分隔线

timestamps = new_data['timestamp'][time_step + pred_offset:]  # 截取新数据时间戳，跳过时间窗口与预测偏移
plt.figure(figsize=(15, 5))  # 创建 15x5 英寸画布绘制新数据预测对比
y_new_flat = y_new_rescaled.flatten()  # 把新数据真实值展平为一维数组
y_new_pred_flat = y_new_pred_rescaled.flatten()  # 把新数据预测值展平为一维数组
plt.plot(timestamps[:len(y_new_flat)], y_new_flat, label='真实温度偏差', linewidth=2)  # 绘制新数据真实温度偏差曲线
plt.plot(timestamps[:len(y_new_pred_flat)], y_new_pred_flat, label='XGBoost预测', linewidth=2)  # 绘制新数据 XGBoost 预测曲线
plt.xlabel('时间')  # 设置横轴标签
plt.ylabel('温度偏差 (℃)')  # 设置纵轴标签
plt.title(f'新数据集温度偏差结果 (R2: {r2_new:.4f}, RMSE: {rmse_new:.4f})')  # 设置含评估指标的标题
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格
plt.savefig('new_data_prediction_XGBoost.png', dpi=600, bbox_inches='tight')  # 保存新数据预测对比图
print("新数据集预测结果已保存: new_data_prediction_XGBoost.png")  # 打印新数据预测图保存提示
print("\n========== 所有任务完成 ==========")  # 打印全部任务完成提示

