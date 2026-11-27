#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制数据文件中的温度、湿度、CO2 随时间变化的曲线。

功能：
1. 支持 CSV / XLS / XLSX
2. 支持自动识别时间列、温度列、湿度列、CO2 列
3. 支持通过 --start 和 --end 选择时间范围；默认使用全部数据
4. 支持手动指定列名，避免自动识别失败
5. 输出为 PNG 图片

示例：
python plot_env_data.py data.csv
python plot_env_data.py data.xlsx --start "2026-03-01 00:00:00" --end "2026-03-02 23:59:59"
python plot_env_data.py data.csv --time-col 时间 --temp-col 温度 --humidity-col 湿度 --co2-col CO2
补充说明：本脚本在本实验工程中用于把原始数据里的温度、湿度与 CO2 绘制成三栏时序对比图，
便于直观检查环境参数的变化趋势与数据质量；它既可作为命令行工具单独运行，也可作为绘图工具被其他脚本导入调用。
"""

from __future__ import annotations  # 导入 __future__ 注解特性，使类型注解延迟求值，兼容较旧版本 Python

import argparse  # 导入 argparse，用于解析命令行参数
from pathlib import Path  # 导入 Path，用于跨平台的路径拼接与判断
from typing import Iterable, Optional  # 导入 typing 中的类型注解辅助类型

import matplotlib.pyplot as plt  # 导入 pyplot，用于绘制三条时间序列曲线
import pandas as pd  # 导入 pandas，用于读取 CSV/Excel 并做数据清洗


TIME_KEYWORDS = [  # 定义时间列的候选关键词列表（中英文）
    "timestamp", "datetime", "date", "time",  # 英文候选关键词
    "日期", "时间", "采样时间", "记录时间", "检测时间", "创建时间",  # 中文候选关键词
]  # 时间关键词列表结束
TEMP_KEYWORDS = [  # 定义温度列的候选关键词列表
    "temperature", "temp", "当前温度",  # 温度候选关键词内容
]  # 温度关键词列表结束
HUMIDITY_KEYWORDS = [  # 定义湿度列的候选关键词列表
    "humidity", "humid", "rh", "湿度", "相对湿度",  # 湿度候选关键词内容
]  # 湿度关键词列表结束
CO2_KEYWORDS = [  # 定义 CO2 列的候选关键词列表
    "CO2", "二氧化碳", "carbon dioxide",  # CO2 候选关键词内容
]  # CO2 关键词列表结束


def normalize_name(name: str) -> str:  # 定义函数：规范化列名或关键词，便于统一比较
    """把列名或关键词规范化：转小写、去首尾空格、去掉下划线与内部空白。

    参数：
        name: 待规范化的名称。
    返回：规范化后的字符串。
    """
    return "".join(str(name).strip().lower().replace("_", " ").split())  # 把名称统一为小写、去掉空格与下划线，便于关键词匹配


def find_column(columns: Iterable[str], keywords: list[str]) -> Optional[str]:  # 定义函数：按关键词优先级在列名中查找匹配列
    """在给定列名中按关键词优先级查找匹配的列。

    参数：
        columns: 候选列名集合。
        keywords: 按优先级排列的关键词列表。
    返回：匹配到的原始列名；均不匹配时返回 None。
    """
    normalized = {col: normalize_name(col) for col in columns}  # 建立原始列名到规范化名称的字典，供后续匹配使用

    # 先精确包含关键词匹配
    for keyword in keywords:  # 按关键词优先级依次尝试匹配
        key = normalize_name(keyword)  # 把当前关键词规范化成统一格式
        for col, norm in normalized.items():  # 遍历数据中的每一列
            if key in norm:  # 若规范化关键词出现在列名中即视为匹配成功
                return col  # 返回匹配到的原始列名

    return None  # 所有关键词都没匹配上时返回 None


def read_data(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:  # 定义函数：根据后缀读取 CSV 或 Excel 文件
    """根据文件后缀读取 CSV 或 Excel 数据。

    参数：
        file_path: 数据文件路径。
        sheet_name: Excel 工作表名称，可选。
    返回：读取到的 DataFrame。
    """
    suffix = file_path.suffix.lower()  # 取得文件后缀并转为小写，便于判断文件类型
    if suffix == ".csv":  # 后缀为 .csv 时按 CSV 方式读取
        for encoding in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:  # 依次尝试多种常见编码读取 CSV
            try:  # 进入 try 块捕获解码错误
                return pd.read_csv(file_path, encoding=encoding)  # 用当前编码读取成功则直接返回 DataFrame
            except UnicodeDecodeError:  # 捕获 UnicodeDecodeError 异常
                continue  # 当前编码失败则尝试下一种
        return pd.read_csv(file_path)  # 所有编码均失败时按 pandas 默认编码再读一次

    if suffix in {".xlsx", ".xls"}:  # 后缀为 xlsx/xls 时按 Excel 方式读取
        return pd.read_excel(file_path, sheet_name=sheet_name)  # 用 pandas 读取指定工作表，未指定则读第一个

    raise ValueError(f"不支持的文件格式: {suffix}，仅支持 CSV / XLS / XLSX")  # 其他后缀不支持，抛出 ValueError 并给出提示


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:  # 定义函数：把时间列转为 datetime 并排序清洗
    """把时间列转为 datetime，丢弃无效时间并按时间升序排序。

    参数：
        df: 原始数据。
        time_col: 时间列列名。
    返回：处理后的 DataFrame 副本。
    """
    df = df.copy()  # 复制数据，避免修改调用方传入的 DataFrame
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")  # 把时间列转成 datetime，无法解析的值置为 NaT
    df = df.dropna(subset=[time_col]).sort_values(time_col)  # 丢弃时间无效的行并按时间升序排序
    return df  # 返回处理好的 DataFrame


def filter_time_range(  # 定义函数：按起止时间筛选数据（参数见下方）
    df: pd.DataFrame,  # 输入 DataFrame
    time_col: str,  # 时间列列名
    start: Optional[str],  # 起始时间字符串，None 表示不限
    end: Optional[str],  # 结束时间字符串，None 表示不限
) -> pd.DataFrame:  # 函数签名结束，返回筛选后的 DataFrame
    """按起止时间筛选数据行。

    参数：
        df: 已含 datetime 时间列的数据。
        time_col: 时间列列名。
        start / end: 起始与结束时间字符串，None 表示不限。
    返回：筛选后的 DataFrame。
    """
    result = df.copy()  # 复制输入数据，避免影响原始数据

    if start:  # 若指定了起始时间则进行下界筛选
        start_ts = pd.to_datetime(start, errors="coerce")  # 把起始时间字符串解析为 Timestamp
        if pd.isna(start_ts):  # 若解析结果为空则说明格式非法
            raise ValueError(f"无法解析起始时间: {start}")  # 抛出 ValueError 提示起始时间无法解析
        result = result[result[time_col] >= start_ts]  # 仅保留时间不早于起始时间的记录

    if end:  # 若指定了结束时间则进行上界筛选
        end_ts = pd.to_datetime(end, errors="coerce")  # 把结束时间字符串解析为 Timestamp
        if pd.isna(end_ts):  # 若解析结果为空则说明格式非法
            raise ValueError(f"无法解析结束时间: {end}")  # 抛出 ValueError 提示结束时间无法解析
        result = result[result[time_col] <= end_ts]  # 仅保留时间不晚于结束时间的记录

    return result  # 返回按时间范围筛选后的数据


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:  # 定义函数：把指定列强制转换为数值类型
    """把指定列强制转换为数值类型，无法转换的值置为 NaN。

    参数：
        df: 输入数据。
        cols: 需要转换的列名列表。
    返回：转换后的 DataFrame 副本。
    """
    df = df.copy()  # 复制数据，避免修改原始 DataFrame
    for col in cols:  # 遍历需要转换数值类型的列
        df[col] = pd.to_numeric(df[col], errors="coerce")  # 把列转为数值，无法转换的值置为 NaN
    return df  # 返回数值化后的数据


def plot_data(  # 定义函数：绘制三栏时序图并保存（参数见下方）
    df: pd.DataFrame,  # 完成清洗与数值化的数据
    time_col: str,  # 时间列列名
    temp_col: str,  # 温度列列名
    humidity_col: str,  # 湿度列列名
    co2_col: str,  # CO2 列列名
    output: Path,  # 图片输出路径
    title: str,  # 图表标题
) -> None:  # 函数签名结束，无返回值
    """绘制温度、湿度、CO2 三栏时序图并保存为 PNG。

    参数：
        df: 完成清洗与数值化的数据。
        time_col / temp_col / humidity_col / co2_col: 各类别的列名。
        output: 图片输出路径。
        title: 图表标题。
    返回：无返回值。
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]  # 设置中文字体候选列表，保证图中中文正常显示
    plt.rcParams["axes.unicode_minus"] = False  # 关闭 Unicode 负号显示，避免负号乱码

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)  # 创建 3 行 1 列、共享横轴的子图画布，尺寸 14x10 英寸

    axes[0].plot(df[time_col], df[temp_col], linewidth=1.2)  # 在第一个子图绘制温度随时间的变化曲线
    axes[0].set_ylabel("温度")  # 设置第一个子图的纵轴标签为温度
    axes[0].set_title(title)  # 设置图表标题
    axes[0].grid(True, alpha=0.3)  # 为第一个子图添加半透明网格线

    axes[1].plot(df[time_col], df[humidity_col], linewidth=1.2)  # 在第二个子图绘制湿度随时间的变化曲线
    axes[1].set_ylabel("湿度")  # 设置第二个子图的纵轴标签为湿度
    axes[1].grid(True, alpha=0.3)  # 为第二个子图添加半透明网格线

    axes[2].plot(df[time_col], df[co2_col], linewidth=1.2)  # 在第三个子图绘制 CO2 随时间的变化曲线
    axes[2].set_ylabel("CO2")  # 设置第三个子图的纵轴标签为 CO2
    axes[2].set_xlabel("时间")  # 设置第三个子图的横轴标签为时间
    axes[2].grid(True, alpha=0.3)  # 为第三个子图添加半透明网格线

    plt.xticks(rotation=20)  # 把横轴刻度标签旋转 20 度，避免时间标签重叠
    plt.tight_layout()  # 自动调整子图间距与布局
    output.parent.mkdir(parents=True, exist_ok=True)  # 确保图片输出目录存在，不存在则递归创建
    plt.savefig(output, dpi=600, bbox_inches="tight")  # 以 600 dpi 保存图片并裁剪掉多余空白
    plt.close(fig)  # 关闭画布，释放绘图内存


def build_parser() -> argparse.ArgumentParser:  # 定义函数：构建命令行参数解析器
    """构建命令行参数解析器，配置输入、输出、时间范围与列名等选项。

    返回：配置好的 argparse.ArgumentParser 对象。
    """
    parser = argparse.ArgumentParser(description="绘制温度、湿度、CO2 随时间变化图")  # 创建参数解析器并写明用途描述
    # 默认使用脚本同目录下的 温湿度数据.csv
    _default_input = str(Path(__file__).parent / "温湿度数据.csv")  # 计算默认输入路径：脚本同目录下的 温湿度数据.csv
    parser.add_argument("input", nargs="?", default=_default_input, help="输入数据文件路径，默认使用脚本同目录的 温湿度数据.csv")  # 添加位置参数 input，默认使用上面的路径
    _default_output = str(Path(__file__).parent / "env_plot.png")  # 计算默认输出路径：脚本同目录下的 env_plot.png
    parser.add_argument("--output", default=_default_output, help="输出图片路径，默认脚本同目录的 env_plot.png")  # 添加 --output 选项，指定输出图片路径
    parser.add_argument("--sheet", default=None, help="Excel 的工作表名称，默认读取第一个 sheet")  # 添加 --sheet 选项，指定 Excel 工作表名称

    parser.add_argument("--start", default=None, help="起始时间，例如 2026-03-01 00:00:00")  # 添加 --start 选项，指定筛选起始时间
    parser.add_argument("--end", default=None, help="结束时间，例如 2026-03-02 23:59:59")  # 添加 --end 选项，指定筛选结束时间

    parser.add_argument("--time-col", default=None, help="时间列名")  # 添加 --time-col 选项，手动指定时间列名
    parser.add_argument("--temp-col", default=None, help="温度列名")  # 添加 --temp-col 选项，手动指定温度列名
    parser.add_argument("--humidity-col", default=None, help="湿度列名")  # 添加 --humidity-col 选项，手动指定湿度列名
    parser.add_argument("--co2-col", default=None, help="CO2 列名")  # 添加 --co2-col 选项，手动指定 CO2 列名

    parser.add_argument("--title", default="温度 / 湿度 / CO2 变化曲线", help="图表标题")  # 添加 --title 选项，指定图表标题
    return parser  # 返回配置完成的解析器对象


def resolve_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, str, str]:  # 定义函数：确定时间/温度/湿度/CO2 四类列名
    """确定时间/温度/湿度/CO2 四列的实际列名。

    优先使用命令行显式指定的列名，否则按关键词自动识别；识别失败则抛异常。
    返回：(时间列, 温度列, 湿度列, CO2 列) 四个字符串组成的元组。
    """
    columns = [str(c) for c in df.columns]  # 把 DataFrame 的列名统一转换为字符串列表

    time_col = args.time_col or find_column(columns, TIME_KEYWORDS)  # 时间列：优先采用命令行参数，否则按关键词自动识别
    temp_col = args.temp_col or find_column(columns, TEMP_KEYWORDS)  # 温度列：优先采用命令行参数，否则按关键词自动识别
    humidity_col = args.humidity_col or find_column(columns, HUMIDITY_KEYWORDS)  # 湿度列：优先采用命令行参数，否则按关键词自动识别
    co2_col = args.co2_col or find_column(columns, CO2_KEYWORDS)  # CO2 列：优先采用命令行参数，否则按关键词自动识别

    missing = []  # 初始化记录未识别列的列表
    if not time_col:  # 时间列没有识别到
        missing.append("时间列(--time-col)")  # 把时间列提示加入缺失列表
    if not temp_col:  # 温度列没有识别到
        missing.append("温度列(--temp-col)")  # 把温度列提示加入缺失列表
    if not humidity_col:  # 湿度列没有识别到
        missing.append("湿度列(--humidity-col)")  # 把湿度列提示加入缺失列表
    if not co2_col:  # CO2 列没有识别到
        missing.append("CO2列(--co2-col)")  # 把 CO2 列提示加入缺失列表

    if missing:  # 只要存在未识别的列就进入报错分支
        raise ValueError(  # 开始构造错误信息
            "自动识别列失败，请手动指定：" + "、".join(missing) + f"\n当前列名：{columns}"  # 错误信息列出缺失列并附上当前所有列名，便于手动指定
        )  # 抛出 ValueError 终止流程

    return time_col, temp_col, humidity_col, co2_col  # 返回识别出的四个列名组成的元组


def main() -> None:  # 定义程序入口函数：读取参数、清洗并绘图
    """程序入口：读取参数、清洗数据、绘图并打印运行摘要。

    依次完成文件存在性检查、读取、列名解析、时间转换与筛选、数值化、
    绘图与结果打印；任一环节数据为空都会抛出明确异常。
    """
    parser = build_parser()  # 创建命令行参数解析器
    args = parser.parse_args()  # 解析命令行输入的参数

    input_path = Path(args.input)  # 把输入参数转换为 Path 对象
    output_path = Path(args.output)  # 把输出参数转换为 Path 对象

    if not input_path.exists():  # 检查输入文件是否真实存在
        raise FileNotFoundError(f"输入文件不存在: {input_path}")  # 文件不存在则抛出 FileNotFoundError

    df = read_data(input_path, sheet_name=args.sheet)  # 根据文件后缀读取数据
    if df.empty:  # 若读取到的是空表则无法绘图
        raise ValueError("输入文件为空，无法绘图")  # 抛出 ValueError 提示输入文件为空

    time_col, temp_col, humidity_col, co2_col = resolve_columns(df, args)  # 解析出时间/温度/湿度/CO2 四类列名
    df = ensure_datetime(df, time_col)  # 把时间列转为 datetime 并按时间排序
    df = filter_time_range(df, time_col, args.start, args.end)  # 按起止时间筛选数据范围

    if df.empty:  # 若时间筛选后没有数据则提示参数可能有误
        raise ValueError("按当前时间范围筛选后没有数据，请检查 --start / --end 参数")  # 抛出 ValueError 提示检查 --start / --end

    df = coerce_numeric(df, [temp_col, humidity_col, co2_col])  # 把温度、湿度、CO2 三列转换为数值类型
    df = df.dropna(subset=[temp_col, humidity_col, co2_col])  # 删除三列中任一含缺失值的行

    if df.empty:  # 若数值化后没有有效数据则进入报错分支
        raise ValueError("温度、湿度或 CO2 列在转换为数值后没有有效数据")  # 抛出 ValueError 提示没有有效数值

    plot_data(  # 调用绘图函数生成图片，参数见下方
        df=df,  # 传入清洗后的数据
        time_col=time_col,  # 传入时间列名
        temp_col=temp_col,  # 传入温度列名
        humidity_col=humidity_col,  # 传入湿度列名
        co2_col=co2_col,  # 传入 CO2 列名
        output=output_path,  # 传入输出图片路径
        title=args.title,  # 传入图表标题
    )  # 绘图函数调用结束

    print("绘图完成")  # 打印绘图完成提示
    print(f"输入文件: {input_path}")  # 打印本次使用的输入文件路径
    print(f"输出图片: {output_path}")  # 打印图片输出路径
    print(f"使用列: 时间={time_col}, 温度={temp_col}, 湿度={humidity_col}, CO2={co2_col}")  # 打印实际使用的四类列名
    print(f"时间范围: start={args.start or '全部'}, end={args.end or '全部'}")  # 打印本次采用的时间范围


if __name__ == "__main__":  # 仅当作为脚本直接运行时才调用入口函数
    main()  # 作为脚本运行时调用入口函数，启动程序
