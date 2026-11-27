"""
鸡舍环境实时监测与预警系统 - Python实现
包含环境监测模块和双指标预警策略

补充说明：本模块以数据类/普通类的方式组织配置与逻辑，主要包括：
1. EnvironmentThresholds：按肉鸡日龄给出温度、湿度、风速的适宜范围以及 CO2 上限；
2. WarningThresholds：定义 TD 偏热/偏冷阈值与 THI 四级风险分级并给出判定方法；
3. EnvironmentMonitor：逐条检查环境参数是否达标，并分析鸡舍前后/左右的温差分布；
4. WarningSystem：基于预测的 TD 与 THI 生成预警类型与调控建议，并输出统计摘要与明细；
5. demo_monitoring / demo_warning：使用模拟数据演示上述两个模块的使用方式。
"""

import pandas as pd  # 导入 pandas（本模块中未直接使用，保留原始导入以备扩展）
import numpy as np  # 导入 numpy（本模块中未直接使用，保留原始导入以备扩展）
from datetime import datetime, timedelta  # 导入 datetime 与 timedelta，用于生成时间戳和预测时刻
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于定义环境阈值配置类
from typing import Dict, List, Tuple  # 导入类型注解工具，用于标注字典、列表与元组类型

# ===================== 环境参数配置 =====================

@dataclass  # 使用 dataclass 装饰器定义环境阈值配置类
class EnvironmentThresholds:
    """环境参数适宜范围配置。

    以类属性的形式保存不同日龄段的温度、湿度、风速适宜范围以及 CO2 浓度上限，
    并通过 get_range_for_age 静态方法按日龄与参数类型查询对应区间。
    """
    
    # 温度适宜范围 (℃)
    TEMP_RANGES = {  # 定义 (日龄下限, 日龄上限) -> (适宜温度下限, 适宜温度上限) 的映射
        (1, 3): (34, 35),  # 1~3 日龄适宜温度 34~35℃
        (4, 7): (32, 33),  # 4~7 日龄适宜温度 32~33℃
        (8, 15): (30, 31),  # 8~15 日龄适宜温度 30~31℃
        (16, 23): (28, 29),  # 16~23 日龄适宜温度 28~29℃
        (24, 31): (26, 27),  # 24~31 日龄适宜温度 26~27℃
        (32, 39): (24, 25),  # 32~39 日龄适宜温度 24~25℃
        (40, 999): (22, 23),  # 40 日龄以上适宜温度 22~23℃
    }
    
    # 相对湿度适宜范围 (%)
    HUMIDITY_RANGES = {  # 定义各日龄段的相对湿度适宜范围
        (1, 10): (65, 70),  # 1~10 日龄湿度 65~70%
        (11, 30): (60, 65),  # 11~30 日龄湿度 60~65%
        (31, 45): (55, 60),  # 31~45 日龄湿度 55~60%
        (46, 999): (50, 55),  # 46 日龄以上湿度 50~55%
    }
    
    # 风速适宜范围 (m/s)
    WIND_SPEED_RANGES = {  # 定义各日龄段的适宜风速范围
        (1, 14): (0, 0.20),  # 1~14 日龄风速 0~0.20 m/s
        (15, 21): (0, 0.51),  # 15~21 日龄风速 0~0.51 m/s
        (22, 28): (0, 1.02),  # 22~28 日龄风速 0~1.02 m/s
        (29, 999): (1.5, 2.5),  # 29 日龄以上风速 1.5~2.5 m/s
    }
    
    # CO2浓度上限 (ppm)
    CO2_LIMIT = 1500  # 定义 CO2 浓度上限为 1500 ppm
    @staticmethod  # 声明为静态方法，无需实例化即可调用
    def get_range_for_age(age: int, param_type: str) -> Tuple[float, float]:
        """根据日龄获取参数适宜范围。

        参数:
            age: 肉鸡日龄（整数）。
            param_type: 参数类型字符串，可选 temperature / humidity / wind_speed。
        返回:
            (下限, 上限) 二元组；若参数类型未知则抛出 ValueError，若日龄未匹配则返回 (None, None)。
        """
        if param_type == 'temperature':  # 若查询类型为温度
            ranges = EnvironmentThresholds.TEMP_RANGES  # 选用温度范围表
        elif param_type == 'humidity':  # 若查询类型为湿度
            ranges = EnvironmentThresholds.HUMIDITY_RANGES  # 选用湿度范围表
        elif param_type == 'wind_speed':  # 若查询类型为风速
            ranges = EnvironmentThresholds.WIND_SPEED_RANGES  # 选用风速范围表
        else:  # 其他未知类型
            raise ValueError(f"未知参数类型: {param_type}")  # 抛出异常提示参数类型不合法
        
        for (min_age, max_age), (lower, upper) in ranges.items():  # 遍历该参数的所有日龄区间
            if min_age <= age <= max_age:  # 若当前日龄落在区间内
                return lower, upper  # 返回该区间对应的适宜范围
        
        return None, None  # 没有任何区间匹配时返回空范围

# ===================== 预警阈值配置 =====================

class WarningThresholds:
    """预警阈值配置。

    定义 TD 偏热/偏冷的温度偏差阈值，以及 THI 的舒适/警戒/预警/危急四级分界，
    并提供获取 THI 风险等级与 TD 状态的静态方法。
    """
    
    # TD预警阈值 (℃)
    TD_HOT_THRESHOLD = 2.5  # TD 大于该值判定为偏热
    TD_COLD_THRESHOLD = -2.5  # TD 小于该值判定为偏冷
    
    # THI风险分级
    THI_COMFORT_MAX = 74  # THI 不大于 74 为舒适
    THI_ALERT_MAX = 78  # THI 不大于 78 为警戒
    THI_WARNING_MAX = 82  # THI 不大于 82 为预警，超过则为危急
    
    @staticmethod  # 声明为静态方法
    def get_thi_level(thi_value: float) -> Tuple[str, int]:
        """获取THI风险等级。

        参数:
            thi_value: 温湿度指数 THI 数值。
        返回:
            (等级名称, 等级数值) 二元组：舒适/0、警戒/1、预警/2、危急/3。
        """
        if thi_value <= WarningThresholds.THI_COMFORT_MAX:  # THI 不高于舒适上限
            return "舒适", 0  # 返回舒适等级
        elif thi_value <= WarningThresholds.THI_ALERT_MAX:  # THI 不高于警戒上限
            return "警戒", 1  # 返回警戒等级
        elif thi_value <= WarningThresholds.THI_WARNING_MAX:  # THI 不高于预警上限
            return "预警", 2  # 返回预警等级
        else:  # THI 超过预警上限
            return "危急", 3  # 返回危急等级
    
    @staticmethod  # 声明为静态方法
    def get_td_status(td_value: float) -> Tuple[str, int]:
        """获取TD状态。

        参数:
            td_value: 温度偏差 TD（℃）。
        返回:
            (状态名称, 状态数值) 二元组：偏热/1、正常/0、偏冷/-1。
        """
        if td_value > WarningThresholds.TD_HOT_THRESHOLD:  # TD 超过偏热阈值
            return "偏热", 1  # 返回偏热状态
        elif td_value < WarningThresholds.TD_COLD_THRESHOLD:  # TD 低于偏冷阈值
            return "偏冷", -1  # 返回偏冷状态
        else:  # TD 处于正常区间
            return "正常", 0  # 返回正常状态

# ===================== 环境监测模块 =====================

class EnvironmentMonitor:
    """鸡舍环境实时监测模块。

    按日龄检查温度、湿度、风速与 CO2 是否处于适宜范围，保存监测历史，
    并支持分析鸡舍前后与左右的温差分布及给出改善建议。
    """
    
    def __init__(self):
        """初始化监测器：创建空的监测历史列表。"""
        self.monitoring_history = []  # 初始化列表，用于逐条保存监测记录
    
    def check_environment_status(self, 
                                 age: int,
                                 temperature: float,
                                 humidity: float,
                                 wind_speed: float,
                                 co2: float,
                                 timestamp: datetime = None) -> Dict:
        """
        检查环境参数状态
        
        参数：
            age: 肉鸡日龄
            temperature: 当前温度 (℃)
            humidity: 当前相对湿度 (%)
            wind_speed: 当前风速 (m/s)
            co2: 当前CO2浓度 (ppm)
            timestamp: 时间戳
        
        返回：
            环境状态字典

        说明：如果 timestamp 为 None，则自动使用当前时间；检查结果会追加到 monitoring_history。
        """
        if timestamp is None:  # 若调用方未传入时间戳
            timestamp = datetime.now()  # 使用当前时间作为记录时间
        
        # 获取适宜范围
        temp_lower, temp_upper = EnvironmentThresholds.get_range_for_age(age, 'temperature')  # 查询该日龄的适宜温度区间
        humid_lower, humid_upper = EnvironmentThresholds.get_range_for_age(age, 'humidity')  # 查询该日龄的适宜湿度区间
        wind_lower, wind_upper = EnvironmentThresholds.get_range_for_age(age, 'wind_speed')  # 查询该日龄的适宜风速区间
        
        # 判别各参数状态
        temp_status = self._check_parameter(temperature, temp_lower, temp_upper)  # 判断温度是否达标
        humid_status = self._check_parameter(humidity, humid_lower, humid_upper)  # 判断湿度是否达标
        wind_status = self._check_parameter(wind_speed, wind_lower, wind_upper)  # 判断风速是否达标
        co2_status = "正常" if co2 <= EnvironmentThresholds.CO2_LIMIT else "异常"  # 判断 CO2 是否超过上限
        
        # 综合判断
        overall_status = "正常" if all([  # 当四项参数全部正常时综合状态才为正常
            temp_status == "正常",  # 温度正常
            humid_status == "正常",  # 湿度正常
            wind_status == "正常",  # 风速正常
            co2_status == "正常"  # CO2 正常
        ]) else "异常"  # 任一项异常则综合状态为异常
        
        monitoring_record = {  # 组装本次监测记录字典
            'timestamp': timestamp,  # 记录时间
            'age': age,  # 肉鸡日龄
            'temperature': temperature,  # 当前温度
            'temp_range': (temp_lower, temp_upper),  # 适宜温度范围
            'temp_status': temp_status,  # 温度状态
            'humidity': humidity,  # 当前湿度
            'humid_range': (humid_lower, humid_upper),  # 适宜湿度范围
            'humid_status': humid_status,  # 湿度状态
            'wind_speed': wind_speed,  # 当前风速
            'wind_range': (wind_lower, wind_upper),  # 适宜风速范围
            'wind_status': wind_status,  # 风速状态
            'co2': co2,  # 当前 CO2 浓度
            'co2_limit': EnvironmentThresholds.CO2_LIMIT,  # CO2 上限
            'co2_status': co2_status,  # CO2 状态
            'overall_status': overall_status,  # 综合状态
        }
        
        self.monitoring_history.append(monitoring_record)  # 将本次记录追加到监测历史
        return monitoring_record  # 返回本次监测记录
    
    @staticmethod  # 声明为静态方法
    def _check_parameter(value: float, lower: float, upper: float) -> str:
        """检查单个参数状态。

        参数:
            value: 参数的实际数值；lower/upper: 适宜范围的下限与上限。
        返回:
            数值落在闭区间 [lower, upper] 内返回 "正常"，否则返回 "异常"。
        """
        if lower <= value <= upper:  # 若数值落在适宜范围内
            return "正常"  # 判定为正常
        else:  # 若数值超出范围
            return "异常"  # 判定为异常
    
    def get_environment_distribution(self, 
                                     front_temp: float,
                                     rear_temp: float,
                                     left_temp: float,
                                     right_temp: float) -> Dict:
        """
        分析鸡舍环境分布差异
        
        参数：
            front_temp: 前方温度
            rear_temp: 后方温度
            left_temp: 左方温度
            right_temp: 右方温度
        
        返回：
            环境分布分析结果

        说明：以 2.0℃ 为均匀性阈值，判断前后/左右温差是否均匀，并给出改善建议。
        """
        front_rear_diff = abs(front_temp - rear_temp)  # 计算前后温差绝对值
        left_right_diff = abs(left_temp - right_temp)  # 计算左右温差绝对值
        
        # 判断分布是否均匀
        uniform_threshold = 2.0  # ℃  # 定义温差均匀性阈值为 2.0℃
        
        return {  # 返回环境分布分析结果字典
            'front_rear_diff': front_rear_diff,  # 前后温差
            'left_right_diff': left_right_diff,  # 左右温差
            'front_rear_uniform': front_rear_diff <= uniform_threshold,  # 前后分布是否均匀
            'left_right_uniform': left_right_diff <= uniform_threshold,  # 左右分布是否均匀
            'overall_uniform': (front_rear_diff <= uniform_threshold and  # 整体均匀需前后与左右同时满足
                               left_right_diff <= uniform_threshold),  # 左右温差也需不超过阈值
            'recommendation': self._get_distribution_recommendation(  # 获取改善建议
                front_rear_diff, left_right_diff  # 传入前后与左右温差
            )
        }
    
    @staticmethod  # 声明为静态方法
    def _get_distribution_recommendation(front_rear_diff: float, 
                                        left_right_diff: float) -> str:
        """获取环境分布改善建议。

        参数:
            front_rear_diff: 前后温差；left_right_diff: 左右温差。
        返回:
            针对温差情况给出的中文调控建议字符串。
        """
        if front_rear_diff > 2.0 and left_right_diff > 2.0:  # 前后与左右温差都超标
            return "环境分布不均，建议调整通风口位置和风速"  # 建议全面调整
        elif front_rear_diff > 2.0:  # 仅前后温差超标
            return "前后温差过大，建议调整前后通风口"  # 建议调整前后通风口
        elif left_right_diff > 2.0:  # 仅左右温差超标
            return "左右温差过大，建议调整左右通风口"  # 建议调整左右通风口
        else:  # 温差均在合理范围
            return "环境分布均匀，无需调整"  # 无需调整

# ===================== 预警系统 =====================

class WarningSystem:
    """实时预警系统。

    根据预测的 TD 与 THI 判定是否触发预警，生成预警类型（偏热/偏冷及 THI 各级）
    与调控建议，保存预警记录，并可打印统计摘要与若干条预警详情。
    """
    
    def __init__(self, time_step: int = 36, pred_offset: int = 6):
        """
        初始化预警系统
        
        参数：
            time_step: 输入时间步长（默认36步 = 180分钟）
            pred_offset: 预测偏移（默认6步 = 30分钟）

        说明：采样间隔固定为 5 分钟，pred_horizon 表示提前预警的分钟数。
        """
        self.time_step = time_step  # 保存输入时间步数
        self.pred_offset = pred_offset  # 保存预测偏移步数
        self.time_interval = 5  # 分钟  # 采样间隔为 5 分钟
        self.pred_horizon = pred_offset * time_interval  # 计算提前预警的分钟数（默认 30 分钟）
        self.warnings = []  # 初始化预警记录列表
    
    def generate_warning(self,
                        td_predicted: float,
                        thi_predicted: float,
                        current_time: datetime,
                        future_time: datetime) -> Dict:
        """
        生成预警信息
        
        参数：
            td_predicted: 预测的温度偏差 (℃)
            thi_predicted: 预测的温湿度指数
            current_time: 当前时间
            future_time: 预测时间
        
        返回：
            预警信息字典

        说明：TD 异常或 THI 达到警戒及以上即触发预警，触发的记录会保存到 self.warnings。
        """
        # 获取TD和THI状态
        td_status, td_level = WarningThresholds.get_td_status(td_predicted)  # 判定预测 TD 的状态与等级
        thi_status, thi_level = WarningThresholds.get_thi_level(thi_predicted)  # 判定预测 THI 的状态与等级
        
        # 判断是否触发预警
        warning_triggered = (td_level != 0) or (thi_level >= 1)  # TD 异常或 THI 警戒及以上即触发
        
        # 生成预警类型
        warning_types = []  # 初始化预警类型列表
        if td_level > 0:  # TD 偏热
            warning_types.append("偏热预警")  # 加入偏热预警类型
        elif td_level < 0:  # TD 偏冷
            warning_types.append("偏冷预警")  # 加入偏冷预警类型
        
        if thi_level == 1:  # THI 警戒
            warning_types.append("THI警戒")  # 加入 THI 警戒类型
        elif thi_level == 2:  # THI 预警
            warning_types.append("THI预警")  # 加入 THI 预警类型
        elif thi_level == 3:  # THI 危急
            warning_types.append("THI危急")  # 加入 THI 危急类型
        
        warning_type = " | ".join(warning_types) if warning_types else "正常"  # 拼接类型字符串，无预警则记为正常
        
        # 生成调控建议
        recommendation = self._get_recommendation(td_level, thi_level)  # 依据等级获取调控建议
        
        warning_info = {  # 组装预警信息字典
            'current_time': current_time,  # 当前时间
            'future_time': future_time,  # 预测对应的未来时间
            'td_predicted': td_predicted,  # 预测 TD
            'thi_predicted': thi_predicted,  # 预测 THI
            'td_status': td_status,  # TD 状态名称
            'thi_status': thi_status,  # THI 状态名称
            'warning_triggered': warning_triggered,  # 是否触发预警
            'warning_type': warning_type,  # 预警类型字符串
            'recommendation': recommendation,  # 调控建议
        }
        
        if warning_triggered:  # 若确实触发预警
            self.warnings.append(warning_info)  # 将预警信息追加到历史列表
        
        return warning_info  # 返回本次预警信息
    
    @staticmethod  # 声明为静态方法
    def _get_recommendation(td_level: int, thi_level: int) -> str:
        """获取环境调控建议。

        参数:
            td_level: TD 等级（1 偏热、0 正常、-1 偏冷）；thi_level: THI 等级（0~3）。
        返回:
            由各项建议以 " | " 拼接而成的字符串；无建议时返回“维持当前环境”。
        """
        recommendations = []  # 初始化建议列表
        
        if td_level > 0:  # TD 偏热
            recommendations.append("启动降温措施（通风、喷雾）")  # 建议降温
        elif td_level < 0:  # TD 偏冷
            recommendations.append("启动升温措施（供暖）")  # 建议升温
        
        if thi_level == 1:  # THI 警戒
            recommendations.append("加强通风，监测热应激")  # 建议加强通风
        elif thi_level == 2:  # THI 预警
            recommendations.append("立即启动降温系统")  # 建议立即降温
        elif thi_level == 3:  # THI 危急
            recommendations.append("紧急启动所有降温措施")  # 建议紧急全开降温
        
        return " | ".join(recommendations) if recommendations else "维持当前环境"  # 拼接建议，空则维持环境
    
    def print_warning_summary(self) -> None:
        """打印预警统计摘要。

        统计并打印总预警次数，以及 TD 偏热/偏冷与 THI 警戒/预警/危急各自的出现次数；
        若没有预警则打印提示并返回。
        """
        if not self.warnings:  # 若当前没有任何预警记录
            print("暂无预警信息")  # 打印提示信息
            return  # 提前返回
        
        print("\n" + "="*70)  # 打印空行与分隔线
        print("预警统计摘要")  # 打印摘要标题
        print("="*70)  # 打印分隔线
        
        total_warnings = len(self.warnings)  # 统计预警总次数
        
        # TD预警统计
        td_hot = sum(1 for w in self.warnings if "偏热" in w['warning_type'])  # 统计偏热预警次数
        td_cold = sum(1 for w in self.warnings if "偏冷" in w['warning_type'])  # 统计偏冷预警次数
        
        # THI预警统计
        thi_alert = sum(1 for w in self.warnings if "警戒" in w['warning_type'])  # 统计 THI 警戒次数
        thi_warning = sum(1 for w in self.warnings if "预警" in w['warning_type'])  # 统计 THI 预警次数
        thi_critical = sum(1 for w in self.warnings if "危急" in w['warning_type'])  # 统计 THI 危急次数
        
        print(f"总预警次数: {total_warnings}")  # 打印总预警次数
        print(f"\nTD预警统计:")  # 打印 TD 预警统计标题
        print(f"  - 偏热预警: {td_hot} 次")  # 打印偏热次数
        print(f"  - 偏冷预警: {td_cold} 次")  # 打印偏冷次数
        print(f"\nTHI预警统计:")  # 打印 THI 预警统计标题
        print(f"  - 警戒级: {thi_alert} 次")  # 打印警戒次数
        print(f"  - 预警级: {thi_warning} 次")  # 打印预警次数
        print(f"  - 危急级: {thi_critical} 次")  # 打印危急次数
        print("="*70 + "\n")  # 打印结束分隔线
    
    def print_warning_details(self, num_samples: int = 5) -> None:
        """打印预警详情。

        参数:
            num_samples: 最多打印的预警条数（默认 5 条）。
        说明：若无预警则打印提示并返回；否则按时间顺序打印前若干条预警的详细信息。
        """
        if not self.warnings:  # 若没有预警记录
            print("暂无预警信息")  # 打印提示
            return  # 提前返回
        
        print("\n" + "="*70)  # 打印空行与分隔线
        print(f"预警详情示例 (前{min(num_samples, len(self.warnings))}条)")  # 打印标题与实际打印条数
        print("="*70)  # 打印分隔线
        
        for idx, warning in enumerate(self.warnings[:num_samples], 1):  # 遍历前 num_samples 条预警，编号从 1 开始
            print(f"\n[预警 {idx}] {warning['current_time'].strftime('%Y-%m-%d %H:%M')}")  # 打印预警编号与当前时间
            print(f"  预测时间: {warning['future_time'].strftime('%Y-%m-%d %H:%M')} "  # 打印预测对应的未来时间
                  f"(未来{self.pred_horizon}分钟)")  # 附注提前预警的分钟数
            print(f"  TD预测值: {warning['td_predicted']:.2f}℃ → {warning['td_status']}")  # 打印预测 TD 与状态
            print(f"  THI预测值: {warning['thi_predicted']:.2f} → {warning['thi_status']}")  # 打印预测 THI 与状态
            print(f"  预警类型: {warning['warning_type']}")  # 打印预警类型
            print(f"  调控建议: {warning['recommendation']}")  # 打印调控建议
        
        print("\n" + "="*70 + "\n")  # 打印结束分隔线

# ===================== 演示程序 =====================

def demo_monitoring():
    """环境监测模块演示。

    使用三种不同日龄的模拟环境数据逐条调用 check_environment_status 并打印结果，
    最后调用 get_environment_distribution 演示前后/左右温差分析。
    """
    print("\n" + "="*70)  # 打印空行与分隔线
    print("鸡舍环境实时监测模块演示")  # 打印演示标题
    print("="*70 + "\n")  # 打印结束分隔线
    
    monitor = EnvironmentMonitor()  # 创建环境监测器实例
    
    # 模拟不同日龄的环境监测
    test_cases = [  # 定义三组模拟监测数据
        {  # 第 1 组：5 日龄
            'age': 5,  # 日龄 5 天
            'temperature': 33.2,  # 温度 33.2℃
            'humidity': 68,  # 湿度 68%
            'wind_speed': 0.15,  # 风速 0.15 m/s
            'co2': 1200,  # CO2 1200 ppm
            'description': '5日龄鸡舍'  # 场景描述
        },
        {  # 第 2 组：20 日龄
            'age': 20,  # 日龄 20 天
            'temperature': 28.5,  # 温度 28.5℃
            'humidity': 62,  # 湿度 62%
            'wind_speed': 0.45,  # 风速 0.45 m/s
            'co2': 1400,  # CO2 1400 ppm
            'description': '20日龄鸡舍'  # 场景描述
        },
        {  # 第 3 组：35 日龄
            'age': 35,  # 日龄 35 天
            'temperature': 24.8,  # 温度 24.8℃
            'humidity': 57,  # 湿度 57%
            'wind_speed': 0.95,  # 风速 0.95 m/s
            'co2': 1600,  # CO2 1600 ppm（超过上限）
            'description': '35日龄鸡舍（CO2超限）'  # 场景描述
        },
    ]
    
    for case in test_cases:  # 遍历每组模拟数据
        print(f"\n监测场景: {case['description']}")  # 打印场景描述
        print("-" * 70)  # 打印分隔线
        
        result = monitor.check_environment_status(  # 调用监测方法检查该场景
            age=case['age'],  # 传入日龄
            temperature=case['temperature'],  # 传入温度
            humidity=case['humidity'],  # 传入湿度
            wind_speed=case['wind_speed'],  # 传入风速
            co2=case['co2']  # 传入 CO2
        )
        
        print(f"日龄: {result['age']}天")  # 打印日龄
        print(f"温度: {result['temperature']}℃ (适宜范围: {result['temp_range'][0]}-{result['temp_range'][1]}℃) "  # 打印温度及适宜范围
              f"→ {result['temp_status']}")  # 附注温度状态
        print(f"湿度: {result['humidity']}% (适宜范围: {result['humid_range'][0]}-{result['humid_range'][1]}%) "  # 打印湿度及适宜范围
              f"→ {result['humid_status']}")  # 附注湿度状态
        print(f"风速: {result['wind_speed']} m/s (适宜范围: {result['wind_range'][0]}-{result['wind_range'][1]} m/s) "  # 打印风速及适宜范围
              f"→ {result['wind_status']}")  # 附注风速状态
        print(f"CO2: {result['co2']} ppm (上限: {result['co2_limit']} ppm) → {result['co2_status']}")  # 打印 CO2 及状态
        print(f"综合状态: {result['overall_status']}")  # 打印综合状态
    
    # 环境分布分析
    print("\n\n环境分布差异分析")  # 打印分布分析标题
    print("-" * 70)  # 打印分隔线
    
    distribution = monitor.get_environment_distribution(  # 调用分布分析方法
        front_temp=28.5,  # 传入前方温度
        rear_temp=26.8,  # 传入后方温度
        left_temp=27.5,  # 传入左方温度
        right_temp=27.8  # 传入右方温度
    )
    
    print(f"前后温差: {distribution['front_rear_diff']:.1f}℃ "  # 打印前后温差
          f"→ {'均匀' if distribution['front_rear_uniform'] else '不均'}")  # 附注是否均匀
    print(f"左右温差: {distribution['left_right_diff']:.1f}℃ "  # 打印左右温差
          f"→ {'均匀' if distribution['left_right_uniform'] else '不均'}")  # 附注是否均匀
    print(f"整体分布: {'均匀' if distribution['overall_uniform'] else '不均'}")  # 打印整体均匀性
    print(f"改善建议: {distribution['recommendation']}")  # 打印改善建议

def demo_warning():
    """预警系统演示。

    使用四组模拟的 TD/THI 预测值调用 generate_warning，逐条打印预警类型与建议，
    最后输出预警统计摘要与详情示例。
    """
    print("\n\n" + "="*70)  # 打印空行与分隔线
    print("实时预警系统演示")  # 打印演示标题
    print("="*70 + "\n")  # 打印结束分隔线
    
    warning_system = WarningSystem(time_step=36, pred_offset=6)  # 创建预警系统实例（默认 36 步输入、6 步预测偏移）
    
    # 模拟预测结果
    test_predictions = [  # 定义四组模拟预测数据
        {'td': 1.5, 'thi': 72, 'desc': '正常环境'},  # 正常场景
        {'td': 3.2, 'thi': 75, 'desc': '偏热+THI警戒'},  # 偏热且 THI 警戒
        {'td': -2.8, 'thi': 79, 'desc': '偏冷+THI预警'},  # 偏冷且 THI 预警
        {'td': 2.1, 'thi': 83, 'desc': '偏热+THI危急'},  # 偏热且 THI 危急
    ]
    
    current_time = datetime.now()  # 以当前时间作为起始时间
    
    for idx, pred in enumerate(test_predictions):  # 遍历每组预测数据
        future_time = current_time + timedelta(minutes=30)  # 计算 30 分钟后的预测时刻
        
        warning = warning_system.generate_warning(  # 调用预警生成方法
            td_predicted=pred['td'],  # 传入预测 TD
            thi_predicted=pred['thi'],  # 传入预测 THI
            current_time=current_time,  # 传入当前时间
            future_time=future_time  # 传入预测时刻
        )
        
        print(f"预测场景 {idx+1}: {pred['desc']}")  # 打印场景编号与描述
        print(f"  TD预测: {pred['td']:.1f}℃ → {warning['td_status']}")  # 打印 TD 预测及状态
        print(f"  THI预测: {pred['thi']:.1f} → {warning['thi_status']}")  # 打印 THI 预测及状态
        print(f"  预警类型: {warning['warning_type']}")  # 打印预警类型
        print(f"  调控建议: {warning['recommendation']}")  # 打印调控建议
        print()  # 打印空行
        
        current_time = future_time  # 将预测时刻作为下一次的当前时间，形成滚动
    
    # 打印统计摘要
    warning_system.print_warning_summary()  # 打印预警统计摘要
    warning_system.print_warning_details(num_samples=3)  # 打印前 3 条预警详情

if __name__ == "__main__":  # 仅当脚本被直接运行时执行演示
    demo_monitoring()  # 运行环境监测演示
    demo_warning()  # 运行预警系统演示
    
    print("\n✓ 演示完成！")  # 打印演示结束提示


















































