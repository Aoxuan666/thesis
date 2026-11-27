#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的测试脚本，直接在当前目录运行
本脚本是预警系统评估模块的简易启动入口：先把当前工作目录切换到脚本自身所在目录，
确保后续按相对路径读取数据时不会因启动位置不同而出错；随后导入 warning_evaluation
模块中的 main_example 示例函数，并在脚本被直接执行时调用它，便于开发阶段快速验证
整个评估流程是否可正常运行。
"""
import os  # 导入 os 模块，用于切换当前工作目录
import sys  # 导入 sys 模块，供系统及路径相关操作备用

# 确保在正确的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 把工作目录切换到脚本自身所在目录，保证相对路径可用

from warning_evaluation import main_example  # 从 warning_evaluation 模块导入示例主函数

if __name__ == '__main__':  # 仅当本脚本被直接运行时才执行下面的调用
    main_example()  # 调用示例主函数，启动预警评估流程















































































































































