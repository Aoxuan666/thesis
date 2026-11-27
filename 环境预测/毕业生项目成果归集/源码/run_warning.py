# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8
"""
预警仿真脚本启动器。

本脚本通过 subprocess 以独立子进程方式调用同目录下的 warning_simulation.py，
把其标准输出直接打印到当前终端，并在最后输出子进程的退出码，
便于在统一入口下运行和排查预警仿真流程。
"""
import subprocess, sys, os  # 导入子进程、解释器信息与操作系统接口模块

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'warning_simulation.py')  # 以当前脚本所在目录为基准拼接预警仿真脚本 warning_simulation.py 的完整路径
print('Running:', script)  # 打印即将运行的预警仿真脚本路径
result = subprocess.run([sys.executable, script], capture_output=False, text=True)  # 调用当前 Python 解释器以子进程方式运行该脚本，并共享标准输出
print('Exit code:', result.returncode)  # 打印子进程运行结束后的退出码，0 通常表示成功



















































