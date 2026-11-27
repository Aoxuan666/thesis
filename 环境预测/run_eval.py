"""
预警评估脚本启动器。

本脚本在 d://resources//pythonTest 下扫描名称含“副本”的子目录，
按“副本”出现次数降序选出层级最深的一个，切换工作目录并把它加入模块搜索路径，
随后读取并执行该目录中的 eval_warning.py，以此在最深副本目录中运行预警评估流程。
"""
import os  # 导入操作系统接口模块，用于遍历目录与切换工作目录
import sys  # 导入解释器相关模块，用于修改模块搜索路径

base = r'd:\resources\pythonTest'  # 定义要扫描的根目录（原始字符串避免反斜杠转义）
target_dirs = []  # 初始化用于存放候选目录名的空列表
for item in os.listdir(base):  # 遍历根目录下的所有条目
    full_path = os.path.join(base, item)  # 拼接出当前条目的完整路径
    if os.path.isdir(full_path) and '副本' in item:  # 仅保留目录，且目录名中包含“副本”字样的条目
        target_dirs.append(item)  # 把符合条件的目录名加入候选列表

target_dirs.sort(key=lambda x: x.count('副本'), reverse=True)  # 按目录名中“副本”出现次数降序排序，副本最多的排在前面

if target_dirs:  # 只有找到候选目录时才执行后续切换与运行逻辑
    target_dir = os.path.join(base, target_dirs[0])  # 用排序后第一个（副本层级最深）的目录拼出目标路径
    os.chdir(target_dir)  # 把当前工作目录切换到目标目录
    sys.path.insert(0, target_dir)  # 把目标目录插入模块搜索路径首位，保证导入优先从该目录查找
    
    # 运行 eval_warning.py
    exec(open('eval_warning.py').read())  # 读取 eval_warning.py 的源码并在当前命名空间中执行，实现预警评估

















































