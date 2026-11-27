"""PyTorch 模型检查点（checkpoint）结构查看脚本。

本脚本加载 THI 目录下的 TCN 模型权重文件，先打印除模型权重之外的元信息键，
再逐项打印 model_state_dict 中各参数/缓冲区的名称与张量形状，便于核对网络结构与权重完整性。
"""

import torch  # 导入 PyTorch，用于加载 .pth 检查点文件

path = 'THI/pth/Model_TCN_pytorch.pth'  # 指定待查看的模型权重文件路径
ck = torch.load(path, map_location='cpu', weights_only=False)  # 将检查点加载到 CPU，并允许还原非纯权重对象
print('meta keys:', [k for k in ck.keys() if k != 'model_state_dict'])  # 打印除 model_state_dict 之外的所有元信息键
print('model_state_dict keys:')  # 打印权重字典键列表的标题
for k, v in ck['model_state_dict'].items():  # 遍历模型状态字典中的每一项（参数名 -> 张量）
    print(f'  {k}: {v.shape}')  # 打印参数名称及其张量形状
