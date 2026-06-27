"""drnet: DR + DME 多任务识别系统。

子模块:
    data    数据预处理、Dataset、增广、划分
    models  主干、注意力、序数头、多任务网络
    losses  序数损失、LDAM、多任务总损失
    engine  训练循环、指标、解耦微调
    explain Grad-CAM 可解释性
    utils   配置、随机种子、日志、checkpoint
"""

__version__ = "0.1.0"
