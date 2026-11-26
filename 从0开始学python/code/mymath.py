# 第08章配套模块：演示"自己写模块"和 __main__ 开关
# 直接运行： python code/mymath.py   → 会执行自测
# 被导入：   import mymath           → 自测不执行

import math


def sigmoid(x):
    """把任意实数压到(0,1)——神经网络常用激活函数"""
    return 1 / (1 + math.exp(-x))


def softmax(logits):
    """把一组数变成和为1的概率分布——大模型输出层的标配"""
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]   # 减最大值防溢出（标准技巧）
    total = sum(exps)
    return [e / total for e in exps]


if __name__ == "__main__":
    # 只在直接运行本文件时执行
    print("自测 sigmoid:", sigmoid(0), sigmoid(2))
    print("自测 softmax:", softmax([2.0, 1.0, 0.1]))
