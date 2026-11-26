# 毕业项目：从零手写神经网络

完整讲解见 `../lessons/15_毕业项目_手写神经网络.md`。

## 运行

```
cd 从0开始学python
python project/pure_python_net.py      # 纯 Python，零依赖，学习 y = x²    （约 3 秒）
python project/numpy_net.py            # NumPy 版，学习 y = sin(x)         （约 0.5 秒）
```

## 两个文件的关系

| | pure_python_net.py | numpy_net.py |
|---|---|---|
| 依赖 | 无 | numpy |
| 数据处理方式 | 一次一个样本，for 循环 | 一次一批样本，矩阵运算 |
| 网络 | 1 → 16 → 1 (tanh) | 1 → 32 → 1 (tanh / relu 可切换) |
| 学的函数 | y = x² | y = sin(x) |
| 用途 | 看清每一个乘法加法 | 看清"向量化"怎么把 for 变成 @ |

两份代码的**类名、方法名、训练循环完全一致**，建议并排打开对照着读。

## 预期结果

- 纯 Python 版：loss 从 ~0.13 降到 ~0.00003，预测误差 < 0.02
- NumPy 版：loss 从 ~0.31 降到 ~0.0024，中间区域预测误差 < 0.01（边缘略大是正常的）

如果你的 loss 不降反升或变成 `nan`，最可能是学习率太大——这本身就是毕业作业第 1 题。
