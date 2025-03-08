# HUST-EIC深度学习课程作业
## 项目结构
```
.
├── activation.py      # 激活函数实现 (Sigmoid, ReLU, LeakyReLU, Tanh, Softmax)
├── model.py           # 神经网络模型实现 (全连接层、神经网络模型)
├── optimizer.py       # 优化器实现 (MiniBatchGradientDescent, Adam)
├── utils.py           # 工具函数 (损失函数、L2正则化、数据加载等)
├── visualize.py       # 可视化工具
└── train.ipynb        # 训练和评估模型的 Jupyter 笔记本
```
## 运行方式
```
python train.py [命令行参数]
```

### 命令行参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--model` | 选择预定义模型 | `custom` | `baseline`, `model1`, `model2`, `custom` |
| `--hidden` | 隐藏层维度 (逗号分隔) | `256,128` | 任意正整数序列 |
| `--activation` | 激活函数 | `relu` | `relu`, `leaky_relu`, `tanh`, `sigmoid` |
| `--optimizer` | 优化器 | `sgd` | `sgd`, `momentum`, `adam` |
| `--lr` | 学习率 | `0.01` | 任意正浮点数 |
| `--batch_size` | 批量大小 | `32` | 任意正整数 |
| `--epochs` | 训练轮数 | `100` | 任意正整数 |
| `--lambda` | L2正则化系数 | `1e-4` | 任意非负浮点数 |
| `--dropout` | Dropout比例 | `0.0` | 0到1之间的浮点数 |
| `--seed` | 随机种子 | `42` | 任意整数 |

**预定义模型**
* baseline: 两层隐藏层[256, 128]，使用ReLU激活函数，L2系数=1e-4
* model1: 两层隐藏层[512, 256]，使用LeakyReLU激活函数，L2系数=1e-3
* model2: 一层隐藏层[128]，使用Tanh激活函数，无正则化(L2系数=0)
 
运行```python --model 预定义模型```可以训练作业中提到的基础模型，需要调参时运行```python 指定参数``` 即可。
## 参考资料
[动手学深度学习](https://zh-v2.d2l.ai/)