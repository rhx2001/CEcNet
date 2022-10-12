# 结构
```
dataset
├─final #最后分割好的片段，放dataset_try.dill
├─reg #数据集分割好的音乐片段，记得把拷给你的片段放进去再运行data_process2
└─reg2 #测试小批量片段
```

- 运行数据集用的是data_process2
- 运行代码用的是train_pc_c.py
- 模型是PCConvNet中的PCConvNetContrastive2
- dataUtil2.py里面修改音频提取特征的方式

结果：
```python
[Testing Contrastive Loss:  1.43163, Testing CE Loss: 1.11905, Testing Accuracy:  0.60278]
[Validation Contrastive Loss:  0.73874, Validation CE Loss: 1.07292, Validation Accuracy:  0.55000]
[Testing Contrastive Loss:  0.05539, Testing CE Loss: 1.01693, Testing Accuracy:  0.50177]
```
