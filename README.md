# HRNet-V2进行语义分割的tensorflow实现
## 文件简介
+ **utils文件夹**中包含*color_utils.py、data_utils.py、HighResolutionModule.py、Models_Block.py、Models_Config.py*.
1. *color_utils.py、data_utils.py*是进行数据预处理的功能函数；
2. *HighResolutionModule.py*是HRNet v2中的进行并行卷积融合的功能模块；
3. *Models_Block.py*是对卷积、批归一化、BasicBlock、Bottleneck的封装；
4. *Models_Config.py*包含了HRNet中每一个阶段的超参数设置；
+ **__pycache__文件夹**是PyCharm编译时产生的文件.
+ **Model.py**是HRNet v2模型的实现
+ **HRNet v2_Train.py**是运用模型进行训练的文件。
## Getting started
1. 数据集使用已经经过预处理适合本模型进行训练的ISPRS 2D Vaihingen数据集。下载地址[百度网盘](链接：https://pan.baidu.com/s/1RqjWXTZOCPO4cRkW6SVglA 
提取码：cslj)。下载后解压至文件根目录；
2. 在根目录新建文件夹，命名为：ckpts。该文件夹用于存储cpkt文件； 
3. 运行HRNet v2_Train.py即可开始模型的训练。
## Dependency
+ tensorflow==1.8.0
+ numpy==1.15.2
+ opencv3==3.1.0
