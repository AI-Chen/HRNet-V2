# HRNet-V2进行语义分割的tensoflow实现
## 文件简介
**utils文件夹**中包含*color_utils.py、data_utils.py、HighResolutionModule.py、Models_Block.py、Models_Config.py*.
1. *color_utils.py、data_utils.py*是进行数据预处理的功能函数；
2. *HighResolutionModule.py*是HRNet v2中的进行并行卷积融合的功能模块；
3. *Models_Block.py*是对卷积、批归一化、BasicBlock、Bottleneck的封装；
4. *Models_Config.py*包含了HRNet中每一个阶段的超参数设置；
**__pycache__文件夹**是PyCharm编译时产生的文件.
**utils文件夹**
**Model.py**是HRNet v2模型的实现
**HRNet v2_Train.py**是运用模型进行训练的文件。
## Getting started

