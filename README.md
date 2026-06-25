# 基于CNN的人脸情绪识别系统

基于卷积神经网络（CNN）和FER2013数据集构建的轻量级人脸情绪识别工具，通过OpenCV实现实时摄像头人脸检测与7种基本情绪分类。

## 技术栈
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- FER2013数据集

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 训练模型：`python train.py`
3. 实时识别：`python main.py`

## 模型性能
- 参数量：35.6万（约1.36MB）
- 验证集准确率：65%-70%
- 推理延迟：<0.5秒/帧