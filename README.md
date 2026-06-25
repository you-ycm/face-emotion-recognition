# 基于CNN的人脸情绪识别系统

基于卷积神经网络（CNN）和FER2013数据集构建的轻量级人脸情绪识别工具，通过OpenCV实现实时摄像头人脸检测与7种基本情绪分类。

## 演示视频
[点击观看项目完整功能演示](https://www.bilibili.com/video/BV1Za7h6YErV/)

## 技术栈
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- FER2013数据集

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 下载数据集：`python download_data.py`
3. 训练模型：`python train.py`
4. 实时识别：`python main.py`

## 模型性能
- 参数量：35.6万（约1.36MB）
- 验证集准确率：65%-70%
- 推理延迟：<0.5秒/帧

## 目录结构
├── data/ # 数据集（需自行下载）
├── models/ # 训练好的模型文件
├── utils/ # 工具模块
│ ├── data_loader.py # 数据加载
│ ├── model_builder.py# 模型构建
│ └── trainer.py # 模型训练
├── main.py # 实时识别入口
├── train.py # 模型训练入口
└── requirements.txt # 项目依赖


## 使用说明
- 运行 `main.py` 前请确保摄像头可用。
- 模型训练时间取决于CPU性能，建议使用GPU加速。
- 项目在 CPU 环境下即可流畅运行实时识别（<0.5秒/帧）。

## 注意事项
- 本项目需在本地 Python 环境运行，无法直接部署为网页应用。
- 演示视频展示了完整功能，可点击上方链接观看。