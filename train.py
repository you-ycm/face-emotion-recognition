from utils.data_loader import load_data
from utils.model_builder import build_cnn_model
from utils.trainer import train_model

# 数据集路径（你的项目路径）
data_dir = './data'

# 加载数据
print("正在加载数据...")
train_gen, val_gen = load_data(data_dir, batch_size=64)

# 构建模型
print("正在构建模型...")
model = build_cnn_model()

# 打印模型结构
model.summary()

# 训练模型
print("开始训练...")
history = train_model(model, train_gen, val_gen, epochs=50)

print("训练完成！模型已保存到 models/emotion_model.h5")