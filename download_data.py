import kagglehub
import shutil
import os

# 下载数据集
path = kagglehub.dataset_download("msambare/fer2013")
print(f"数据集下载到: {path}")

# 解压并复制到项目目录
# 具体操作见控制台输出后手动处理