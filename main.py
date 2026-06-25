import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载模型
print("正在加载模型...")
model = load_model('models/emotion_model.h5')
print("模型加载成功！")

# 情绪标签
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def preprocess_face(face_img, target_size=(48, 48)):
    """
    预处理人脸图像
    """
    # 转为灰度图
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # 调整大小
    face_img = cv2.resize(face_img, target_size)

    # 归一化并增加维度
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)

    return face_img


# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头，请检查摄像头是否连接")
    exit()

print("按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图进行人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_roi = frame[y:y + h, x:x + w]

        # 预处理
        processed_face = preprocess_face(face_roi)

        # 预测情绪
        predictions = model.predict(processed_face, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion_label = EMOTION_LABELS[emotion_idx]
        confidence = np.max(predictions[0])

        # 绘制人脸框和标签
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{emotion_label}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow('情绪识别系统 - 按q退出', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()