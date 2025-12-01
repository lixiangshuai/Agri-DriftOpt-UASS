import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MHATransformerPredictor:
    def __init__(self):
        # 加载模型
        self.model = tf.keras.models.load_model("results/MHA-Transformer_best_model.h5")
        # 加载预处理工具
        self.label_encoders = np.load("preprocessing/label_encoders.npy", allow_pickle=True, encoding='latin1').item()
        scaler_params = np.load("preprocessing/scaler_params.npy", allow_pickle=True, encoding='latin1')
        self.scaler = StandardScaler()
        self.scaler.mean_ = scaler_params[0]
        self.scaler.scale_ = scaler_params[1]

    def preprocess_input(self, input_data):
        # 输入格式: [机型, 药剂, 作业高度, 作业速度, 风速, 风向与采集带夹角, 温度, 湿度]
        categorical = pd.DataFrame({
            '机型': [input_data[0]],
            '药剂': [input_data[1]]
        })
        numerical = pd.DataFrame({
            '解离常数':[input_data[2]],
            '作业高度': [input_data[3]],
            '作业速度': [input_data[4]],
            '风速': [input_data[5]],
            '风向与采集带夹角': [input_data[6]],
            '温度': [input_data[7]],
            '湿度': [input_data[8]]
        }, dtype=float)

        # 编码分类特征
        for col in ['机型', '药剂']:
            le = LabelEncoder()
            le.classes_ = self.label_encoders[col].classes_
            categorical[col] = le.transform(categorical[col])

        # 标准化数值特征
        numerical_scaled = self.scaler.transform(numerical)
        processed = np.concatenate([categorical.values, numerical_scaled], axis=1)
        return processed

    def predict(self, input_data):
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)
        return prediction[0][0]


# 示例使用
if __name__ == "__main__":
    # 输入参数示例: [机型, 药剂, 解离常数,作业高度, 作业速度, 风速, 风向与采集带夹角, 温度, 湿度]
    input_example = [1, 1,0.00001319, 2, 3, 1.7, 15, 27.45, 13.45]

    print('开始')
    predictor = MHATransformerPredictor()
    result = predictor.predict(input_example)
    print(f"预测的实际飘移距离: {result:.4f}")