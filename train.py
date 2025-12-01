import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.history = None

    def preprocess_data(self, file_path):
        """数据预处理"""
        # 读取数据
        df = pd.read_excel(file_path, sheet_name='Sheet1')

        # 定义特征
        categorical_features = ['机型', '药剂']
        numerical_features = ['解离常数', '作业高度', '作业速度', '风速', '风向与采集带夹角', '温度', '湿度']
        target = '基于药剂与蜜蜂嗅觉蛋白安全距离'

        # 处理分类特征
        X_categorical = df[categorical_features].copy()
        for feature in categorical_features:
            le = LabelEncoder()
            X_categorical[feature] = le.fit_transform(X_categorical[feature])
            self.label_encoders[feature] = le

        # 处理数值特征
        X_numerical = df[numerical_features].copy()
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(X_numerical_scaled, columns=numerical_features)

        # 合并特征
        X = pd.concat([X_categorical, X_numerical], axis=1)
        y = df[target]

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def plot_training_history(self):
        history_df = pd.DataFrame(self.history.history)
        plt.figure(figsize=(12, 6))
        plt.plot(history_df['loss'], label='训练损失')
        plt.plot(history_df['val_loss'], label='验证损失')
        plt.title(f'{self.model_name} - 训练历史')
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{self.model_name}_training_history.png')
        plt.close()


class AttentionTransformerModel(BaseModel):
    def __init__(self):
        super().__init__('MHA-Transformer')

    def build_model(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(64)(inputs)
        x = layers.Reshape((1, 64))(x)

        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=8)(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)

        ffn = layers.Dense(128, activation='relu')(x)
        ffn = layers.Dense(64)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1)(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )


def train_model():
    model = AttentionTransformerModel()
    X_train, X_test, y_train, y_test = model.preprocess_data("基于药剂与蜜蜂嗅觉蛋白安全距离.xlsx")
    model.build_model(input_shape=(X_train.shape[1],))

    # 训练配置
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'results/{model.model_name}_best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )

    model.history = model.model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # 绘制训练历史
    model.plot_training_history()

    # 保存模型和预处理工具
    model.model.save(f'results/{model.model_name}_final_model.h5')
    np.save("preprocessing/label_encoders.npy", model.label_encoders)
    np.save("preprocessing/scaler_params.npy", (model.scaler.mean_, model.scaler.scale_))


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("preprocessing"):
        os.makedirs("preprocessing")
    train_model()