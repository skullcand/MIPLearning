"""
训练 MNIST 手写数字识别模型 - 使用国内镜像源
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import time
import ssl
import urllib.request

def download_mnist_from_mirror():
    """从镜像源下载 MNIST 数据集"""
    # 忽略 SSL 警告
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # 镜像源地址（使用清华大学镜像）
    mirror_urls = [
        "https://mirrors.tuna.tsinghua.edu.cn/tensorflow/keras/datasets/mnist.npz",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    ]
    
    cache_dir = os.path.expanduser("~/.keras/datasets")
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, "mnist.npz")
    
    # 如果文件已存在，直接加载
    if os.path.exists(file_path):
        print("📂 使用本地缓存的数据集")
        with np.load(file_path, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
            return (x_train, y_train), (x_test, y_test)
    
    # 尝试从镜像源下载
    for url in mirror_urls:
        try:
            print(f"📥 尝试从 {url} 下载...")
            urllib.request.urlretrieve(url, file_path)
            print("✅ 下载成功")
            with np.load(file_path, allow_pickle=True) as f:
                x_train, y_train = f['x_train'], f['y_train']
                x_test, y_test = f['x_test'], f['y_test']
                return (x_train, y_train), (x_test, y_test)
        except Exception as e:
            print(f"⚠️ 从 {url} 下载失败: {e}")
            continue
    
    raise Exception("所有下载源均失败，请检查网络连接")

def train_model():
    """训练并保存模型"""
    print("=" * 50)
    print("开始训练 MNIST 手写数字识别模型")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n📂 加载 MNIST 数据集...")
    try:
        (x_train, y_train), (x_test, y_test) = download_mnist_from_mirror()
        print("✅ 数据集加载成功")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        raise
    
    # 2. 预处理
    print("🔄 预处理数据...")
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"   训练集: {x_train.shape[0]} 张图片")
    print(f"   测试集: {x_test.shape[0]} 张图片")
    
    # 3. 构建模型
    print("🏗️ 构建模型...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # 4. 训练
    print("🚀 开始训练...")
    start_time = time.time()
    history = model.fit(x_train, y_train, 
                        epochs=5, 
                        batch_size=32,
                        validation_data=(x_test, y_test),
                        verbose=1)
    train_time = time.time() - start_time
    
    # 5. 评估
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ 训练完成!")
    print(f"   ⏱️ 训练耗时: {train_time:.2f} 秒")
    print(f"   📊 测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 6. 保存模型
    model_path = 'mnist_model.h5'
    model.save(model_path)
    print(f"💾 模型已保存到: {model_path}")
    print("=" * 50)
    
    return model

if __name__ == "__main__":
    train_model()