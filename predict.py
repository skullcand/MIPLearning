"""
预测手写数字图片
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
import glob

def predict_single_image(image_path, model):
    """
    预测单张手写数字图片
    
    参数:
        image_path: 图片路径
        model: 训练好的模型
    
    返回:
        predicted_digit: 预测的数字
        confidence: 置信度
    """
    try:
        # 1. 打开图片并转换为灰度图
        img = Image.open(image_path).convert('L')
        
        # 2. 调整大小为 28x28
        img = img.resize((28, 28))
        
        # 3. 转换为 numpy 数组
        img_array = np.array(img)
        
        # 4. 反转颜色（如果是白底黑字）
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # 5. 归一化
        img_array = img_array / 255.0
        
        # 6. 调整形状
        img_array = img_array.reshape(1, 28, 28)
        
        # 7. 预测
        predictions = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        return predicted_digit, confidence
    
    except Exception as e:
        print(f"❌ 处理图片失败: {e}")
        return None, 0

def batch_predict(folder_path, model):
    """
    批量预测文件夹中的所有图片
    
    参数:
        folder_path: 图片文件夹路径
        model: 训练好的模型
    """
    # 支持常见的图片格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"❌ 在文件夹 {folder_path} 中未找到图片文件")
        return
    
    print(f"\n📁 找到 {len(image_files)} 张图片")
    print("-" * 50)
    
    for img_path in sorted(image_files):
        digit, confidence = predict_single_image(img_path, model)
        if digit is not None:
            # 获取文件名
            filename = os.path.basename(img_path)
            print(f"✅ {filename:20s} -> 预测: {digit}, 置信度: {confidence:.2%}")

def main():
    """主函数"""
    # 检查模型是否存在
    model_path = 'mnist_model.h5'
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在，请先运行 train.py 训练模型")
        sys.exit(1)
    
    # 加载模型
    print("📂 加载模型...")
    model = tf.keras.models.load_model(model_path)
    print("✅ 模型加载成功!")
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print(f"  预测单张图片: python {os.path.basename(__file__)} <图片路径>")
        print(f"  批量预测:     python {os.path.basename(__file__)} <文件夹路径>")
        print("\n示例:")
        print(f"  python {os.path.basename(__file__)} test_images/digit_7.png")
        print(f"  python {os.path.basename(__file__)} test_images/")
        sys.exit(1)
    
    target_path = sys.argv[1]
    
    # 判断是文件还是文件夹
    if os.path.isfile(target_path):
        # 单张图片预测
        print(f"\n🔍 预测图片: {target_path}")
        digit, confidence = predict_single_image(target_path, model)
        if digit is not None:
            print(f"   ✅ 预测结果: {digit}")
            print(f"   📊 置信度: {confidence:.2%}")
    
    elif os.path.isdir(target_path):
        # 批量预测
        batch_predict(target_path, model)
    
    else:
        print(f"❌ 路径不存在: {target_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()