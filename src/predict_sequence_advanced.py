"""
识别一串手写数字（高级版 - 投影法分割）
使用方法：
    python predict_sequence_advanced.py <图片路径>
示例：
    python predict_sequence_advanced.py handwritten_sequence.png
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import os
from scipy.ndimage import label, find_objects

def preprocess_for_sequence(image_path):
    """
    预处理图片：增强对比度，去除噪声
    """
    # 打开图片
    img = Image.open(image_path).convert('L')
    
    # 使用Otsu自适应阈值
    img_array = np.array(img)
    
    # 使用更智能的二值化
    # 如果是白底黑字，反转
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # 增强对比度
    # 将像素值映射到0-255
    img_array = np.clip(img_array, 0, 255)
    
    return img_array

def find_digit_regions(img_array):
    """
    使用连通域分析找到所有数字区域
    
    返回:
        regions: 每个数字的边界框列表 [(x1, y1, x2, y2), ...]
    """
    # 二值化：大于阈值的是数字
    binary = img_array > 50
    
    # 使用连通域标记
    labeled, num_features = label(binary)
    
    # 找到每个连通域的位置
    regions = []
    for i in range(1, num_features + 1):
        # 获取该连通域的坐标
        coords = np.where(labeled == i)
        if len(coords[0]) > 20:  # 至少20个像素
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # 添加边距
            margin = 2
            y_min = max(0, y_min - margin)
            y_max = min(img_array.shape[0], y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(img_array.shape[1], x_max + margin)
            
            regions.append((x_min, y_min, x_max, y_max))
    
    # 按x坐标排序（从左到右）
    regions.sort(key=lambda r: r[0])
    
    return regions

def extract_digit(img_array, region, target_size=(28, 28)):
    """
    从图片中提取单个数字并调整到目标大小
    """
    x1, y1, x2, y2 = region
    
    # 提取数字区域
    digit = img_array[y1:y2, x1:x2]
    
    if digit.size == 0:
        return np.zeros(target_size)
    
    # 确保数字区域是黑底白字
    # 如果数字区域大部分是白色，反转
    if np.mean(digit) > 127:
        digit = 255 - digit
    
    # 去除白边（如果数字偏小）
    rows = np.where(np.sum(digit, axis=1) > 0)[0]
    cols = np.where(np.sum(digit, axis=0) > 0)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros(target_size)
    
    digit = digit[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    
    # 计算缩放比例（保持宽高比）
    h, w = digit.shape
    target_h, target_w = target_size
    
    # 留边距，数字占18x18，居中放置
    max_size = 20
    scale = min(max_size / h, max_size / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 确保至少1像素
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    # 缩放
    pil_img = Image.fromarray(digit)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(pil_img)
    
    # 归一化到0-1
    if resized.max() > 0:
        resized = resized / 255.0
    
    # 创建目标画布
    canvas = np.zeros(target_size)
    
    # 居中放置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def predict_sequence_advanced(image_path, model_path='result\mnist_model.h5'):
    """
    识别一串手写数字（高级版）
    """
    # 加载模型
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None
    
    print(f"📂 加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ 模型加载成功!")
    
    # 预处理图片
    print(f"🔍 处理图片: {image_path}")
    img_array = preprocess_for_sequence(image_path)
    
    # 找到所有数字区域
    print("🔎 查找数字区域...")
    regions = find_digit_regions(img_array)
    
    if not regions:
        print("❌ 未检测到数字")
        return None, None
    
    print(f"✅ 检测到 {len(regions)} 个数字区域")
    
    # 逐个提取并识别
    result = ""
    details = []
    
    for i, region in enumerate(regions):
        # 提取数字
        digit_img = extract_digit(img_array, region)
        
        # 预测
        input_img = digit_img.reshape(1, 28, 28)
        predictions = model.predict(input_img, verbose=0)
        digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        result += str(digit)
        details.append({
            'position': i + 1,
            'digit': digit,
            'confidence': confidence,
            'region': region
        })
        
        print(f"  {i+1}️⃣ 数字: {digit}, 置信度: {confidence:.2%}")
    
    return result, details

def visualize_regions(image_path, regions, output_path='result\segmented.png'):
    """
    可视化分割结果（需要matplotlib）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        img = Image.open(image_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(img, cmap='gray')
        
        for i, (x1, y1, x2, y2) in enumerate(regions):
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{i+1}', color='red', fontsize=12, weight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"📸 分割结果已保存到: {output_path}")
        plt.show()
    except ImportError:
        print("⚠️ 需要安装 matplotlib: pip install matplotlib")

def main():
    if len(sys.argv) < 2:
        print("\n" + "=" * 50)
        print("MNIST 手写数字序列识别（高级版）")
        print("=" * 50)
        print("\n使用方法:")
        print(f"  python {os.path.basename(__file__)} <图片路径> [--visualize]")
        print("\n示例:")
        print(f"  python {os.path.basename(__file__)} sequence.png")
        print(f"  python {os.path.basename(__file__)} sequence.png --visualize")
        print("\n" + "=" * 50)
        sys.exit(1)
    
    image_path = sys.argv[1]
    visualize = '--visualize' in sys.argv
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        sys.exit(1)
    
    # 识别
    result, details = predict_sequence_advanced(image_path)
    
    if result is not None:
        print("\n" + "=" * 50)
        print("📊 识别结果")
        print("=" * 50)
        print(f"📁 图片: {os.path.basename(image_path)}")
        print(f"🔢 识别数字: {result}")
        
        # 计算平均置信度
        avg_conf = np.mean([d['confidence'] for d in details])
        print(f"📈 平均置信度: {avg_conf:.2%}")
        print("=" * 50)
        
        # 可视化
        if visualize and details:
            regions = [d['region'] for d in details]
            visualize_regions(image_path, regions)

if __name__ == "__main__":
    main()