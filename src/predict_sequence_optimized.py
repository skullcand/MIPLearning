#!/usr/bin/env python3
"""
优化版手写数字序列识别
支持：智能分割、置信度阈值、候选显示、可视化、多种模型

使用方法：
    python predict_sequence_optimized.py <图片路径> [选项]
    
示例：
    python predict_sequence_optimized.py test_number.png
    python predict_sequence_optimized.py test_number.png --visualize
    python predict_sequence_optimized.py test_number.png --threshold 0.8
    python predict_sequence_optimized.py test_number.png --model mnist_cnn_model.h5
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import os
import argparse
from datetime import datetime

# ============================================
# 图片预处理
# ============================================

def preprocess_image(image_path):
    """
    预处理图片：灰度化、二值化、去噪
    
    参数:
        image_path: 图片路径
    
    返回:
        img_array: 预处理后的二值化图片数组 (dtype: uint8)
    """
    # 打开图片
    img = Image.open(image_path).convert('L')
    
    # 转换为numpy数组
    img_array = np.array(img, dtype=np.uint8)
    
    # 如果图片是白底黑字，反转
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # 二值化
    img_array = np.where(img_array > 30, 255, 0).astype(np.uint8)
    
    return img_array

def remove_noise(img_array, min_pixels=10):
    """
    去除小噪点
    
    参数:
        img_array: 二值化图片
        min_pixels: 最小像素数阈值
    
    返回:
        cleaned: 去噪后的图片
    """
    from scipy.ndimage import label
    
    # 标记连通域
    labeled, num_features = label(img_array > 0)
    
    cleaned = np.zeros_like(img_array)
    
    for i in range(1, num_features + 1):
        # 获取连通域像素数
        pixels = np.sum(labeled == i)
        if pixels >= min_pixels:
            cleaned[labeled == i] = 255
    
    return cleaned.astype(np.uint8)

# ============================================
# 数字分割
# ============================================

def find_digit_regions(img_array):
    """
    使用连通域分析找到所有数字区域
    
    参数:
        img_array: 二值化图片
    
    返回:
        regions: 每个数字的边界框列表 [(x1, y1, x2, y2), ...]
    """
    from scipy.ndimage import label
    
    # 二值化
    binary = img_array > 50
    
    # 标记连通域
    labeled, num_features = label(binary)
    
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
    
    参数:
        img_array: 原始图片数组 (uint8)
        region: 边界框 (x1, y1, x2, y2)
        target_size: 目标大小 (height, width)
    
    返回:
        digit_img: 处理后的数字图片 (28, 28), dtype: float32, 范围 0-1
    """
    x1, y1, x2, y2 = region
    
    # 提取数字区域
    digit = img_array[y1:y2, x1:x2].copy()
    
    if digit.size == 0:
        return np.zeros(target_size, dtype=np.float32)
    
    # 确保数据类型正确
    digit = digit.astype(np.uint8)
    
    # 去除空白边缘
    rows = np.where(np.sum(digit, axis=1) > 0)[0]
    cols = np.where(np.sum(digit, axis=0) > 0)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros(target_size, dtype=np.float32)
    
    digit = digit[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    
    # 计算缩放比例（保持宽高比）
    h, w = digit.shape
    target_h, target_w = target_size
    
    # 留边距，数字占20x20，居中放置
    max_size = 20
    scale = min(max_size / h, max_size / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 确保至少1像素
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    # 缩放 - 需要确保数据类型正确
    pil_img = Image.fromarray(digit, mode='L')
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(pil_img, dtype=np.float32)
    
    # 归一化到0-1
    if resized.max() > 0:
        resized = resized / 255.0
    
    # 创建目标画布
    canvas = np.zeros(target_size, dtype=np.float32)
    
    # 居中放置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def split_digits_simple(img_array, num_digits=None):
    """
    简单等距分割（备用方案）
    
    参数:
        img_array: 二值化图片
        num_digits: 数字个数
    
    返回:
        digits: 分割后的数字列表
    """
    # 裁剪左右空白
    col_sums = np.sum(img_array, axis=0)
    non_zero_cols = np.where(col_sums > 0)[0]
    
    if len(non_zero_cols) == 0:
        return []
    
    left = non_zero_cols[0]
    right = non_zero_cols[-1]
    cropped = img_array[:, left:right]
    
    # 如果指定了数字个数，等距分割
    if num_digits:
        width = cropped.shape[1]
        digit_width = width // num_digits
        
        digits = []
        for i in range(num_digits):
            start = i * digit_width
            end = (i + 1) * digit_width
            digit_img = cropped[:, start:end]
            digit_img = extract_digit_simple(digit_img)
            digits.append(digit_img)
        
        return digits
    
    # 自动检测：通过空白间隙分割
    digits = []
    col_sums_cropped = np.sum(cropped, axis=0)
    in_digit = False
    start = 0
    
    for i, sum_val in enumerate(col_sums_cropped):
        if sum_val > 0 and not in_digit:
            in_digit = True
            start = i
        elif sum_val == 0 and in_digit:
            in_digit = False
            if i - start > 5:
                digit_img = cropped[:, start:i]
                digit_img = extract_digit_simple(digit_img)
                digits.append(digit_img)
    
    if in_digit:
        digit_img = cropped[:, start:]
        digit_img = extract_digit_simple(digit_img)
        digits.append(digit_img)
    
    return digits

def extract_digit_simple(digit_img):
    """简单提取数字"""
    # 确保数据类型正确
    digit_img = digit_img.astype(np.uint8)
    
    rows = np.where(np.sum(digit_img, axis=1) > 0)[0]
    cols = np.where(np.sum(digit_img, axis=0) > 0)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    
    cropped = digit_img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    
    h, w = cropped.shape
    max_size = 20
    scale = min(max_size / h, max_size / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    pil_img = Image.fromarray(cropped, mode='L')
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(pil_img, dtype=np.float32)
    
    if resized.max() > 0:
        resized = resized / 255.0
    
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

# ============================================
# 预测引擎
# ============================================

def predict_with_candidates(model, digit_img, top_k=3):
    """
    预测数字，返回 Top-K 候选
    
    参数:
        model: 训练好的模型
        digit_img: 数字图片 (28, 28)
        top_k: 返回前 K 个候选
    
    返回:
        candidates: [(数字, 置信度), ...]
    """
    # 确保输入是 float32
    digit_img = digit_img.astype(np.float32)
    
    # 检查输入形状
    if len(digit_img.shape) == 2:
        input_img = digit_img.reshape(1, 28, 28)
    else:
        input_img = digit_img.reshape(1, 28, 28)
    
    # 检查模型是否需要通道维度
    if len(model.input_shape) == 4:  # (batch, height, width, channels)
        input_img = input_img.reshape(1, 28, 28, 1)
    
    predictions = model.predict(input_img, verbose=0)[0]
    
    # 获取 Top-K
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    candidates = [(int(idx), float(predictions[idx])) for idx in top_indices]
    
    return candidates

def predict_sequence_optimized(image_path, model_path='result\mnist_model.h5', 
                               threshold=0.7, visualize=False, top_k=3,
                               simple_split=False, num_digits=None):
    """
    优化版数字序列识别
    
    参数:
        image_path: 图片路径
        model_path: 模型文件路径
        threshold: 置信度阈值 (0-1)
        visualize: 是否可视化分割结果
        top_k: 显示前K个候选数字
        simple_split: 使用简单等距分割
        num_digits: 指定数字个数（用于简单分割）
    
    返回:
        result: 识别结果字符串
        details: 每个数字的详情
    """
    # ===== 1. 检查文件 =====
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return None, None
    
    # ===== 2. 加载模型 =====
    if not os.path.exists(model_path):
        # 尝试其他可能的模型路径
        alternatives = ['mnist_cnn_model.h5', 'mnist_model.h5', 'mnist_cnn.keras']
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                break
        else:
            print(f"❌ 找不到模型文件")
            print(f"   请确保以下文件之一存在:")
            for alt in alternatives:
                print(f"   - {alt}")
            return None, None
    
    try:
        print(f"📂 加载模型: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("✅ 模型加载成功!")
        
        # 显示模型信息
        model.summary()
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None
    
    # ===== 3. 预处理图片 =====
    print(f"\n🔍 处理图片: {image_path}")
    img_array = preprocess_image(image_path)
    
    # 显示图片信息
    print(f"   📐 图片尺寸: {img_array.shape}")
    print(f"   📊 像素范围: {img_array.min()} - {img_array.max()}")
    
    # ===== 4. 分割数字 =====
    print("\n✂️ 分割数字...")
    
    if simple_split:
        digits = split_digits_simple(img_array, num_digits)
        print(f"   使用简单等距分割")
    else:
        # 先去噪
        img_array = remove_noise(img_array, min_pixels=10)
        # 查找数字区域
        regions = find_digit_regions(img_array)
        
        if not regions:
            print("❌ 未检测到数字")
            return None, None
        
        # 提取每个数字
        digits = []
        for region in regions:
            digit_img = extract_digit(img_array, region)
            digits.append(digit_img)
    
    if not digits:
        print("❌ 分割失败，未检测到数字")
        return None, None
    
    print(f"✅ 检测到 {len(digits)} 个数字")
    
    # ===== 5. 识别每个数字 =====
    print("\n🔎 识别中...")
    print("-" * 50)
    
    result = ""
    details = []
    uncertain_positions = []
    
    for i, digit_img in enumerate(digits):
        # 获取候选
        candidates = predict_with_candidates(model, digit_img, top_k=top_k)
        
        digit, confidence = candidates[0]
        result += str(digit)
        
        # 记录详情
        details.append({
            'position': i + 1,
            'digit': digit,
            'confidence': confidence,
            'candidates': candidates
        })
        
        # 判断置信度
        is_certain = confidence >= threshold
        status = "✅" if is_certain else "⚠️"
        
        if not is_certain:
            uncertain_positions.append(i + 1)
        
        # 构建候选字符串
        cand_str = ", ".join([f"{d}({conf:.1%})" for d, conf in candidates[1:]])
        
        # 打印结果
        print(f"  {i+1}️⃣ {status} 数字: {digit}, 置信度: {confidence:.2%}")
        if cand_str and top_k > 1:
            print(f"        候选: {cand_str}")
    
    print("-" * 50)
    
    # ===== 6. 统计信息 =====
    if details:
        avg_confidence = np.mean([d['confidence'] for d in details])
        min_confidence = min([d['confidence'] for d in details])
        max_confidence = max([d['confidence'] for d in details])
    else:
        avg_confidence = min_confidence = max_confidence = 0
    
    # ===== 7. 输出结果 =====
    print("\n" + "=" * 50)
    print("📊 识别结果")
    print("=" * 50)
    print(f"📁 图片: {os.path.basename(image_path)}")
    print(f"🔢 识别数字: {result}")
    print(f"📈 平均置信度: {avg_confidence:.2%}")
    print(f"📊 置信度范围: {min_confidence:.2%} - {max_confidence:.2%}")
    
    if uncertain_positions:
        print(f"\n⚠️ 警告: 有 {len(uncertain_positions)} 个数字置信度低于 {threshold:.0%}")
        print(f"   位置: {uncertain_positions}")
        print("   建议: 检查这些数字是否书写清晰")
    
    # 评估整体质量
    if avg_confidence > 0.9:
        quality = "🌟 优秀"
    elif avg_confidence > 0.7:
        quality = "✅ 良好"
    elif avg_confidence > 0.5:
        quality = "⚠️ 一般"
    else:
        quality = "❌ 较差"
    
    print(f"📊 识别质量: {quality}")
    print("=" * 50)
    
    # ===== 8. 可视化 =====
    if visualize:
        visualize_digits(image_path, digits, details, threshold)
    
    return result, details

# ============================================
# 可视化
# ============================================

def visualize_digits(image_path, digits, details, threshold=0.7, output_path=None):
    """
    可视化分割出的每个数字
    
    参数:
        image_path: 原始图片路径
        digits: 分割出的数字列表
        details: 识别详情
        threshold: 置信度阈值
        output_path: 输出文件路径
    """
    try:
        import matplotlib.pyplot as plt
        
        if output_path is None:
            output_path = f"result\segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        num_digits = len(digits)
        
        # 创建图形：上面显示原始图片，下面显示分割的数字
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, num_digits * 1.5), 6))
        
        # 显示原始图片
        img = Image.open(image_path)
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f"原始图片: {os.path.basename(image_path)}")
        ax1.axis('off')
        
        # 显示分割的数字
        if num_digits <= 1:
            ax2.imshow(digits[0], cmap='gray')
            ax2.set_title(f"数字 {details[0]['digit']} ({details[0]['confidence']:.1%})")
            ax2.axis('off')
        else:
            # 计算网格布局
            cols = min(10, num_digits)
            rows = (num_digits + cols - 1) // cols
            
            for i, digit_img in enumerate(digits):
                row = i // cols
                col = i % cols
                
                # 创建子图
                ax = plt.subplot(rows, cols, i + 1)
                ax.imshow(digit_img, cmap='gray')
                
                # 显示预测结果
                detail = details[i]
                digit = detail['digit']
                confidence = detail['confidence']
                is_certain = confidence >= threshold
                status = "✓" if is_certain else "?"
                
                ax.set_title(f"{status} {digit} ({confidence:.1%})", 
                            color='green' if is_certain else 'red', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📸 可视化结果已保存到: {output_path}")
        plt.show()
        
    except ImportError:
        print("\n⚠️ 需要安装 matplotlib 才能可视化")
        print("   运行: pip install matplotlib")

# ============================================
# 命令行接口
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='优化版手写数字序列识别',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python predict_sequence_optimized.py test_number.png
  
  # 使用 CNN 模型
  python predict_sequence_optimized.py test_number.png --model mnist_cnn_model.h5
  
  # 显示候选数字
  python predict_sequence_optimized.py test_number.png --top-k 3
  
  # 可视化分割结果
  python predict_sequence_optimized.py test_number.png --visualize
  
  # 设置置信度阈值
  python predict_sequence_optimized.py test_number.png --threshold 0.8
  
  # 使用简单等距分割（指定数字个数）
  python predict_sequence_optimized.py test_number.png --simple --num-digits 4
        """
    )
    
    parser.add_argument('image_path', help='图片文件路径')
    parser.add_argument('--model', default='result\mnist_model.h5', 
                       help='模型文件路径 (默认: mnist_model.h5)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='置信度阈值 (0-1, 默认: 0.7)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='显示前K个候选数字 (默认: 3)')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化分割结果')
    parser.add_argument('--simple', action='store_true',
                       help='使用简单等距分割（适用于等距数字）')
    parser.add_argument('--num-digits', type=int, default=None,
                       help='指定数字个数（配合 --simple 使用）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.threshold < 0 or args.threshold > 1:
        print("❌ 阈值必须在 0-1 之间")
        sys.exit(1)
    
    if args.top_k < 1:
        print("❌ top-k 必须 >= 1")
        sys.exit(1)
    
    # 执行识别
    result, details = predict_sequence_optimized(
        image_path=args.image_path,
        model_path=args.model,
        threshold=args.threshold,
        visualize=args.visualize,
        top_k=args.top_k,
        simple_split=args.simple,
        num_digits=args.num_digits
    )
    
    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()