#!/usr/bin/env python3
"""
手写数字序列识别 - 完整修复版
专门处理分割问题
"""

import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import sys
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================
# 高级图片预处理
# ============================================

def preprocess_image_advanced(image_path):
    """
    高级预处理：增强对比度、去噪、自适应二值化
    
    参数:
        image_path: 图片路径
    
    返回:
        img_array: 预处理后的二值化图片数组 (uint8)
        original: 原始图片数组（用于调试）
    """
    # 1. 打开图片
    img = Image.open(image_path).convert('L')
    original = np.array(img)
    
    # 2. 增强对比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # 增强对比度
    
    # 3. 转换为数组
    img_array = np.array(img, dtype=np.uint8)
    
    # 4. 自适应二值化
    # 检测是白底黑字还是黑底白字
    mean_val = np.mean(img_array)
    if mean_val > 127:
        # 白底黑字，需要反转
        img_array = 255 - img_array
    
    # 5. 使用自适应阈值二值化
    from scipy.ndimage import gaussian_filter
    
    # 高斯模糊去噪
    blurred = gaussian_filter(img_array.astype(np.float32), sigma=1)
    
    # 自适应阈值（局部阈值）
    from skimage.filters import threshold_local
    local_thresh = threshold_local(blurred, block_size=15, offset=10)
    binary = (blurred > local_thresh).astype(np.uint8) * 255
    
    # 如果二值化效果不好，使用全局阈值
    if np.mean(binary) < 5 or np.mean(binary) > 250:
        # 使用Otsu全局阈值
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(img_array)
        binary = (img_array > thresh).astype(np.uint8) * 255
    
    return binary, original

def connected_components_analysis(binary_img, min_size=30, max_size=3000):
    """
    连通域分析，找到所有数字区域
    
    参数:
        binary_img: 二值化图片
        min_size: 最小区域大小
        max_size: 最大区域大小
    
    返回:
        regions: 区域列表 [(x1, y1, x2, y2, area), ...]
        labeled: 标记后的数组
    """
    from scipy.ndimage import label, find_objects
    
    # 标记连通域
    labeled, num_features = label(binary_img > 0)
    
    regions = []
    for i in range(1, num_features + 1):
        # 获取该连通域的坐标
        coords = np.where(labeled == i)
        area = len(coords[0])
        
        # 过滤太小或太大的区域
        if area < min_size or area > max_size:
            continue
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 添加边距
        margin = 5
        y_min = max(0, y_min - margin)
        y_max = min(binary_img.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(binary_img.shape[1], x_max + margin)
        
        regions.append({
            'x1': x_min, 'y1': y_min,
            'x2': x_max, 'y2': y_max,
            'area': area,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'aspect_ratio': (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 0
        })
    
    # 按x坐标排序（从左到右）
    regions.sort(key=lambda r: r['x1'])
    
    return regions, labeled

def split_connected_digits(binary_img, max_width=50):
    """
    分割粘连的数字
    
    参数:
        binary_img: 二值化图片
        max_width: 单个数字的最大宽度
    
    返回:
        regions: 分割后的区域列表
    """
    from scipy.ndimage import label
    
    # 投影分割
    # 计算垂直投影
    col_sums = np.sum(binary_img, axis=0)
    
    # 找到可能的分割点
    split_points = []
    in_digit = False
    gap_start = 0
    
    for i, sum_val in enumerate(col_sums):
        if sum_val > 0 and not in_digit:
            in_digit = True
        elif sum_val == 0 and in_digit:
            in_digit = False
            # 计算间隙宽度
            gap_width = i - gap_start
            if gap_width > 5:  # 间隙至少5像素
                split_points.append((gap_start, i))
            gap_start = i
        elif sum_val == 0 and not in_digit:
            gap_start = i
    
    # 如果没有足够的间隙，尝试等距分割
    if len(split_points) < 1:
        # 找到所有数字的边界
        non_zero_cols = np.where(col_sums > 0)[0]
        if len(non_zero_cols) > 0:
            left = non_zero_cols[0]
            right = non_zero_cols[-1]
            total_width = right - left
            
            # 估计数字个数
            avg_digit_width = 30  # 平均数字宽度
            estimated_digits = max(1, total_width // avg_digit_width)
            digit_width = total_width // estimated_digits
            
            # 等距分割
            for i in range(estimated_digits):
                x1 = left + i * digit_width
                x2 = x1 + digit_width
                if x2 > right:
                    x2 = right
                split_points.append((x1, x2))
    
    return split_points

def extract_digit_robust(binary_img, region, target_size=(28, 28)):
    """
    鲁棒的数字提取
    
    参数:
        binary_img: 二值化图片
        region: 区域 (x1, y1, x2, y2) 或分割点 (x1, x2)
        target_size: 目标大小
    
    返回:
        digit_img: 处理后的数字图片
    """
    # 处理不同的区域格式
    if len(region) == 2:  # (x1, x2)
        x1, x2 = region
        # 找到数字的垂直范围
        col_sums = np.sum(binary_img[:, x1:x2], axis=1)
        non_zero_rows = np.where(col_sums > 0)[0]
        if len(non_zero_rows) > 0:
            y1 = non_zero_rows[0]
            y2 = non_zero_rows[-1]
        else:
            # 如果没有找到数字，使用整个高度
            y1 = 0
            y2 = binary_img.shape[0]
        
        # 添加边距
        margin = 5
        y1 = max(0, y1 - margin)
        y2 = min(binary_img.shape[0], y2 + margin)
        x1 = max(0, x1 - margin)
        x2 = min(binary_img.shape[1], x2 + margin)
    else:
        x1 = region['x1']
        y1 = region['y1']
        x2 = region['x2']
        y2 = region['y2']
    
    # 提取数字
    digit = binary_img[y1:y2, x1:x2].copy()
    
    if digit.size == 0 or np.sum(digit) == 0:
        return np.zeros(target_size, dtype=np.float32)
    
    # 确保数据类型正确
    digit = digit.astype(np.uint8)
    
    # 去除空白边缘
    rows = np.where(np.sum(digit, axis=1) > 0)[0]
    cols = np.where(np.sum(digit, axis=0) > 0)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros(target_size, dtype=np.float32)
    
    digit = digit[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    
    # 确保数字至少3x3
    if digit.shape[0] < 3 or digit.shape[1] < 3:
        return np.zeros(target_size, dtype=np.float32)
    
    # 缩放并居中
    h, w = digit.shape
    target_h, target_w = target_size
    
    # 计算缩放比例
    max_size = 20
    scale = min(max_size / h, max_size / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    # 使用PIL缩放
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

def debug_visualize(image_path, binary_img, regions, output_path='result\debug_split.png'):
    """
    可视化分割过程，用于调试
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原始图片
        img = Image.open(image_path)
        ax1.imshow(img, cmap='gray')
        ax1.set_title('原始图片')
        ax1.axis('off')
        
        # 显示二值化图片和分割框
        ax2.imshow(binary_img, cmap='gray')
        ax2.set_title(f'分割结果 (找到 {len(regions)} 个区域)')
        
        # 绘制边界框
        for i, region in enumerate(regions):
            if isinstance(region, dict):
                x1, y1 = region['x1'], region['y1']
                x2, y2 = region['x2'], region['y2']
            else:
                x1, x2 = region
                y1, y2 = 0, binary_img.shape[0]
            
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, str(i+1), color='red', fontsize=12, weight='bold')
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📸 调试可视化已保存到: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

# ============================================
# 预测引擎
# ============================================

def predict_sequence_fixed(image_path, model_path='result\mnist_model.h5', 
                           visualize=False, debug=False):
    """
    修复版数字序列识别
    """
    # 1. 加载模型
    if not os.path.exists(model_path):
        alternatives = ['result\mnist_cnn_model.h5', 'result\mnist_model.h5']
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                break
        else:
            print(f"❌ 找不到模型文件")
            return None, None
    
    print(f"📂 加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ 模型加载成功!")
    
    # 2. 预处理图片
    print(f"\n🔍 处理图片: {image_path}")
    binary_img, original_img = preprocess_image_advanced(image_path)
    
    print(f"   📐 图片尺寸: {binary_img.shape}")
    print(f"   📊 二值化范围: {binary_img.min()} - {binary_img.max()}")
    
    # 3. 尝试多种分割策略
    print("\n✂️ 分割数字...")
    
    digits = []
    regions = []
    
    # 策略1：连通域分析
    regions_found, labeled = connected_components_analysis(binary_img)
    
    if regions_found:
        print(f"   ✅ 策略1 (连通域): 找到 {len(regions_found)} 个区域")
        
        # 检查是否有过大的区域（可能包含多个数字）
        max_area = 2000
        large_regions = [r for r in regions_found if r['area'] > max_area]
        
        if large_regions:
            print(f"   ⚠️ 发现 {len(large_regions)} 个过大区域，尝试二次分割")
            
            # 对于过大的区域，尝试二次分割
            for region in large_regions:
                x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
                sub_img = binary_img[y1:y2, x1:x2]
                sub_regions, _ = connected_components_analysis(sub_img, min_size=20)
                
                if sub_regions:
                    for sub_region in sub_regions:
                        # 将坐标转换回原图坐标系
                        sub_region['x1'] += x1
                        sub_region['x2'] += x1
                        sub_region['y1'] += y1
                        sub_region['y2'] += y1
                        regions.append(sub_region)
                else:
                    # 如果无法二次分割，使用投影分割
                    split_points = split_connected_digits(binary_img[y1:y2, x1:x2])
                    if split_points:
                        for x_start, x_end in split_points:
                            x_start += x1
                            x_end += x1
                            regions.append({'x1': x_start, 'y1': y1, 
                                          'x2': x_end, 'y2': y2, 
                                          'area': 100, 'width': x_end-x_start,
                                          'height': y2-y1, 'aspect_ratio': 0})
            # 添加其他正常大小的区域
            for r in regions_found:
                if r['area'] <= max_area:
                    regions.append(r)
        else:
            regions = regions_found
    else:
        # 策略2：投影分割
        print("   ⚠️ 连通域未找到区域，尝试投影分割")
        split_points = split_connected_digits(binary_img)
        if split_points:
            for x1, x2 in split_points:
                regions.append({'x1': x1, 'y1': 0, 'x2': x2, 
                              'y2': binary_img.shape[0], 'area': 100,
                              'width': x2-x1, 'height': binary_img.shape[0],
                              'aspect_ratio': 0})
    
    # 4. 提取数字
    if regions:
        for region in regions:
            digit_img = extract_digit_robust(binary_img, region)
            if np.sum(digit_img) > 0:
                digits.append(digit_img)
        
        print(f"   ✅ 最终提取 {len(digits)} 个数字")
    else:
        print("   ❌ 分割失败")
        return None, None
    
    # 5. 调试可视化
    if debug and regions:
        debug_visualize(image_path, binary_img, regions)
    
    # 6. 识别数字
    print("\n🔎 识别中...")
    print("-" * 50)
    
    result = ""
    details = []
    
    for i, digit_img in enumerate(digits):
        # 预测
        input_img = digit_img.reshape(1, 28, 28)
        if len(model.input_shape) == 4:
            input_img = input_img.reshape(1, 28, 28, 1)
        
        predictions = model.predict(input_img, verbose=0)[0]
        digit = np.argmax(predictions)
        confidence = np.max(predictions)
        
        result += str(digit)
        details.append({
            'position': i + 1,
            'digit': digit,
            'confidence': confidence
        })
        
        print(f"  {i+1}️⃣ 数字: {digit}, 置信度: {confidence:.2%}")
    
    print("-" * 50)
    
    # 7. 输出结果
    print("\n" + "=" * 50)
    print("📊 识别结果")
    print("=" * 50)
    print(f"📁 图片: {os.path.basename(image_path)}")
    print(f"🔢 识别数字: {result}")
    print(f"📈 检测到 {len(digits)} 个数字")
    
    if len(digits) < 7:
        print(f"⚠️ 期望7个数字，实际检测到 {len(digits)} 个")
        print("   建议: 确保数字之间有清晰间距")
    
    print("=" * 50)
    
    return result, details

# ============================================
# 命令行接口
# ============================================

def main():
    parser = argparse.ArgumentParser(description='手写数字序列识别 - 修复版')
    parser.add_argument('image_path', help='图片文件路径')
    parser.add_argument('--model', default='result\mnist_model.h5', help='模型文件路径')
    parser.add_argument('--visualize', action='store_true', help='显示结果')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ 图片不存在: {args.image_path}")
        sys.exit(1)
    
    result, details = predict_sequence_fixed(
        args.image_path,
        args.model,
        args.visualize,
        args.debug
    )

if __name__ == "__main__":
    main()