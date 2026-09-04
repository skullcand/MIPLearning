import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.image as mpimg  # 专门用来读图片

# ========== 1. 读取你的照片 ==========
# 把路径换成你的照片路径！！！
img_path = 'eg2.jpg'  # ← 改成你自己的照片路径

# 读取照片（自动变成 numpy 数组）
img_original = mpimg.imread(img_path)

# 如果是彩色图（RGB三通道），转为灰度图
if len(img_original.shape) == 3:
    img_gray = np.dot(img_original[..., :3], [0.2989, 0.5870, 0.1140])
else:
    img_gray = img_original

# 归一化到 0~1 方便处理
img_gray = img_gray / np.max(img_gray)

print(f"图片尺寸: {img_gray.shape}")
print(f"像素值范围: {img_gray.min():.3f} ~ {img_gray.max():.3f}")

# ========== 2. 显示原图 ==========
plt.figure(figsize=(8, 8))
plt.imshow(img_gray, cmap='gray')
plt.title('Your Original Photo')
plt.axis('off')
plt.show()

# ========== 3. 傅里叶变换 + 频谱可视化 ==========
f_transform = fft2(img_gray)
f_shifted = fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_shifted) + 1)

plt.figure(figsize=(8, 8))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frequency Spectrum (Log Scale)')
plt.axis('off')
plt.show()

# ========== 4. 低通滤波（模糊/去噪） ==========
rows, cols = img_gray.shape
crow, ccol = rows // 2, cols // 2

# 调整半径控制模糊程度（数值越小越模糊）
radius_low = 30  

mask_low = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= radius_low**2:
            mask_low[i, j] = 1

f_filtered_low = f_shifted * mask_low
img_lowpass = np.real(ifft2(ifftshift(f_filtered_low)))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_low, cmap='gray')
plt.title(f'Low-pass Mask (radius={radius_low})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_lowpass, cmap='gray')
plt.title('Blurred Photo (Low Frequency Only)')
plt.axis('off')
plt.show()

# ========== 5. 高通滤波（边缘提取） ==========
radius_high = 30
mask_high = np.ones((rows, cols))
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= radius_high**2:
            mask_high[i, j] = 0

f_filtered_high = f_shifted * mask_high
img_highpass = np.real(ifft2(ifftshift(f_filtered_high)))

# 因为高通滤波后会有负值，需要重新归一化到 0~1 显示
img_highpass_normalized = (img_highpass - img_highpass.min()) / (img_highpass.max() - img_highpass.min())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_high, cmap='gray')
plt.title(f'High-pass Mask (radius={radius_high})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_highpass_normalized, cmap='gray')
plt.title('Edge Detection (High Frequency Only)')
plt.axis('off')
plt.show()

# ========== 6. 带通滤波（只保留特定频率范围） ==========
# 只保留中间频率（去掉极低频和极高频）
radius_inner = 10   # 内圈半径（去掉直流）
radius_outer = 60   # 外圈半径（去掉超高频率）

mask_bandpass = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        dist = (i - crow)**2 + (j - ccol)**2
        if radius_inner**2 < dist <= radius_outer**2:
            mask_bandpass[i, j] = 1

f_filtered_band = f_shifted * mask_bandpass
img_bandpass = np.real(ifft2(ifftshift(f_filtered_band)))
img_bandpass_normalized = (img_bandpass - img_bandpass.min()) / (img_bandpass.max() - img_bandpass.min())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_bandpass, cmap='gray')
plt.title(f'Band-pass Mask ({radius_inner}~{radius_outer})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_bandpass_normalized, cmap='gray')
plt.title('Band-pass Filtered Photo')
plt.axis('off')
plt.show()

# ========== 7. 理想重建（证明逆变换无损失） ==========
img_reconstructed = np.real(ifft2(ifftshift(f_shifted)))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Photo')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_reconstructed, cmap='gray')
plt.title('Reconstructed from Spectrum (Perfect)')
plt.axis('off')
plt.show()

# ========== 8. 彩蛋：只保留水平/垂直纹理 ==========
# 只保留垂直方向的频率（水平轴上的亮斑）
mask_vertical = np.zeros((rows, cols))
keep_width = 5  # 保留宽度
for offset in [-30, 30]:  # 选取特定频率
    if 0 <= ccol + offset < cols:
        mask_vertical[crow-keep_width:crow+keep_width+1, 
                      ccol+offset-keep_width:ccol+offset+keep_width+1] = 1

f_vertical = f_shifted * mask_vertical
img_vertical = np.real(ifft2(ifftshift(f_vertical)))
img_vertical_normalized = (img_vertical - img_vertical.min()) / (img_vertical.max() - img_vertical.min())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_vertical, cmap='gray')
plt.title('Only Horizontal Frequencies')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_vertical_normalized, cmap='gray')
plt.title('Only Vertical Edges Preserved')
plt.axis('off')
plt.show()