import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift  

========== 1. 生成棋盘格图片 ==========
grid_size = 8          # 8x8 棋盘格
pixel_per_cell = 32    # 每个格子 32x32 像素
img_size = grid_size * pixel_per_cell  # 256x256

chessboard = np.zeros((img_size, img_size))

for i in range(grid_size):
    for j in range(grid_size):
        if (i + j) % 2 == 0:  # 黑白相间
            chessboard[i*pixel_per_cell:(i+1)*pixel_per_cell, 
                       j*pixel_per_cell:(j+1)*pixel_per_cell] = 255

plt.figure(figsize=(6,6))
plt.imshow(chessboard, cmap='gray')
plt.title('Original Chessboard (Time Domain)')
plt.axis('off')
plt.show()

# image_path = 'result\eg1.jpg'
# chessboard = plt.imread(image_path)

# ========== 2. 傅里叶变换 + 频谱可视化 ==========
f_transform = fft2(chessboard)          # 二维傅里叶变换
f_shifted = fftshift(f_transform)       # 低频移到中心

# 取幅度谱（+1 防止 log(0)，再压缩动态范围）
magnitude_spectrum = np.log(np.abs(f_shifted) + 1)

plt.figure(figsize=(6,6))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frequency Spectrum (Log Scale)')
plt.axis('off')
plt.show()


# ========== 3. 低通滤波（只保留低频 → 模糊） ==========
rows, cols = img_size, img_size
crow, ccol = rows // 2, cols // 2
radius = 30

# 创建圆形低通滤波器
mask_low = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= radius**2:
            mask_low[i, j] = 1

# 应用滤波器
f_filtered_low = f_shifted * mask_low

# 逆变换回时域
f_ishift_low = ifftshift(f_filtered_low)  # ← 这里用了 ifftshift
img_lowpass = np.real(ifft2(f_ishift_low))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_low, cmap='gray')
plt.title('Low-pass Filter Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_lowpass, cmap='gray')
plt.title('After Low-pass Filter (Blurred)')
plt.axis('off')
plt.show()


# ========== 4. 高通滤波（只保留高频 → 边缘） ==========
radius_high = 30
mask_high = np.ones((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= radius_high**2:
            mask_high[i, j] = 0

f_filtered_high = f_shifted * mask_high
f_ishift_high = ifftshift(f_filtered_high)
img_highpass = np.real(ifft2(f_ishift_high))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_high, cmap='gray')
plt.title('High-pass Filter Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_highpass, cmap='gray')
plt.title('After High-pass Filter (Edges only)')
plt.axis('off')
plt.show()


# ========== 5. 只保留某一方向的纹理（竖线） ==========
mask_horizontal = np.zeros((rows, cols))
# 在垂直轴上保留亮斑（离中心 32 像素处）
for offset in [-32, 32]:
    if 0 <= crow + offset < rows:
        mask_horizontal[crow + offset - 2:crow + offset + 3, ccol-2:ccol+3] = 1

f_horizontal = f_shifted * mask_horizontal
img_horizontal = np.real(ifft2(ifftshift(f_horizontal)))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask_horizontal, cmap='gray')
plt.title('Only Keep Vertical Frequencies')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_horizontal, cmap='gray')
plt.title('Reconstructed: Only Vertical Edges')
plt.axis('off')
plt.show()