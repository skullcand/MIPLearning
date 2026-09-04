import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 第1步：生成一个简单的信号 ============
# 信号 = 低频波(2Hz) + 高频波(10Hz)
fs = 50              # 采样频率：每秒采50个点
t = np.linspace(0, 2, 100)  # 时间：0到2秒，共100个点

# 构造信号：两个正弦波叠加
signal = 3 * np.sin(2 * np.pi * 2 * t) + 1.5 * np.sin(2 * np.pi * 10 * t)
#         振幅3   频率2Hz          振幅1.5  频率10Hz

# ============ 第2步：傅里叶变换 ============
fft_result = fft(signal)           # 做FFT
freq = fftfreq(len(signal), 1/fs)  # 计算对应的频率
magnitude = np.abs(fft_result)     # 幅度（复数取模）

# ============ 第3步：只取正频率部分 ============
half = len(signal) // 2
freq_positive = freq[:half]
mag_positive = magnitude[:half] * 2 / len(signal)  # 归一化

# ============ 第4步：画图 ============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# 左图：原始信号（时域）
ax1.plot(t, signal, 'b-', linewidth=2)
ax1.set_xlabel('Time (秒)')
ax1.set_ylabel('amplitude')
ax1.set_title('origin signal：2Hz + 10Hz ')
ax1.grid(True, alpha=0.3)

# 在图上标注频率成分
ax1.text(0.5, 3.5, '2Hz  (amplitude3)', color='red', fontsize=11)
ax1.text(0.5, -3, '10Hz  (amplitude1.5)', color='green', fontsize=11)

# 右图：频谱（频域）
ax2.stem(freq_positive, mag_positive, basefmt=' ', linefmt='r-', markerfmt='ro')
ax2.set_xlabel('frequency (Hz)')
ax2.set_ylabel('magnitufr')
ax2.set_title('After FFT  amplitude')
ax2.grid(True, alpha=0.3)

# 标注峰值
for f, m in zip(freq_positive, mag_positive):
    if m > 0.1:  # 只显示明显的峰值
        ax2.annotate(f'{f}Hz', xy=(f, m), xytext=(f, m+0.3),
                    ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# ============ 第5步：打印结果 ============
print("=" * 50)
print("Result of FFT ")
print("=" * 50)
print(f"signal time: 2秒")
print(f"sample frequency: {fs} Hz")
print(f"size of sample: {len(signal)} 个")
print("\npercent of sample frequency:")
for f, m in zip(freq_positive, mag_positive):
    if m > 0.1:
        print(f"  ✓ frequency: {f:.1f} Hz, magnitude: {m:.2f}")
print("=" * 50)