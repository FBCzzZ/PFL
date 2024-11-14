import numpy as np
import matplotlib.pyplot as plt

# 生成一个示例信号：1Hz和3Hz的正弦波叠加
fs = 100  # 采样频率 (Hz)
t = np.arange(0, 2, 1/fs)  # 2秒的时间轴
signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

# 1. 计算傅里叶变换
signal_fft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=1/fs)  # 获取频率轴

# 2. 只取前一半频率（因为是对称的）
positive_freqs = frequencies[:len(frequencies)//2]
magnitude = np.abs(signal_fft[:len(signal)//2]**2)  # 幅值

# 3. 绘制频谱
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.grid()
plt.show()

if __name__ == '__main__':
    pass