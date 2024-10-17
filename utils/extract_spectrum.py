import numpy as np
import torch
from torch import nn


def compute_frequency_spectrum(model):
    weights = []

    # 如果传入的是权重字典 (self.get_ExFeature_weight 返回)
    if isinstance(model, dict):
        for key, weight in model.items():
            weights.append(weight.cpu().numpy().flatten())  # 展平并转换为 numpy 数组
        weights = np.concatenate(weights)

    # 如果传入的是权重列表
    elif isinstance(model, list):
        weights = torch.cat([w.view(-1) for w in model]).cpu().numpy()  # 展平并拼接

    # 如果传入的是 nn.Module 模型
    elif isinstance(model, nn.Module):
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        weights = np.concatenate(weights)

    else:
        raise TypeError("Input must be a dict, list of weights, or a PyTorch model.")

    # 计算傅里叶变换
    frequency_spectrum = np.fft.fft(weights)
    return frequency_spectrum


def extract_low_freq(spectrum, percent):
    freqs = np.fft.fftfreq(len(spectrum))

    abs_freqs = np.abs(freqs)
    sorted_freqs = np.sort(abs_freqs)
    # 计算阈值位置
    index = int(percent * len(sorted_freqs))-1
    # 确保索引不超出范围
    index = min(index, len(sorted_freqs) - 1)
    threshold = sorted_freqs[index]

    mask = np.abs(freqs) < threshold
    low_freq_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
    low_freq_spectrum[mask] = spectrum[mask]

    # low_freq_spectrum = np.abs(low_freq_spectrum)
    # low_freq_spectrum /= np.sum(low_freq_spectrum)
    return np.abs(low_freq_spectrum)

