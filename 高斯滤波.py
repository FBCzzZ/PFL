import cv2
import numpy as np
import matplotlib.pyplot as plt
from DataSets.dataLoad import get_dataLoad

import torchvision.transforms as transforms
from PIL import Image

dataDir = './DataSets/data'
batch_size = 64
train_data, test_data = get_dataLoad(dataDir, 'syn', batch_size)

for data in train_data:
    image = data[0][0].numpy()
    break

image = np.transpose(image, (1, 2, 0))
plt.imshow(image)
plt.title('Original Color Image')
plt.show()

# 创建一个3x3的均值滤波器
kernel = np.ones((3, 3), np.float32) / 9

# 应用低通滤波器
filtered_image = cv2.filter2D(image, -1, kernel)

# 显示滤波后的彩色图像
plt.imshow(filtered_image)
plt.title('Low-Pass Filtered Color Image')
plt.show()


# 应用高斯低通滤波器
gaussian_filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

# 显示高斯低通滤波后的彩色图像
plt.imshow(gaussian_filtered_image)
plt.title('Gaussian Low-Pass Filtered Color Image')
plt.show()
