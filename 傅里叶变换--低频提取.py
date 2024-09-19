import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift


def compute_fft(image):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    return magnitude_spectrum


# Example for one MNIST image
from torchvision import datasets, transforms

mnist_data = datasets.MNIST(root='../../data0/federated_learning/mnist', train=True, download=True, transform=transforms.ToTensor())
image, label = mnist_data[0]
image = image.squeeze().numpy()  # Convert to 2D array

magnitude_spectrum = compute_fft(image)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title(f'Frequency Spectrum of MNIST Image - Label: {label}')
plt.show()


def apply_filter(f_transform_shifted, low_freq=True):
    rows, cols = f_transform_shifted.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(f_transform_shifted)

    if low_freq:
        mask[crow - 5:crow + 5, ccol - 5:ccol + 5] = 1  # Keep low frequencies
    else:
        mask = 1 - mask  # Keep high frequencies

    return f_transform_shifted * mask


filtered_fft = apply_filter(fftshift(fft2(image)), low_freq=True)
filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Low Frequencies)')
plt.show()
