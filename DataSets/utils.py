import torch
from torch.utils.data import Dataset, DataLoader


class SynDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.images, self.labels  = torch.load(file_path)  # 加载.pt文件
        self.transform = transform  # 可选的图像变换

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = image.numpy() if isinstance(image, torch.Tensor) else image
            image = self.transform(image)  # 应用变换
        return image, label
