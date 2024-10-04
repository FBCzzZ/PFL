import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


def non_iid_sampling(args):
    np.random.seed(args.seed)
    Phi = np.random.binomial(1, args.p,
                             size=(args.num_users, args.num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    n_client_per_class = np.sum(Phi, axis=0)

    # 检查是否有客户端没有选择任何类别，或者有类别未被任何客户端选择
    while np.min(n_classes_per_client) == 0 or np.min(n_client_per_class) == 0:
        # 修正客户端没有选择任何类别的情况
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        Phi[invalid_idx] = np.random.binomial(1, args.p, size=(len(invalid_idx), args.num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)

        # 修正类别未被任何客户端选择的情况
        missing_classes = np.where(n_client_per_class == 0)[0]
        for missing_class in missing_classes:
            # 随机选择一个客户端为其分配该类别
            random_client = np.random.choice(args.num_users)
            Phi[random_client, missing_class] = 1

        n_client_per_class = np.sum(Phi, axis=0)

    # 生成每个客户端选择的类别列表
    Psi = [list(np.where(Phi[i, :] == 1)[0]) for i in
           range(args.num_users)]  # indicate the clients that choose each class

    return Psi


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


class MeterDigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录，包含按标签命名的文件夹。
            transform (callable, optional): 可选的转换函数。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历根目录下的每个标签文件夹
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                # 遍历标签文件夹中的每个图像文件
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if img_file.endswith('.png') or img_file.endswith('.jpg'):  # 仅支持 png 和 jpg 文件
                        self.images.append(img_path)
                        self.labels.append(int(label))  # 将标签转为整数

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")

        # 应用转换（如果有的话）
        if self.transform:
            image = self.transform(image)

        return image, label




