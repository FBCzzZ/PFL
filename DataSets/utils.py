import torch
from torch.utils.data import Dataset
import numpy as np


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


