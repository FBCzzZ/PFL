import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision


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


class DatasetLoader(Dataset):
    def __init__(self, args, type, id=None):
        self.dataDir = args.data_path
        self.dataName = args.dataName
        self.batch_size = args.batch_size

        if type == 'server':
            self.train_dataset, self.test_dataset = self.get_dataLoad()

        elif type == 'client':
            self.id = id
            self.train_dataset = self.read_client_data(is_train=True)
            self.test_dataset = self.read_client_data(is_train=False)

    def get_dataLoad(self):
        train = None
        test = None
        if self.dataName == 'Cifar10':
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize
            ])

            # 4. 读取仪表数字数据集
            # 加载训练集
            # train_file = f'{self.dataDir}/meterdigits'
            train_file = f'{self.dataDir}/md-dataset/meterdigits'

            train = MeterDigitDataset(train_file, transform=transform)

            transform = transforms.Compose(
                [transforms.ToTensor(), normalize])
            test = torchvision.datasets.CIFAR10(root=self.dataDir +
                                         "/cifar10/Cifar10/rawdata", train=False, download=True, transform=transform)
            # test = torchvision.datasets.CIFAR10(root=self.dataDir +
            #                              "/Cifar10/rawdata", train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train, self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test, self.batch_size, shuffle=False)

        return train_loader, test_loader

    def read_data(self, is_train=True):
        dataPath = f'{self.dataDir}/cifar10-dir0-1'
        # dataPath = f'{self.dataDir}'
        if is_train:
            train_data_dir = os.path.join(dataPath, self.dataName, 'train/')

            train_file = train_data_dir + str(self.id) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            return train_data

        else:
            test_data_dir = os.path.join(dataPath, self.dataName, 'test/')

            test_file = test_data_dir + str(self.id) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()

            return test_data

    def read_client_data(self, is_train):
        data = self.read_data(is_train)
        X = torch.Tensor(data['x']).type(torch.float32)
        Y = torch.Tensor(data['y']).type(torch.int64)

        data = [(x, y) for x, y in zip(X, Y)]

        return torch.utils.data.DataLoader(data, self.batch_size, drop_last=False, shuffle=False)

