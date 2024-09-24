import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import MNIST, USPS, SVHN
from DataSets.utils import SynDataset
from torch.utils.data import Dataset

def repeat_channels(x):
    return x.repeat(3, 1, 1)


class IMBALANCEDataset(Dataset):
    def __init__(self, args, dataName, selected_classes=None):
        self.selected_classes = selected_classes
        self.train_dataset, self.test_dataset = self.get_dataLoad(dataDir=args.data_path, dataName=dataName)
        self.cls_num = len(selected_classes)

        # 如果有指定的类别，则进行筛选
        if self.selected_classes:
            self.train_dataset = self.filter_classes(self.train_dataset, self.selected_classes)

        np.random.seed(args.seed)
        img_num_list = self.get_img_num_per_cls(args.imb_type, args.imb_factor)
        print(f"数据集{dataName}类别分布：{img_num_list}")
        imbl_data, imbl_target = self.gen_imbalanced_data(img_num_list)

        # 使用生成的不平衡数据更新训练集
        self.train_dataset = torch.utils.data.TensorDataset(torch.tensor(imbl_data, dtype=torch.float32),
                                                            torch.tensor(imbl_target, dtype=torch.long))

    def filter_classes(self, dataset, classes_to_keep):
        """
        筛选数据集中只包含指定类别的数据
        """
        filtered_data = []
        filtered_targets = []

        for idx in range(len(dataset)):
            img, label = dataset[idx]  # 通过索引获取图像和标签
            if label in classes_to_keep:
                filtered_data.append(img)
                filtered_targets.append(label)

        return torch.utils.data.TensorDataset(
            torch.stack(filtered_data),  # 将图像堆叠成一个张量
            torch.tensor(filtered_targets, dtype=torch.long)  # 转换标签为张量
        )


    def get_img_num_per_cls(self, imb_type, imb_factor):
        """
        根据不平衡类型和不平衡因子，生成每个类别的样本数量
        """
        img_max = len(self.train_dataset) / self.cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (imb_factor ** (cls_idx / (self.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2, self.cls_num):
                img_num_per_cls.append(int(img_max * imb_factor))

        elif imb_type == 'random':
            for cls_idx in range(self.cls_num):
                num = img_max * np.random.uniform(0.1, 1.0)  # 在50%到100%范围内随机选择
                img_num_per_cls.append(int(num))

        else:
            img_num_per_cls.extend([int(img_max)] * self.cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        """
        根据每个类别的样本数量，生成不平衡数据集
        """
        new_data = []
        new_targets = []

        targets = np.array(self.train_dataset.tensors[1].numpy(), dtype=np.int64)
        data = np.array(self.train_dataset.tensors[0].numpy())
        classes = np.unique(targets)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)

        # 将生成的不平衡数据重新赋值
        return np.vstack(new_data), new_targets


    def get_cls_num_list(self):
        """
        获取每个类别的样本数量
        """
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_dataLoad(self, dataDir, dataName):
        if dataName == 'mnist':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(repeat_channels),
                normalize
            ])

            # 1. 读取MNIST数据集
            train = MNIST(root=f'{dataDir}/mnist/', train=True, download=True, transform=transform)
            test = MNIST(root=f'{dataDir}/mnist/', train=False, download=True, transform=transform)
            # train = MNIST(root=f'{dataDir}/mnist-dataset/mnist/', train=True, download=True, transform=transform)
            # test = MNIST(root=f'{dataDir}/mnist-dataset/mnist/', train=False, download=True, transform=transform)

        elif dataName == 'usps':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(repeat_channels),
                normalize
            ])

            # 2. 读取USPS数据集
            train = USPS(root=f'{dataDir}/usps/', train=True, download=True, transform=transform)
            test = USPS(root=f'{dataDir}/usps/', train=False, download=True, transform=transform)
            # train = USPS(root=f'{dataDir}/usps-datas/usps/', train=True, download=True, transform=transform)
            # test = USPS(root=f'{dataDir}/usps-datas/usps/', train=False, download=True, transform=transform)


        elif dataName == 'svhn':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize
            ])

            # 3. 读取SVHN数据集
            train = SVHN(root=f'{dataDir}/svhn/', split='train', download=True, transform=transform)
            test = SVHN(root=f'{dataDir}/svhn/', split='test', download=True, transform=transform)
            # train = SVHN(root=f'{dataDir}/svhn-dataset/svhn/', split='train', download=True, transform=transform)
            # test = SVHN(root=f'{dataDir}/svhn-dataset/svhn/', split='test', download=True, transform=transform)

        elif dataName == 'syn':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                normalize
            ])

            # 4. 读取SYN数据集
            # 加载训练集和测试集
            train_file = f'{dataDir}/syn/processed/synth_train.pt'
            test_file = f'{dataDir}/syn/processed/synth_test.pt'
            # train_file = f'{dataDir}/syn-dataset/syn/processed/synth_train.pt'
            # test_file = f'{dataDir}/syn-dataset/syn/processed/synth_test.pt'

            train = SynDataset(train_file, transform=transform)
            test = SynDataset(test_file, transform=transform)

        else:
            raise ValueError("Unsupported dataset: {}".format(dataName))

        # train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

        return train, test


if __name__ == '__main__':
    # data = get_dataLoad('mnist')
    pass
    # 现在你可以使用PyTorch的DataLoader来加载数据
    # train_loader_mnist = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
    # train_loader_usps = torch.utils.data.DataLoader(usps_train, batch_size=64, shuffle=True)
    # train_loader_svhn = torch.utils.data.DataLoader(svhn_train, batch_size=64, shuffle=True)
    # train_loader_syn = torch.utils.data.DataLoader(syn_train, batch_size=64, shuffle=True)


    # for image, label in syn_train:
    #     # 显示图像
    #     if image.shape[0] == 3:
    #         image = image.permute(1, 2, 0)
    #     # plt.imshow(image.squeeze(), cmap='gray')
    #     plt.imshow(image)
    #     plt.title(f"Label: {label}")
    #     plt.show()
    #     break