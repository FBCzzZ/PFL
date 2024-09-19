import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, USPS, SVHN
from DataSets.utils import SynDataset
import matplotlib.pyplot as plt


def repeat_channels(x):
    return x.repeat(3, 1, 1)


def get_dataLoad(dataDir, dataName, batch_size):
    train_loader, test_loader = None, None
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
        mnist_train = MNIST(root=f'{dataDir}/mnist-dataset/mnist/', train=True, download=True, transform=transform)
        mnist_test = MNIST(root=f'{dataDir}/mnist-dataset/mnist/', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

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
        usps_train = USPS(root=f'{dataDir}/usps-datas/usps/', train=True, download=True, transform=transform)
        usps_test = USPS(root=f'{dataDir}/usps-datas/usps/', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(usps_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(usps_test, batch_size=batch_size, shuffle=False)

    elif dataName == 'svhn':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        # 3. 读取SVHN数据集
        svhn_train = SVHN(root=f'{dataDir}/svhn-dataset/svhn/', split='train', download=True, transform=transform)
        svhn_test = SVHN(root=f'{dataDir}/svhn-dataset/svhn/', split='test', download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=batch_size, shuffle=False)

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
        train_file = f'{dataDir}/syn-dataset/syn/processed/synth_train.pt'
        test_file = f'{dataDir}/syn-dataset/syn/processed/synth_test.pt'

        syn_train = SynDataset(train_file, transform=transform)
        syn_test = SynDataset(test_file, transform=transform)

        train_loader = torch.utils.data.DataLoader(syn_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(syn_test, batch_size=batch_size, shuffle=False)

    else:
        raise ValueError("Unsupported dataset: {}".format(dataName))
    return train_loader, test_loader


if __name__ == '__main__':
    data = get_dataLoad('mnist')
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