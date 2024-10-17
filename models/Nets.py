import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入通道为3，输出通道为6，卷积核大小为3x3，填充1
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)  # 输出尺寸为 (32x32x6)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层：2x2，步长2，输出尺寸为 (16x16x6)

        # 第二个卷积层：输入通道为6，输出通道为16，卷积核大小为3x3，填充1
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)  # 输出尺寸为 (16x16x16)
        # 池化层：2x2，步长2，输出尺寸为 (8x8x16)

        # 全连接层：输入尺寸为8*8*16
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 假设输出类别数为10

    def forward(self, x, with_classify=True):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x1 = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x1))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if with_classify:
            return x
        else:
            return x1

    def freeze_feature_extractor(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.fc2.parameters():
            param.requires_grad = False

    def freeze_classifier(self):

        for param in self.fc3.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        for param in self.conv1.parameters():
            param.requires_grad = True
        for param in self.conv2.parameters():
            param.requires_grad = True

        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True

    def unfreeze_classifier(self):

        for param in self.fc3.parameters():
            param.requires_grad = True
