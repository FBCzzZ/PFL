import torch
import copy
import numpy as np
from torch import nn
import torch.nn.functional as F
from DataSets.dataLoad import DatasetLoader
from utils.extract_spectrum import compute_frequency_spectrum, extract_low_freq


class Client(object):
    def __init__(self, args, dataName, net, local_ep, client_class):
        self.args = args
        imbalance_dataset = DatasetLoader(args, dataName, client_class)

        self.dataset_train = torch.utils.data.DataLoader(imbalance_dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.dataset_test = torch.utils.data.DataLoader(imbalance_dataset.test_dataset, batch_size=args.batch_size, shuffle=False)

        self.net = net
        self.local_ep = local_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_kl = nn.KLDivLoss(reduction="batchmean")
        self.dataName = dataName


    def train_convs(self, low_freq_spectrum_glob):
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for epoch in range(self.local_ep):
            batch_loss = []
            Spec_loss = 0
            T = 2
            self.net.freeze_classifier()  # 冻结分类器
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度


                output = self.net(images)
                base_loss = self.loss_func(output, labels)

                spectrum = compute_frequency_spectrum(self.get_ExFeature_weight())
                low_freq_spectrum = extract_low_freq(spectrum, 0.5)

                if low_freq_spectrum_glob is not None:
                    # 全局低频谱对本地特征提取器的谱蒸馏

                    l_t = torch.tensor(low_freq_spectrum, dtype=torch.float)
                    g_t = torch.tensor(low_freq_spectrum_glob, dtype=torch.float)

                    l_probs = F.log_softmax(l_t / T, dim=0)
                    g_probs = F.softmax(g_t / T, dim=0)

                    Spec_loss = self.loss_func_kl(l_probs, g_probs)


                loss = base_loss + Spec_loss
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Local Epoch-conv: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(self.dataset_train.dataset),
                              100. * batch_idx / len(self.dataset_train), loss.item()))
                batch_loss.append(loss.item())

            self.net.unfreeze_classifier()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(f'localEpochLoss-conv:{sum(epoch_loss) / len(epoch_loss)}')
        return self.net.state_dict(), low_freq_spectrum, sum(epoch_loss) / len(epoch_loss)

    def train_fc(self, net_glob):
        self.net.train()
        net_glob.eval()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for epoch in range(self.local_ep):
            batch_loss = []
            T = 2
            self.net.freeze_feature_extractor()  # 冻结特征提取器
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度
                output = self.net(images)
                base_loss = self.loss_func(output, labels)

                # 全局模型对本地模型的知识蒸馏
                g_output = net_glob(images)
                l_probs = F.log_softmax(output/T, dim=0)
                g_probs = F.softmax(g_output/T, dim=0)
                dist_loss = self.loss_func_kl(l_probs, g_probs)

                loss = base_loss + dist_loss
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('Local Epoch-fc: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(self.dataset_train.dataset),
                              100. * batch_idx / len(self.dataset_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.net.unfreeze_feature_extractor()
        print(f'localEpochLoss-fc:{sum(epoch_loss) / len(epoch_loss)}')
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def eval(self):
        self.net.eval()
        # testing
        test_loss = 0
        correct = 0
        for idx, (data, target) in enumerate(self.dataset_test):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(self.dataset_test.dataset)
        accuracy = 100.00 * correct / len(self.dataset_test.dataset)
        print('\nTest set: Average loss: {:.4f} \n{}_Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, self.dataName, correct, len(self.dataset_test.dataset), accuracy))
        return accuracy, test_loss

    def get_ExFeature_weight(self):
        # 提取卷积层的权重
        conv_weights = []
        for name, param in self.net.named_parameters():
            if 'conv' in name or 'fc1' in name:  # 筛选出卷积层权重
                conv_weights.append(param.data)
        return conv_weights

    def update_weight_ExFeature(self, conv_weights):
        with torch.no_grad():  # 不计算梯度
            for idx, (name, param) in enumerate(self.net.named_parameters()):
                if 'conv' in name or 'fc1' in name:  # 筛选出卷积层权重
                    param.data.copy_(conv_weights[idx])

    def update_weight_classifier(self, fc_weights):
        with torch.no_grad():  # 不计算梯度
            for idx, (name, param) in enumerate(self.net.named_parameters()):
                if 'fc' in name:  # 筛选出分类层权重
                    param.data.copy_(fc_weights[idx])  # 根据索引加载权重

    def save(self):
        w = self.net.state_dict()
        # torch.save(w, f'{self.dataName}_model_state_dict.pth')
        torch.save(w, f'/kaggle/working/{self.dataName}_model_state_dict.pth')