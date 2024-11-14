import random
import numpy as np
import torch
import copy
import torch.nn.functional as F
from collections import defaultdict
from utils.extract_spectrum import compute_frequency_spectrum, extract_low_freq
from models.client.clientbase import Clientbase

class Clientour(Clientbase):
    def __init__(self, args, id, net, local_ep):
        super().__init__(args, id, net, local_ep)
        self.feature_distributions = None

    def train_convs(self, low_freq_spectrum_g):
        # 初始化标签分布字典，结构为{label: {'mean': [], 'var': [], 'count': int}}
        self.feature_distributions = {}
        
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        spectrums = []
        # self.net.freeze_classifier()  # 冻结分类器
        for epoch in range(self.local_ep):
            batch_loss = []
            T = 2
            Spec_loss = 0
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if len(labels) < self.args.batch_size:
                    # 获取缺少的数据量
                    missing_count = self.args.batch_size - len(labels)

                    # 从已有的样本中随机采样补全
                    indices = random.sample(range(len(self.dataset_train.dataset)), missing_count)
                    sampled_images, sampled_labels = zip(*[self.dataset_train.dataset[i] for i in indices])

                    # 将采样的数据转移到设备并添加到当前 batch 中
                    sampled_images = torch.stack([img.to(self.args.device) for img in sampled_images])
                    sampled_labels = torch.tensor(sampled_labels, device=self.args.device)

                    # 将采样到的图像和标签添加到当前 batch
                    images = torch.cat([images, sampled_images], dim=0)
                    labels = torch.cat([labels, sampled_labels], dim=0)

                self.net.zero_grad()  # 清除梯度


                feature = self.net(images, with_classify=False)
                # 对每个标签更新其特征分布
                for label in torch.unique(labels):
                    # 获取当前标签对应的样本特征
                    indices = (labels == label).nonzero(as_tuple=True)[0]
                    features = feature[indices].detach()

                    # 计算该标签在当前 batch 的均值和方差
                    batch_mean = features.mean(dim=0)
                    batch_var = features.var(dim=0, unbiased=False)
                    batch_count = features.size(0)

                    # 如果标签是新标签，初始化均值和方差
                    if label.item() not in self.feature_distributions:
                        self.feature_distributions[label.item()] = {
                            'mean': [],
                            'var': [],
                            'count': []
                        }

                    # 存储分布统计
                    self.feature_distributions[label.item()]['mean'].append(batch_mean)
                    self.feature_distributions[label.item()]['var'].append(batch_var)
                    self.feature_distributions[label.item()]['count'].append(batch_count)


                output = self.net(images, with_classify=True)

                base_loss = self.loss_func(output, labels)

                spectrum_l = compute_frequency_spectrum(feature)
                # low_freq_spectrum_l = extract_low_freq(spectrum_l, 0.5)

                if low_freq_spectrum_g is not None:
                    # 全局低频谱对本地特征提取器的谱蒸馏
                    l_t = torch.tensor(spectrum_l, dtype=torch.float)
                    g_t = torch.tensor(low_freq_spectrum_g, dtype=torch.float)

                    l_probs = F.log_softmax(l_t / T, dim=0)
                    g_probs = F.softmax(g_t / T, dim=0)

                    Spec_loss = self.loss_func_kl(l_probs, g_probs)

                loss = base_loss + Spec_loss
                loss.backward()
                optimizer.step()

                spectrums.append(copy.deepcopy(spectrum_l))

                if batch_idx % 10 == 0:
                    print('Local Epoch-conv: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(self.dataset_train.dataset),
                              100. * batch_idx / len(self.dataset_train), loss.item()))
                batch_loss.append(loss.item())

        for label, stats in self.feature_distributions.items():
            # 提取该标签下所有 batch 的均值、方差和样本数列表
            mean_list = stats['mean']
            var_list = stats['var']
            count_list = stats['count']

            # 计算总体的样本数
            total_count = sum(count_list)

            # 计算全局均值
            weighted_means = [mean * count for mean, count in zip(mean_list, count_list)]
            global_mean = sum(weighted_means) / total_count

            # 计算全局方差 (基于Welford's方法)
            squared_diffs = [
                var * count + (mean - global_mean).pow(2) * count
                for var, mean, count in zip(var_list, mean_list, count_list)
            ]
            global_var = sum(squared_diffs) / total_count
            # 存储最终结果
            self.feature_distributions[label] = {
                'mean': [global_mean],
                'var': [global_var],
                'count': total_count
            }

        # self.net.unfreeze_classifier()
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(f'localEpochLoss-conv:{sum(epoch_loss) / len(epoch_loss)}')


        low_freq_spectrum_l = extract_low_freq(np.mean(spectrums, axis=0), 0.5)
        # np.save(f'/kaggle/working/client{self.id}_spectrum.npy', np.mean(spectrums, axis=0))
        # np.save(f'/kaggle/working/client{self.id}_lowSpectrum.npy', low_freq_spectrum_l)

        return low_freq_spectrum_l, sum(epoch_loss) / len(epoch_loss)


    def train_fc(self, public_dataset_train, server):
        self.net.train()
        server.net.eval()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        T = 2
        epoch_loss = []
        self.net.freeze_feature_extractor()  # 冻结特征提取器
        for epoch in range(self.args.g_ep):
            batch_loss = []

            for batch_idx, (features, labels) in enumerate(public_dataset_train):
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度

                output = self.net(features, just_classify=True)
                output_g = server.net(features, just_classify=True)


                l_probs = F.log_softmax(output / T, dim=0)
                g_probs = F.softmax(output_g / T, dim=0)

                dis_loss = self.loss_func_kl(l_probs, g_probs)

                base_loss = self.loss_func(output, labels)

                loss = base_loss + dis_loss
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('Local Epoch-fc: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(features), len(public_dataset_train.dataset),
                              100. * batch_idx / len(public_dataset_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.net.unfreeze_feature_extractor()
        print(f'localEpochLoss-fc:{sum(epoch_loss) / len(epoch_loss)}')
        return sum(epoch_loss) / len(epoch_loss)


    def eval(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        # 初始化每个标签的正确个数和总数
        correct_per_label = defaultdict(int)
        total_per_label = defaultdict(int)

        for idx, (data, target) in enumerate(self.dataset_test):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.net(data)
            # 计算损失
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # 获取预测结果
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            # 计算总体正确个数
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            # 统计每个标签的正确个数和总数
            for i in range(len(target)):
                label = target[i].item()
                total_per_label[label] += 1
                if y_pred[i].item() == label:
                    correct_per_label[label] += 1

        test_loss /= len(self.dataset_test.dataset)
        accuracy = 100.00 * correct / len(self.dataset_test.dataset)

        # 输出每个标签的正确率
        for label in sorted(total_per_label.keys()):
            label_accuracy = 100.00 * correct_per_label[label] / total_per_label[label]
            print(
                f'Label {label}: Accuracy: {label_accuracy:.2f}% ({correct_per_label[label]}/{total_per_label[label]})')

        print('\nTest set: Average loss: {:.4f} \n{}_Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, self.args.dataName + str(self.id), correct, len(self.dataset_test.dataset), accuracy))

        return accuracy, test_loss, correct_per_label, total_per_label


    def get_ExFeature_weight(self):
        conv_weights = {}
        for name, param in self.net.named_parameters():
            if 'conv' in name or 'fc1' in name or 'fc2' in name:
                conv_weights[name] = param.data
        return conv_weights

    def get_cla_weight(self):
        cla_weights = {}
        for name, param in self.net.named_parameters():
            if 'fc3' in name:
                cla_weights[name] = param.data
        return cla_weights

    def update_weight_ExFeature(self, conv_weights):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if 'conv' in name or 'fc1' in name or 'fc2' in name:
                    param.data.copy_(conv_weights[name])

    def update_weight_classifier(self, fc_weights):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if 'fc3' in name:
                    param.data.copy_(fc_weights[name])
