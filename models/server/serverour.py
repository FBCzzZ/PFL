import numpy as np
import torch
from models.server.serverbase import Serverbase


class Serverour(Serverbase):
    def __init__(self, args):
        super().__init__(args)
        self.global_feature_distributions = {}
        # 动量系数
        self.momentum = 0.3
        # 学习率
        self.lr = 0.01


    def train_fc(self):
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # 生成伪特征
        pseudo_features = self.generate_feature()

        # 将伪特征和标签转换为训练数据
        X = torch.cat(list(pseudo_features.values()), dim=0)  # 所有伪特征
        y = torch.cat([torch.full((features.size(0),), label) for label, features in pseudo_features.items()])  # 标签


        # 创建数据集和数据加载器
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        epoch_loss = []
        self.net.freeze_feature_extractor()  # 冻结特征提取器
        for epoch in range(self.args.g_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度
                output = self.net(images, just_classify=True)

                loss = self.loss_func(output, labels)

                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('glob Epoch-fc: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(dataset),
                               100. * batch_idx / len(dataset), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.net.unfreeze_feature_extractor()
        print(f'globEpochLoss-fc:{sum(epoch_loss) / len(epoch_loss)}')
        return self.get_cla_weight(), sum(epoch_loss) / len(epoch_loss)


    def generate_feature(self):
        # 定义每个标签生成的伪特征数量
        num_samples_per_label = 1000

        # 用于存储生成的伪特征
        pseudo_features = {label: [] for label in self.global_feature_distributions.keys()}

        # 生成伪特征
        for label, stats in self.global_feature_distributions.items():
            mean = stats['mean']
            var = stats['var']
            std_dev = torch.sqrt(var+ 1e-6)  # 标准差为方差的平方根

            # 逐个生成伪特征
            for _ in range(num_samples_per_label):
                sample = torch.normal(mean=mean, std=std_dev).clone()
                pseudo_features[label].append(sample.unsqueeze(0))  # 添加一个维度以保持一致性

         # 将每个标签的伪特征转换为张量并合并
        for label in pseudo_features:
            pseudo_features[label] = torch.cat(pseudo_features[label], dim=0)  # 合并成一个张量

        return pseudo_features


    def update_distributions(self, client_distributions):
        # 对每个标签的特征分布进行动量更新
        for label, client_stats in client_distributions.items():
            client_mean = client_stats['mean'][0]
            client_var = client_stats['var'][0]

            # 初始化标签统计量
            if label not in self.global_feature_distributions:
                self.global_feature_distributions[label] = {
                    'mean': client_mean,
                    'var': client_var
                }

                # 使用动量进行更新
                self.global_feature_distributions[label]['mean'] = (
                        self.momentum * self.global_feature_distributions[label]['mean'] + (1 - self.momentum) * client_mean
                )
                self.global_feature_distributions[label]['var'] = (
                        self.momentum * self.global_feature_distributions[label]['var'] + (1 - self.momentum) * client_var
                )


    def agg_spec(self, w_FFT_list):
        w_FFT_np = np.vstack(w_FFT_list)
        w_FFT_glob = np.mean(w_FFT_np, axis=0)
        return w_FFT_glob


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

