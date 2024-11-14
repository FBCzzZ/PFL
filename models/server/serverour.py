import numpy as np
import torch
import random
from models.server.serverbase import Serverbase


class Serverour(Serverbase):
    def __init__(self, args):
        super().__init__(args)
        self.global_feature_distributions = {}
        global_model_weights = self.get_cla_weight()
        self.momentum = {key: torch.zeros_like(val) for key, val in global_model_weights.items()}
        # 动量系数
        self.beta = 0.9

        # 平滑系数
        self.alpha = 0.3
        # 学习率
        self.lr = 0.01


    def train_fc(self, global_correct_per_label, global_total_per_label):
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # 生成伪特征
        pseudo_features = self.generate_feature(global_correct_per_label, global_total_per_label)

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
        return sum(epoch_loss) / len(epoch_loss)


    def generate_feature(self, global_correct_per_label, global_total_per_label):
        # 定义每个标签生成的伪特征数量
        base_sample_size = 1000

        samples_per_label = {}
        # 计算全局标签正确率
        for label in global_total_per_label:
            if global_total_per_label[label] > 0:
                accuracy = global_correct_per_label[label] / global_total_per_label[label]
            else:
                accuracy = 0.0

            samples_per_label[label] = int(base_sample_size * (1 - accuracy) * random.uniform(0.8, 1.2)) + 1


        # 用于存储生成的伪特征
        pseudo_features = {label: [] for label in self.global_feature_distributions.keys()}

        # 生成伪特征
        for label, stats in self.global_feature_distributions.items():
            mean = stats['mean']
            var = stats['var']
            std_dev = torch.sqrt(var+ 1e-6)  # 标准差为方差的平方根

            # 逐个生成伪特征
            for _ in range(samples_per_label[label]):
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
                        self.alpha * self.global_feature_distributions[label]['mean'] + (1 - self.alpha) * client_mean
                )
                self.global_feature_distributions[label]['var'] = (
                        self.alpha * self.global_feature_distributions[label]['var'] + (1 - self.alpha) * client_var
                )


    def agg_spec(self, w_FFT_list):
        w_FFT_np = np.vstack(w_FFT_list)
        w_FFT_glob = np.mean(w_FFT_np, axis=0)
        return w_FFT_glob


    def agg_cla_mom(self, cla_w_local):
        avg_weights = self.average_weights_dict(cla_w_local)
        global_weights = self.get_cla_weight()
        # 计算动量并更新全局模型
        for key in global_weights.keys():
            # 动量更新
            self.momentum[key] = self.beta * self.momentum[key] + (avg_weights[key] - global_weights[key])
            # 用动量更新全局模型
            global_weights[key] += self.lr * self.momentum[key]
        self.update_weight_classifier(global_weights)


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

