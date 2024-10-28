import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mode

from utils.extract_spectrum import compute_frequency_spectrum, extract_low_freq
from models.client.clientbase import Clientbase

class Clientour(Clientbase):
    def __init__(self, args, id, net, local_ep):
        super().__init__(args, id, net, local_ep)
        self.E = 0 # 特征分布期望
        self.V = 0 # 特征分布方差
        self.m = 0.5 # 平滑因子

    def train_convs(self, server):
        self.net.train()
        server.net.eval()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for epoch in range(self.local_ep):
            batch_loss = []
            T = 2
            self.net.freeze_classifier()  # 冻结分类器
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度


                feature = self.net(images, with_classify=False)
                self.E = self.E * self.m + (1-self.m) * np.mean(feature.detach().cpu().numpy(), axis=0)
                self.V = self.V * self.m + (1-self.m) * np.var(feature.detach().cpu().numpy(), axis=0)

                output = self.net(images, with_classify=True)
                feature_g = server.net(images, with_classify=False)

                base_loss = self.loss_func(output, labels)

                spectrum_l = compute_frequency_spectrum(feature)
                # low_freq_spectrum_l = extract_low_freq(spectrum_l, 0.5)

                spectrum_g = compute_frequency_spectrum(feature_g)
                low_freq_spectrum_g = extract_low_freq(spectrum_g, 0.5)


                # 全局低频谱对本地特征提取器的谱蒸馏
                l_t = torch.tensor(spectrum_l, dtype=torch.float)
                g_t = torch.tensor(low_freq_spectrum_g, dtype=torch.float)

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
        return self.get_ExFeature_weight(), sum(epoch_loss) / len(epoch_loss)


    def train_fc(self, server, clients):
        self.net.train()
        server.net.eval()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for epoch in range(self.local_ep):
            batch_loss = []
            self.net.freeze_feature_extractor()  # 冻结特征提取器
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度
                output = self.net(images)

                generate_feature = server.generate_feature(labels.shape[0])
                generate_feature = torch.tensor(generate_feature, dtype=torch.float32).to(self.args.device)

                generate_labels = []
                for i in range(self.args.num_users):
                    clients[i].net.to(self.args.device)  # 确保模型在正确的设备上
                    clients[i].net.eval()  # 设置模型为评估模式
                    with torch.no_grad():  # 禁用梯度计算
                        logits = clients[i].net(generate_feature, just_classify = True)  # 获取模型输出
                        probabilities = F.softmax(logits, dim=1)  # 计算概率分布
                        predicted_labels = torch.argmax(probabilities, dim=1)  # 获取预测标签
                        generate_labels.append(predicted_labels)

                # 将生成的标签转换为 NumPy 数组以便计算众数
                generate_labels = torch.stack(generate_labels).cpu().numpy()

                # 计算每列的众数
                generate_labels = mode(generate_labels, axis=0).mode

                generate_output = self.net(generate_feature, just_classify = True)
                generate_labels = torch.tensor(generate_labels, dtype=torch.long, device=self.args.device)

                base_loss = self.loss_func(output, labels)

                generate_loss = self.loss_func(generate_output, generate_labels)

                loss = base_loss + generate_loss
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
        return self.get_cla_weight(), sum(epoch_loss) / len(epoch_loss)

    def get_ExFeature_weight(self):
        conv_weights = {}
        for name, param in self.net.named_parameters():
            if 'conv' in name:
                conv_weights[name] = param.data
        return conv_weights

    def get_cla_weight(self):
        cla_weights = {}
        for name, param in self.net.named_parameters():
            if 'fc' in name:
                cla_weights[name] = param.data
        return cla_weights

    def update_weight_ExFeature(self, conv_weights):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if 'conv' in name:
                    param.data.copy_(conv_weights[name])

    def update_weight_classifier(self, fc_weights):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if 'fc' in name:
                    param.data.copy_(fc_weights[name])
