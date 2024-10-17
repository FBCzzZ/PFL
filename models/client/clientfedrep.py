import torch
from models.client.clientbase import Clientbase


class Clientfedrep(Clientbase):
    def __init__(self, args, id, net, local_ep):
        super().__init__(args, id, net, local_ep)

    def train_convs(self):
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for epoch in range(self.local_ep):
            batch_loss = []
            self.net.freeze_classifier()  # 冻结分类器
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()  # 清除梯度

                output = self.net(images)
                base_loss = self.loss_func(output, labels)

                loss = base_loss
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

    def train_fc(self):
        self.net.train()
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
                base_loss = self.loss_func(output, labels)

                loss = base_loss
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
