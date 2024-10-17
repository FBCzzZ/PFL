import torch
from torch import nn
from models.Nets import CNN
import torch.nn.functional as F
from DataSets.utils import DatasetLoader


class Serverbase(object):
    def __init__(self, args):
        self.args = args
        if args.model == 'cnn':
            self.net = CNN().to(args.device)
        else:
            exit('Error: unrecognized model')
        print(self.net)
        dataset = DatasetLoader(args, 'server')
        self.dataset_train = dataset.train_dataset
        self.dataset_test = dataset.test_dataset

        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_kl = nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def average_weights_dict(weights_dict_list):
        """
        对多个客户端的特征提取层权重字典进行平均聚合
        :param weights_dict_list: 包含多个客户端权重字典的列表，每个字典键为层名称，值为张量
        :return: 聚合后的平均权重字典
        """
        # 初始化一个空字典用于存储平均后的权重
        avg_weights = {}

        # 遍历第一个字典的键
        for key in weights_dict_list[0].keys():
            # 收集所有客户端对应键的权重
            layer_weights = torch.stack([client_weights[key] for client_weights in weights_dict_list], dim=0)
            # 对该键的权重进行平均
            avg_weights[key] = torch.mean(layer_weights, dim=0)

        return avg_weights


    def get_weight(self):
        return self.net.state_dict()

    def update_weight(self, weight):
        self.net.load_state_dict(weight)

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
            test_loss, self.args.dataName, correct, len(self.dataset_test.dataset), accuracy))
        return accuracy, test_loss

