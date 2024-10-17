import torch
from torch import nn
import torch.nn.functional as F
from DataSets.utils import DatasetLoader


class Clientbase(object):
    def __init__(self, args, id, net, local_ep):
        self.args = args

        self.net = net
        self.local_ep = local_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_kl = nn.KLDivLoss(reduction="batchmean")
        self.id = id

        dataset = DatasetLoader(args, 'client', id=self.id)
        self.dataset_train = dataset.train_dataset
        self.dataset_test = dataset.test_dataset

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
            test_loss, self.args.dataName+str(self.id), correct, len(self.dataset_test.dataset), accuracy))
        return accuracy, test_loss

    def save(self):
        w = self.net.state_dict()
        torch.save(w, f'/kaggle/working/{self.args.dataName}_{self.id}_model_state_dict.pth')

