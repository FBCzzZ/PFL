import torch
import copy
import numpy as np
from torch import nn
import torch.nn.functional as F
from DataSets.dataLoad import IMBALANCEDataset
from utils.FFT import spectral_cal


class Client(object):
    def __init__(self, args, dataName, net, local_ep, client_class):
        self.args = args
        imbalance_dataset = IMBALANCEDataset(args, dataName, client_class)

        self.dataset_train = torch.utils.data.DataLoader(imbalance_dataset.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.dataset_test = torch.utils.data.DataLoader(imbalance_dataset.test_dataset, batch_size=args.batch_size, shuffle=False)

        self.net = net
        self.local_ep = local_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func_kl = nn.KLDivLoss(reduction="batchmean")
        self.dataName = dataName

    def train(self, w_FFT_glob):
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.local_ep):
            batch_loss = []
            Spec_loss = 0
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()

                log_probs = self.net(images)
                base_loss = self.loss_func(log_probs, labels)
                w_FFT = spectral_cal(self.net)

                if w_FFT_glob is not None:
                    w_FFT = torch.tensor(w_FFT, dtype=torch.float)
                    w_FFT_glob = torch.tensor(w_FFT_glob, dtype=torch.float)
                    Spec_loss = self.loss_func_kl(w_FFT, w_FFT_glob)

                loss = base_loss + Spec_loss
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Local Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.dataset_train.dataset),
                              100. * batch_idx / len(self.dataset_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(f'localEpochLoss:{sum(epoch_loss) / len(epoch_loss)}')
        return w_FFT, sum(epoch_loss) / len(epoch_loss)

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

    def update_weight(self, weights):
        self.net.load_state_dict(weights)

    def save(self):
        w = self.net.state_dict()
        torch.save(w, f'{self.dataName}_model_state_dict.pth')
