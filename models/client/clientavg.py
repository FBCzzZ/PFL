import torch
from models.client.clientbase import Clientbase


class Clientavg(Clientbase):
    def __init__(self, args, id, net, local_ep):
        super().__init__(args, id, net, local_ep)

    def train(self):
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataset_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Local Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.dataset_train.dataset),
                              100. * batch_idx / len(self.dataset_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(f'localEpochLoss:{sum(epoch_loss) / len(epoch_loss)}')
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weight(self, weights):
        self.net.load_state_dict(weights)
