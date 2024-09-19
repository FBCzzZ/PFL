import torch
import copy
import torch.nn.functional as F
from models.client_fedavg import Client
from models.Nets import CNN
from options import args_parser


data_list = ['mnist', 'usps', 'svhn', 'syn']
local_ep_list = [10, 10, 3, 3]
client_list = []


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    net = CNN()
    for i in range(args.num_users):
        client_list.append(Client(args, data_list[i], copy.deepcopy(net).to(args.device), local_ep_list[i]))
        weight = torch.load(f'./save/{client_list[i].dataName}_model_state_dict.pth')
        client_list[i].update_weight(weight)

    dataset_test = [client_list[i].dataset_test for i in range(args.num_users)]
    dataset_test_len = [len(dataset.dataset) for dataset in dataset_test]

    for i in range(args.num_users):
        net = client_list[i].net
        net.eval()
        for j in range(args.num_users):
            # testing
            test_loss = 0
            correct = 0
            for idx, (data, target) in enumerate(dataset_test[j]):
                if args.gpu != -1:
                    data, target = data.cuda(), target.cuda()
                log_probs = net(data)
                # sum up batch loss
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            test_loss /= dataset_test_len[j]
            accuracy = 100.00 * correct / dataset_test_len[j]
            print('\nTest set: Average loss: {:.4f} \n{}Model_{}_Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, data_list[i], data_list[j], correct, dataset_test_len[j], accuracy))