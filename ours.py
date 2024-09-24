import copy

import numpy as np
import torch

from options import args_parser
from models.Nets import CNN
from models.client import Client
import torch.nn.functional as F
from utils.aggregation import aggregation_spec
from DataSets.utils import non_iid_sampling
from models.utils import aggregation_avg



data_list = ['mnist', 'usps', 'svhn', 'syn']
local_ep_list = [5, 15, 5, 2]
client_list = []
w_local_list = []
w_FFT_list = []
w_FFT_glob = None
net_glob = None
client_class = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model
    if args.model == 'cnn':
        net_glob = CNN().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    if args.noIid:
        client_class = non_iid_sampling(args)

    # 初始化客户端
    for i in range(args.num_users):
        client_list.append(Client(args, data_list[i], copy.deepcopy(net_glob).to(args.device), local_ep_list[i], client_class[i]))
        print(f"client{i},数据集{data_list[i]}:{client_class[i]}")
        w_local_list.append(w_glob)
        # client_list[i].update_weight(w_glob)
        w_FFT_list.append(None)


    for iter in range(args.epochs):
        for i in range(args.num_users):
            client_list[i].update_weight(w_glob)

            # 每个客户端进行本地训练
            loss_locals = []
            w, w_fft, loss = client_list[i].train(w_FFT_glob, net_glob)
            w_FFT_list[i] = copy.deepcopy(w_fft)
            w_local_list[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            client_list[i].eval()

        # update global weights
        w_FFT_glob = aggregation_spec(w_FFT_list)

        # update global weights
        w_glob = aggregation_avg(w_local_list)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        for i in range(args.num_users):
            net_glob.eval()
            # testing
            test_loss = 0
            correct = 0
            dataset_test = client_list[i].dataset_test
            dataset_test_len = len(dataset_test.dataset)
            for idx, (data, target) in enumerate(dataset_test):
                if args.gpu != -1:
                    data, target = data.cuda(), target.cuda()
                log_probs = net_glob(data)
                # sum up batch loss
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            test_loss /= dataset_test_len
            accuracy = 100.00 * correct / dataset_test_len
            print('\nTest set: Average loss: {:.4f} \n{}_glob_Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, client_list[i].dataName, correct, dataset_test_len, accuracy))

    # 保存模型
    for i in range(args.num_users):
        client_list[i].save()

