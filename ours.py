import copy
import torch
from options import args_parser
from models.client import Client
from models.server import Server
import torch.nn.functional as F
from DataSets.utils import non_iid_sampling


data_list = ['mnist', 'usps', 'svhn', 'syn']
local_ep_list = [5, 15, 5, 2]
client_list = []
w_local_list = []
convs_FFT_list = []
convs_FFT_glob = None
client_class = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化服务器
    server = Server(args)

    # copy weights
    w_glob = server.get_weight()

    if args.noIid:
        client_class = non_iid_sampling(args)

    # 初始化客户端
    for i in range(args.num_users):
        client_list.append(Client(args, data_list[i], copy.deepcopy(server.net).to(args.device), local_ep_list[i], client_class[i]))
        print(f"client{i},数据集{data_list[i]}:{client_class[i]}")
        w_local_list.append(w_glob)
        convs_FFT_list.append(None)

    for epoch in range(args.epochs):
        for i in range(args.num_users):
            # client_list[i].update_weight_conv(server.get_conv_weight())
            # 冻结分类器，每个客户端进行特征提取器本地训练
            loss_locals = []
            w, convs_FFT, loss = client_list[i].train_convs(convs_FFT_glob)

            convs_FFT_list[i] = copy.deepcopy(convs_FFT)
            w_local_list[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        # 聚合低频频谱
        convs_FFT_glob = server.agg_spec(convs_FFT_list)

        # 聚合特征提取器
        w_avg = server.agg(w_local_list)
        server.update_weight(w_avg)

        for i in range(args.num_users):
            # 下发特征提取器
            client_list[i].update_weight_conv(server.get_conv_weight())
            # 训练分类器
            w, loss = client_list[i].train_fc(server.net)

            client_list[i].eval()

        # 聚合分类器
        w_avg = server.agg(w_local_list)
        server.update_weight(w_avg)

        for i in range(args.num_users):
            server.net.eval()
            # testing
            test_loss = 0
            correct = 0
            dataset_test = client_list[i].dataset_test
            dataset_test_len = len(dataset_test.dataset)
            for idx, (data, target) in enumerate(dataset_test):
                if args.gpu != -1:
                    data, target = data.cuda(), target.cuda()
                log_probs = server.net(data)
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

