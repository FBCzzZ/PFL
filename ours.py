import copy
import torch
from options import args_parser
from models.client import Client
from models.server import Server
import torch.nn.functional as F
from DataSets.utils import non_iid_sampling


data_list = ['mnist', 'usps', 'svhn', 'syn']
server_data = 'md'
local_ep_list = [5, 15, 5, 2]
glob_ep = 10
client_list = []
w_local_list = []
low_freq_spectrum_list = []
low_freq_spectrum_glob = None
client_class = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化服务器
    server = Server(args, server_data)

    # copy weights
    w_glob = server.get_weight()

    if args.noIid:
        client_class = non_iid_sampling(args)

    # 初始化客户端
    for i in range(args.num_users):
        client_list.append(Client(args, data_list[i], copy.deepcopy(server.net).to(args.device), local_ep_list[i], client_class[i]))
        print(f"client{i},数据集{data_list[i]}:{client_class[i]}")
        w_local_list.append(w_glob)
        low_freq_spectrum_list.append(None)

    for c in range(args.epochs):
        for i in range(args.num_users):
            # client_list[i].update_weight_conv(server.get_conv_weight())
            # 冻结分类器，每个客户端进行特征提取器本地训练
            loss_locals = []
            w, low_freq_spectrum, loss = client_list[i].train_convs(low_freq_spectrum_glob)

            low_freq_spectrum_list[i] = copy.deepcopy(low_freq_spectrum)
            w_local_list[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        # 聚合低频频谱
        low_freq_spectrum_glob = server.agg_spec(low_freq_spectrum_list)

        # 聚合特征提取器
        w_avg = server.agg(w_local_list)
        server.update_weight(w_avg)

        server.net.train()
        optimizer = torch.optim.SGD(server.net.parameters(), lr=args.lr, momentum=args.momentum)
        server.net.freeze_feature_extractor()  # 冻结特征提取器

        # 训练全局分类器
        for g_e in range(glob_ep):
            batch_loss = []
            epoch_loss = []
            T = 2
            for batch_idx, (images, labels) in enumerate(server.dataset_train):
                ouput_list = []
                server.net.zero_grad()  # 清除梯度
                images, labels = images.to(args.device), labels.to(args.device)
                output = server.net(images)
                base_loss = server.loss_func(output, labels)

                for i in range(args.num_users):
                    client_list[i].net.eval()
                    ouput_list.append(client_list[i].net(images))

                output_avg = torch.mean(torch.stack(ouput_list), dim=0)

                # 本地模型对全局模型的知识蒸馏
                g_probs = F.log_softmax(output/T, dim=0)
                avg_probs = F.softmax(output_avg/T, dim=0)
                dist_loss = server.loss_func_kl(g_probs, avg_probs)

                loss = base_loss + dist_loss
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('glob Epoch-fc: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        g_e, batch_idx * len(images), len(server.dataset_train.dataset),
                              100. * batch_idx / len(server.dataset_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        server.net.unfreeze_feature_extractor()
        print(f'GlobEpochLoss-fc:{sum(epoch_loss) / len(epoch_loss)}')

        for i in range(args.num_users):
            # 下发特征提取器
            client_list[i].update_weight_ExFeature(server.get_conv_weight())
            # 训练分类器
            w, loss = client_list[i].train_fc(server.net)

            client_list[i].eval()

        # 聚合分类器
        # w_avg = server.agg(w_local_list)
        # server.update_weight(w_avg)

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

