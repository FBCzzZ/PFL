import copy
import torch
from options import args_parser
from models.client.clientour import Clientour
from models.server.serverour import Serverour
import torch.nn.functional as F


glob_ep = 1
client_list = []
ExFeature_w_local = []
cla_w_local = []
low_freq_spectrum_list = []
low_freq_spectrum_glob = None
client_class = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化服务器
    server = Serverour(args)

    # copy weights
    w_glob = server.get_weight()

    # 初始化客户端
    for i in range(args.num_users):
        client_list.append(Clientour(args, i, copy.deepcopy(server.net).to(args.device), 1))
        ExFeature_w_local.append(None)
        cla_w_local.append(None)
        low_freq_spectrum_list.append(None)

    for c in range(args.epochs):
        for i in range(args.num_users):
            # 下发特征提取器
            client_list[i].update_weight_ExFeature(server.get_ExFeature_weight())
            # 训练分类器
            w, loss = client_list[i].train_fc(server.net)

            cla_w_local[i] = copy.deepcopy(w)

        for i in range(args.num_users):
            # client_list[i].update_weight_classifier(server.get_cla_weight())
            # 冻结分类器，每个客户端进行特征提取器本地训练
            loss_locals = []
            w, low_freq_spectrum, loss = client_list[i].train_convs(low_freq_spectrum_glob)

            low_freq_spectrum_list[i] = copy.deepcopy(low_freq_spectrum)
            ExFeature_w_local[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

            client_list[i].eval()

        # 聚合低频频谱
        low_freq_spectrum_glob = server.agg_spec(low_freq_spectrum_list)


        # server.net.train()
        # optimizer = torch.optim.SGD(server.net.parameters(), lr=args.lr, momentum=args.momentum)
        # server.net.freeze_feature_extractor()  # 冻结特征提取器

        # 训练全局分类器
        # for g_e in range(glob_ep):
        #     batch_loss = []
        #     epoch_loss = []
        #     T = 2
        #     for batch_idx, (images, labels) in enumerate(server.dataset_train):
        #         ouput_list = []
        #         server.net.zero_grad()  # 清除梯度
        #         images, labels = images.to(args.device), labels.to(args.device)
        #         output = server.net(images)
        #         base_loss = server.loss_func(output, labels)
        #
        #         for i in range(args.num_users):
        #             client_list[i].net.eval()
        #             ouput_list.append(client_list[i].net(images))
        #
        #         output_avg = torch.mean(torch.stack(ouput_list), dim=0)
        #
        #         # 本地模型对全局模型的知识蒸馏
        #         g_probs = F.log_softmax(output/T, dim=0)
        #         avg_probs = F.softmax(output_avg/T, dim=0)
        #         dist_loss = server.loss_func_kl(g_probs, avg_probs)
        #
        #         loss = base_loss + dist_loss
        #         loss.backward()
        #         optimizer.step()
        #
        #         if batch_idx % 10 == 0:
        #             print('glob Epoch-fc: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 g_e, batch_idx * len(images), len(server.dataset_train.dataset),
        #                       100. * batch_idx / len(server.dataset_train), loss.item()))
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
        #
        # server.net.unfreeze_feature_extractor()
        # print(f'GlobEpochLoss-fc:{sum(epoch_loss) / len(epoch_loss)}')


        # 聚合特征提取器
        w_avg_ExFeature = server.average_weights_dict(ExFeature_w_local)
        server.update_weight_ExFeature(w_avg_ExFeature)

        # 聚合分类器
        w_avg_cla = server.agg_cla_Momentum(cla_w_local)
        server.update_weight_classifier(w_avg_cla)


        server.eval()

    # 保存模型
    for i in range(args.num_users):
        client_list[i].save()

