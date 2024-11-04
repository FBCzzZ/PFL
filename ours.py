import copy
import torch
from options import args_parser
from collections import defaultdict
from models.client.clientour import Clientour
from models.server.serverour import Serverour


client_list = []
ExFeature_w_local = []
cla_w_local = []
E_list = []
V_list = []
low_freq_spectrum_g = None


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

    for c in range(args.epochs):
        low_freq_spectrum_l = []
        global_correct_per_label = defaultdict(int)
        global_total_per_label = defaultdict(int)
        for i in range(args.num_users):
            # 冻结分类器，每个客户端进行特征提取器本地训练
            loss_locals = []
            low_freq_spectrum, w, loss = client_list[i].train_convs(low_freq_spectrum_g)
            low_freq_spectrum_l.append(low_freq_spectrum)

            ExFeature_w_local[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

            # 更新全局分布
            server.update_distributions(client_list[i].feature_distributions)

            _, _ , correct_per_label, total_per_label = client_list[i].eval()

            for label in total_per_label:
                global_total_per_label[label] += total_per_label[label]
                global_correct_per_label[label] += correct_per_label[label]

        low_freq_spectrum_g = server.agg_spec(low_freq_spectrum_l)

        # 训练公共头部分类器
        server.train_fc(global_correct_per_label, global_total_per_label)

        for i in range(args.num_users):
            # 下发共享头部
            client_list[i].update_weight_classifier(server.get_cla_weight())

        server.eval()

    # 保存模型
    for i in range(args.num_users):
        client_list[i].save()

