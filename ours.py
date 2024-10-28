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
client_class = []
E_list = []
V_list = []

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

        for i in range(args.num_users):
            # 冻结分类器，每个客户端进行特征提取器本地训练
            loss_locals = []
            w, loss = client_list[i].train_convs(server)

            ExFeature_w_local[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

            E_list.append(client_list[i].E)
            V_list.append(client_list[i].V)

        server.agg_E_V(E_list, V_list)

        for i in range(args.num_users):
            # 下发特征提取器
            client_list[i].update_weight_ExFeature(server.get_ExFeature_weight())
            # 训练分类器
            w, loss = client_list[i].train_fc(server, copy.deepcopy(client_list))
            cla_w_local[i] = copy.deepcopy(w)
            client_list[i].eval()


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

