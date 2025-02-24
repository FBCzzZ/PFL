import copy
import torch
import numpy as np
from options import args_parser
from models.client.clientfedrep import Clientfedrep
from models.server.serverfedrep import Serverfedrep


client_list = []
ExFeature_w_local = []
cla_w_local = []
acc_total = []
loss_total = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化服务器
    server = Serverfedrep(args)

    # copy weights
    w_glob = server.get_weight()

    # 初始化客户端
    for i in range(args.num_users):
        client_list.append(Clientfedrep(args, i, copy.deepcopy(server.net).to(args.device), 1))
        ExFeature_w_local.append(None)
        cla_w_local.append(None)


    for c in range(args.epochs):
        acc_list = []
        loss_list = []
        for i in range(args.num_users):
            # 下发特征提取器
            client_list[i].update_weight_ExFeature(server.get_ExFeature_weight())
            # 训练分类器
            w, loss = client_list[i].train_fc()

            cla_w_local[i] = copy.deepcopy(w)

        for i in range(args.num_users):
            # 冻结分类器，每个客户端进行特征提取器本地训练
            loss_locals = []
            w, loss = client_list[i].train_convs()

            ExFeature_w_local[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

            test_acc, test_loss = client_list[i].eval()

            acc_list.append(test_acc)  # 测试准确率
            loss_list.append(test_loss)  # 测试损失
        acc_total.append(acc_list)
        loss_total.append(loss_list)
        # 聚合特征提取器
        w_avg = server.average_weights_dict(ExFeature_w_local)
        server.update_weight_ExFeature(w_avg)


        server.eval()

    np.save("/kaggle/working/acc_total.npy", np.array(acc_total))
    np.save("/kaggle/working/loss_total.npy", np.array(loss_total))
    # 保存模型
    for i in range(args.num_users):
        client_list[i].save()

