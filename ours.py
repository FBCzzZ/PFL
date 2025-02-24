import copy
import torch
import numpy as np
from options import args_parser
from collections import defaultdict
from models.client.clientour import Clientour
from models.server.serverour import Serverour


client_list = []
ExFeature_w_local = []
E_list = []
V_list = []
low_freq_spectrum_g = None

acc_total = []
loss_total = []

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

    for c in range(args.epochs):
        low_freq_spectrum_l = []
        cla_w_local = []
        global_correct_per_label = defaultdict(int)
        global_total_per_label = defaultdict(int)

        acc_list = []
        loss_list = []
        for i in range(args.num_users):
            # 冻结分类器，每个客户端进行特征提取器本地训练
            client_list[i].update_weight_ExFeature(server.get_ExFeature_weight()) # 下发公共特征提取器
            loss_locals = []
            low_freq_spectrum, loss = client_list[i].train_convs(low_freq_spectrum_g)
            low_freq_spectrum_l.append(low_freq_spectrum)

            ExFeature_w_local[i] = copy.deepcopy(client_list[i].get_ExFeature_weight())
            loss_locals.append(copy.deepcopy(loss))

            cla_w_local.append(copy.deepcopy(client_list[i].get_cla_weight()))

            # 更新全局分布
            server.update_distributions(client_list[i].feature_distributions)

            test_acc, test_loss , correct_per_label, total_per_label = client_list[i].eval()

            acc_list.append(test_acc) # 测试准确率
            loss_list.append(test_loss) # 测试损失

            for label in total_per_label:
                global_total_per_label[label] += total_per_label[label]
                global_correct_per_label[label] += correct_per_label[label]

        acc_total.append(acc_list)
        loss_total.append(loss_list)

        low_freq_spectrum_g = server.agg_spec(low_freq_spectrum_l)


        glob_weight_ExFeature = server.average_weights_dict(ExFeature_w_local)
        server.update_weight_ExFeature(glob_weight_ExFeature)

        server.agg_cla_mom(cla_w_local)

        pseudo_features = server.generate_feature(global_correct_per_label, global_total_per_label)
        # 将伪特征和标签转换为训练数据
        X = torch.cat(list(pseudo_features.values()), dim=0)  # 所有伪特征
        y = torch.cat([torch.full((features.size(0),), label) for label, features in pseudo_features.items()])  # 标签

        # 创建数据集和数据加载器
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        acc_list = []
        loss_list = []
        for i in range(args.num_users):
            # 训练个性化头部分类器
            client_list[i].train_fc(dataloader, server)
            if c == args.epochs-1:
                test_acc, test_loss, _, _ = client_list[i].eval()

                acc_list.append(test_acc)  # 测试准确率
                loss_list.append(test_loss)  # 测试损失
        if len(acc_list):
            acc_total.append(acc_list)
            loss_total.append(loss_list)
        server.eval()

    np.save("/kaggle/working/acc_total.npy", np.array(acc_total))
    np.save("/kaggle/working/loss_total.npy", np.array(loss_total))

    # 保存模型
    for i in range(args.num_users):
        client_list[i].save()

