import copy
import torch

from options import args_parser
from models.client.clientavg import Clientavg
from models.server.serveravg import Serveravg


client_list = []
w_local_list = []
net_glob = None


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 初始化服务器
    server = Serveravg(args)

    # copy weights
    w_glob = server.get_weight()

    server.net.train()

    for i in range(args.num_users):
        client_list.append(Clientavg(args, i, copy.deepcopy(server.net).to(args.device), 1))
        w_local_list.append(w_glob)

    for e in range(args.epochs):
        for i in range(args.num_users):
            client_list[i].update_weight(w_glob)

            loss_locals = []
            w, loss = client_list[i].train()
            w_local_list[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

            client_list[i].eval()

        # update global weights
        w_glob = server.average_weights_dict(w_local_list)

        # copy weight to net_glob
        server.update_weight(w_glob)

        server.eval()

    # torch.save(w_glob, 'model_state_dict.pth')
    torch.save(w_glob, '/kaggle/working/model_state_dict.pth')
