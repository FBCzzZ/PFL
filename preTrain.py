import copy
import torch

from options import args_parser
from models.Nets import CNN
from models.client.clientavg import Client

data_list = ['mnist', 'usps', 'svhn', 'syn']
local_ep_list = [50, 150, 50, 20]
client_list = []
model = None


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model
    if args.model == 'cnn':
        model = CNN().to(args.device)
    else:
        exit('Error: unrecognized model')

    for i in range(args.num_users):
        client_list.append(Client(args, data_list[i], copy.deepcopy(model).to(args.device), local_ep_list[i]))

    for i in range(args.num_users):
        w, loss = client_list[i].train()
        torch.save(w, f'{client_list[i].dataName}_model_state_dict.pth')
        client_list[i].eval()



