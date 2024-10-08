import numpy as np
import copy
import torch
from models.Nets import CNN

class Server(object):
    def __init__(self, args):
        if args.model == 'cnn':
            self.net = CNN().to(args.device)
        else:
            exit('Error: unrecognized model')
        print(self.net)

    @staticmethod
    def agg(w_list):
        w_avg = {}
        for key in w_list[0].keys():
            # 对每个参数进行求平均
            w_avg[key] = torch.mean(torch.stack([state_dict[key] for state_dict in w_list]), dim=0)
        return w_avg

    def agg_spec(self, w_FFT_list):
        w_FFT_np = np.vstack(w_FFT_list)
        w_FFT_glob = np.mean(w_FFT_np, axis=0)
        return w_FFT_glob

    def get_weight(self):
        return self.net.state_dict()

    def get_conv_weight(self):
        # 提取卷积层的权重
        conv_weights = []
        for name, param in self.net.named_parameters():
            if 'conv' in name or 'fc1' in name:  # 筛选出卷积层权重
                conv_weights.append(param.data)
        return conv_weights

    def update_weight(self, weight):
        self.net.load_state_dict(weight)
